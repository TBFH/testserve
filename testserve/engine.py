import time
import copy
from typing import List, Optional
import asyncio

import ray
# from fastserve.ray_utils import ClusterMonitor, check_cluster_status
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
import torch

from testserve.config import ModelConfig, ParallelConfig, CacheConfig, SchedConfig
from testserve.logger import init_logger
from testserve.request import (
    SamplingParams,
    SamplingParams,
    Request,
    create_request,
)
from testserve.worker import ParaWorker
from testserve.tokenizer import get_tokenizer
from testserve.scheduler import get_scheduler
from testserve.utils import Counter, profile_vram, get_gpu_memory, GB, MB
from testserve.block_manager import BlockManager

# 配置相关环境变量，防止通信问题导致的程序卡死
# import os
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'
# os.environ['NCCL_SOCKET_IFNAME'] = 'eno4'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

logger = init_logger(__name__)

@ray.remote(num_gpus=1)
def resource_inspect():
    # GPU Overall Inspect
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    device_name = device.name()
    context = device.make_context()
    total_memory = device.total_memory() / (1024 ** 2)
    free_memory = cuda.mem_get_info()[0] / (1024 ** 2)
    used_memory = total_memory - free_memory
    context.pop()
    return {
        "GPU_Name": device_name,
        "Total_VRAM": total_memory,
        "Used_VRAM": used_memory,
        "Free_VRAM": free_memory,
    }

class StepOutput:
    """The output of request in one step of inference.
    It contains the information of corresponding request and the generated tokens until this step.
    """

    def __init__(self, request: Request, new_token: str, new_token_id: int):
        self.request = request
        self.request_id = request.request_id
        self.prompt = request.prompt
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = request.is_finished

    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )


class LLMEngine:
    """An LLMEngine launches the model executor workers and maintains runtime information.
    It receives requests from upper wrapper class and provides interface LLMEngine.step()
    to execute one iteration inference on a batch of requests chosen by the scheduler.

    Note: Users may not use LLMEngine directly, but use more user-friendly wrapper classes
    OfflineLLM and AsyncLLM instead.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: SchedConfig,
        enable_records: bool,
        deployments: List[str],
        pre_benchmark_mode: bool
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.sched_config = sched_config
        self.deployments = deployments

        self.request_counter = Counter()
        self.step_counter = Counter()
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )

        self.placement_groups = None
        # stages[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.stages = []

        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline = []
        self.batches_ret_futures = []
        self.node_resources = {}
        self.device_map = {}
        self.enable_records = enable_records
        self.pre_benchmark_mode = pre_benchmark_mode
        self.failset = set()
        self.logs = ""
        self.min_gpu_blocks = -1

        # initialization
        self._init_placement_groups()
        self._init_workers()
        self._init_model()
        self.num_gpu_blocks, self.num_cpu_blocks = self._init_kvcache()
        self._gpu_usage_summary()

        self.block_manager = BlockManager(
            self.num_gpu_blocks,
            self.num_cpu_blocks,
            model_config,
            parallel_config,
            cache_config,
            self.remote_call_all_workers_async,
        )
        self.scheduler = get_scheduler(
            sched_config,
            parallel_config,
            self.block_manager,
        )
        logger.info(self.scheduler)
        logger.info(f"{self.block_manager}")

    def _device_nodeid_mapping(self):
        nodes = ray.nodes()
        for node in nodes:
            if node['Alive']:
                node_id = node['NodeID']
                resources = list(node['Resources'].keys())
                for el in resources:
                    if el.startswith('device'):
                        device = el.split(':',1)[1]
                self.device_map[node_id] = device

    def _init_inspect(self):
        nodes = ray.nodes()
        futures = []
        for i, node in enumerate(nodes):
            if node['Alive']:
                node_id = node['NodeID']
                future = resource_inspect.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote()
                futures.append((node_id, future))
        # Save & Print out Inspects Data
        for idx, (node_id, future) in enumerate(futures):
            result = ray.get(future)
            self.node_resources[node_id] = result
            outlog = ''
            outlog += f'[Worker{idx}] NodeID: {node_id}\n'
            outlog += f'[Worker{idx}] Device: {self.device_map[node_id]}\n'
            outlog += f'[Worker{idx}] Used/Total VRAM: {result["Used_VRAM"]/1024:.1f}/{result["Total_VRAM"]/1024:.1f} GB ({(result["Used_VRAM"]/result["Total_VRAM"])*100:.1f}%)\n'
            outlog += f'[Worker{idx}] Free VRAM: {result["Free_VRAM"]/1024:.1f} GB\n'
            # print(outlog)
    
    def _gpu_usage_summary(self):
        futures = self.remote_call_all_workers_async("resource_inspect")
        # Save & Print out Inspects Data
        for future in (futures):
            result = ray.get(future)
            node_id = result["NodeID"]
            # print(f'[{node_id}] GPU Device {self.node_resources[node_id]["GPU_Name"]} Allocated VRAM: {allocated_vram:.2f} MiB ------> [Rank {result["Rank"]}] with {result["Num_Layers"]} Layers')
            print(
                f'[{node_id}] GPU Device {self.device_map[node_id]} ------> [Rank {result["Rank"]}] with {result["Num_Layers"]} Layers'
            )
        # if len(self.deployments) > 0:
        #     device_map_rvt = {v:k for k,v in self.device_map.items()}
        #     for idx, node in enumerate(self.deployments):
        #         if node not in device_map_rvt:
        #             raise RuntimeError(f"Node [{node}] not found in cluster")
        #         print(f'[{device_map_rvt[node]}] GPU Device {node} ------> [Rank {idx}] with {self.parallel_config.pipeline_distribution[idx]} Layers')

    def _init_placement_groups(self):
        if not ray.is_initialized():
            ray.init(
                # include_dashboard=False
                address="ray://219.222.20.79:30807"
            )
        self._device_nodeid_mapping()
        self._init_inspect()
        num_cluster_gpus = ray.cluster_resources().get("GPU", 0)
        if (
            num_cluster_gpus
            < self.parallel_config.tensor_parallel_size
            * self.parallel_config.pipeline_parallel_size
        ):
            raise ValueError("No GPU resources available")

        # self.placement_groups = []

        # # for each pipeline stage, create a placement group with tensor_parallel_size GPUs
        # # the 'PACK' strategy will pack the GPUs into one node
        # for _ in range(self.parallel_config.pipeline_parallel_size):
        #     placement_group = ray.util.placement_group(
        #         [
        #             {
        #                 "GPU": 1,
        #             }
        #         ]
        #         * self.parallel_config.tensor_parallel_size,
        #         strategy="PACK",
        #     )
        #     ray.get(placement_group.ready(), timeout=1000)
        #     self.placement_groups.append(placement_group)
        #     logger.info(
        #         f"creating placement group with {self.parallel_config.tensor_parallel_size} GPUs"
        #     )

    def _init_workers(self):
        """
        for each pipeline stage, create tensor_parallel_size workers
        each worker will be assigned a GPU
        the worker will be placed in the corresponding placement group
        """
        # self.pp_id = ray.put(copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id()))
        # wait until pp_id is ready
        # ray.get(self.pp_id)
        # init_handlers = []
        # for i in range(self.parallel_config.pipeline_parallel_size):
        #     workers = []
        #     # tp_id = ray.put(copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id()))
        #     # wait until tp_id is ready
        #     # ray.get(tp_id)
        #     for j in range(self.parallel_config.tensor_parallel_size):
        #         tmp_parallel_config = copy.deepcopy(self.parallel_config)
        #         tmp_parallel_config.pipeline_parallel_rank = i
        #         tmp_parallel_config.tensor_parallel_rank = j
        #         worker = ParaWorker.options(
        #             scheduling_strategy=PlacementGroupSchedulingStrategy(
        #                 placement_group=self.placement_groups[i],
        #                 placement_group_bundle_index=j,
        #             )
        #         ).remote(
        #             model_config=self.model_config,
        #             cache_config=self.cache_config,
        #             sched_config=self.sched_config,
        #             parallel_config=tmp_parallel_config,
        #             # pipeline_parallel_id=self.pp_id,
        #             # tensor_parallel_id=tp_id,
        #         )
        #         workers.append(worker)
        #         init_handlers.append(worker.ready.remote())
        #     self.stages.append(workers)
        # # Ray will block until all workers are ready
        # ray.get(init_handlers)

        deployment = []
        if len(self.deployments) > 0:
            device_map_rvt = {v:k for k,v in self.device_map.items()}
            for node in self.deployments:
                if node not in device_map_rvt:
                    raise RuntimeError(f"Node [{node}] not found in cluster")
                deployment.append(device_map_rvt[node])
        else:
            for node_id, res in self.node_resources.items():
                if res['Free_VRAM'] > 4096 and 'jetson' in self.device_map[node_id]:
                    if 'jetson-64g' in self.device_map[node_id]:
                        deployment.insert(0, node_id)
                    else:
                        deployment.append(node_id)
        
        init_handlers = []
        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []
            # tp_id = ray.put(copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id()))
            # wait until tp_id is ready
            # ray.get(tp_id)
            for j in range(self.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(self.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j
                worker = ParaWorker.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=deployment[i],
                        soft=False
                    )
                ).remote(
                    model_config=self.model_config,
                    cache_config=self.cache_config,
                    sched_config=self.sched_config,
                    parallel_config=tmp_parallel_config,
                    # pipeline_parallel_id=self.pp_id,
                    # tensor_parallel_id=tp_id,
                    enable_records=self.enable_records,
                    pre_benchmark_mode=self.pre_benchmark_mode
                )
                workers.append(worker)
                init_handlers.append(worker.ready.remote())
            self.stages.append(workers)
        # Ray will block until all workers are ready
        ray.get(init_handlers)


    def _init_model(self):
        """
        init model by call init_model() on all workers
        """
        self.remote_call_all_workers("init_model")

    def _init_kvcache(self):
        # num_gpu_blocks, num_cpu_blocks = ray.get(
        #     self.stages[0][0]._profile_num_available_blocks.remote(
        #         self.cache_config.block_size,
        #         self.cache_config.gpu_memory_utilization,
        #         self.cache_config.cpu_swap_space,
        #     )
        # )
        futures = self.remote_call_all_workers_async(
            "_profile_num_available_blocks",
            self.cache_config.block_size,
            self.cache_config.gpu_memory_utilization,
            self.cache_config.cpu_swap_space,
        )
        outlogs = "[GPU Profiles] \n"
        gpu_blocks = []
        for future in futures:
            result = ray.get(future)
            node_name = self.device_map[result['node_id']]
            gpu_block = result['num_gpu_blocks']
            available_vram = result['available_vram']
            model_vram = result['peak_vram']
            kvcache_vram = available_vram * self.cache_config.gpu_memory_utilization - model_vram
            gpu_blocks.append(gpu_block)
            outlogs += f"{node_name}: \n"
            outlogs += f"\t Block Available: {gpu_block} \n"
            outlogs += f"\t VRAM Available: {((available_vram) / (1024**3)):.2f} GB \n"
            outlogs += f"\t VRAM Utilization Constraint: {self.cache_config.gpu_memory_utilization * 100}% \n"
            outlogs += f"\t VRAM for Model: {((model_vram) / (1024**3)):.2f} GB \n"
            outlogs += f"\t VRAM for KVCache: {((kvcache_vram) / (1024**3)):.2f} GB \n"
        print(outlogs)
        self.logs += outlogs[:-1]
        print("gpu_blocks: ", gpu_blocks)
        # gpu_blocks[-1] = int(gpu_blocks[-1] * 0.3)
        num_gpu_blocks = min(gpu_blocks)
        self.min_gpu_blocks = num_gpu_blocks
        num_cpu_blocks = 1
        # num_gpu_blocks, num_cpu_blocks = (3000, 1)
        print(f"num_gpu_blocks: {num_gpu_blocks}, num_cpu_blocks: {num_cpu_blocks}")
        self.remote_call_all_workers(
            "init_kvcache_and_swap", num_gpu_blocks, num_cpu_blocks
        )
        return num_gpu_blocks, num_cpu_blocks
        
        # available_gpu_memory = profile_vram(self.deployments)
        # peak_runtime_memory = (
        #     available_gpu_memory * 0.05
        #     + self.model_config.get_model_size_in_bytes(
        #         parallel_config=self.parallel_config
        #     )
        # )


    def remote_call_all_workers(self, func_name: str, *args):
        """
        call func_name on all workers, blocked until all workers finish, return all the results
        """
        handlers = []
        for stage in self.stages:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args))
        return ray.get(handlers)

    def remote_call_all_workers_async(self, func_name: str, *args):
        """
        call func_name asynchronously on all workers, return the futures immediately
        """
        handlers = []
        for stage in self.stages:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args))
        return handlers
    
    def remote_forward_async(self, *args):
        """
        call all worker's forward asynchronously and transmit intermediate results between workers
        """
        intermed = None
        for stage in self.stages:
            for worker in stage:
                intermed = worker.step.remote(*args, intermed)
        return intermed
    
    def collect(self):
        import requests
        try:
            response = requests.post("http://219.222.20.79:31063/collect", timeout=5)
            if response.status_code == 200:
                records = response.json()
            else:
                print(f"Collect Failed, State Code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Collect Failed: {e}")
            return None
        # records = self.remote_call_all_workers('submit_records')
        all_records = []
        for stage_record in records:
            counter = {}
            for record in stage_record:
                rid = record["req_id"]
                if rid not in counter:
                    counter[rid] = 1
                else:
                    counter[rid] += 1
                record["desc"] = counter[rid]
                all_records.append(record)
        # 写入csv表格
        import csv
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(os.path.expanduser('~'), f"stats/uneven_partition/pp-records_{timestamp}.csv")
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory): # 如果目录不存在，则创建（包括所有中间目录）
            os.makedirs(directory, exist_ok=True)
        fieldnames = list(all_records[0].keys())
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"PP_Gantte saved to {filename}")

    def collect_prebenchmarks(self):
        res = []
        futures = self.remote_call_all_workers_async("get_prebenchmarks")
        for fut in futures:
            reslut = ray.get(fut)
            res.append(reslut)
        return res

    def add_request(
        self,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[str]],
        sampling_params: SamplingParams,
        arrival_time: Optional[float] = None,
        request_id: Optional[int] = None,
    ):
        req = create_request(
            prompt,
            prompt_token_ids,
            sampling_params,
            self.request_counter,
            self.tokenizer,
            arrival_time,
            request_id,
        )
        self.scheduler.add_request(req)

    def warmup(self):
        self.remote_call_all_workers_async('toggle_record', False)
        # remote_call = self.remote_forward_async([], [], [], [])
        remote_call = self.remote_forward_async(
            [1, 2],
            [[21, 22], [21, 22]],
            [0, 0],
            [[0], [1]]
        )
        self.remote_call_all_workers_async('toggle_record', self.enable_records)
        ray.get(remote_call)

    def step(self):
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """
        # print block usage every 100 steps
        if next(self.step_counter) % 100 == 0:
            self.block_manager.print_block_usage()

        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size

        # pick next batch from scheduler
        # this may trigger swap_in if some requests have been swapped out to CPU
        # this may also trigger swap_out if GPU blocks are not enough
        batched_requests = self.scheduler.get_next_batch()
        batched_requests.sort_requests_context_first()
        # logger.info(f"batched_requests: {batched_requests}")

        # allocate blocks as needed
        failed_reqs = self.block_manager.allocate_blocks_batched(batched_requests)
        if len(failed_reqs) > 0:
            # 通知各worker释放失败请求的所有KVCache显存占用
            self.remote_call_all_workers_async(
                "clear_request_resource_batched", failed_reqs
            )
            # 失败请求重新加入等待队列等待处理
            for req in failed_reqs:
                # new_req = create_request(
                #     prompt=req.prompt,
                #     prompt_token_ids=None,
                #     sampling_params=req.sampling_params,
                #     request_counter=self.request_counter,
                #     tokenizer=self.tokenizer,
                #     arrival_time=req.arrival_time,
                #     request_id=req.request_id
                # )
                req.prompt = req.prompt + ' ' + ' '.join(req.generated_tokens)
                req.prompt_token_ids += req.generated_token_ids
                req.generated_tokens = []
                req.generated_token_ids = []
                req.is_finished = False
                req.is_running = False
                self.scheduler.add_request(req)
                self.failset.add(req.request_id)    # 记录失败请求

        # Check if all requests are on GPU (i.e. not swapped out)
        assert self.block_manager.is_all_requests_on_gpu(
            batched_requests
        ), "Some requests are currently swapped out to CPU"

        # push the batch into pipeline
        batched_requests.start_one_iteration(time.time())
        self.batches_in_pipeline.append(batched_requests)
        # remote_calls = self.remote_call_all_workers_async(
        #     "step",
        #     batched_requests.get_request_ids(),
        #     batched_requests.get_input_tokens_batched(),
        #     batched_requests.get_first_token_indexes(),
        #     self.block_manager.get_partial_block_table(
        #         batched_requests.get_request_ids()
        #     ),
        # )
        remote_call = self.remote_forward_async(
            batched_requests.get_request_ids(),
            batched_requests.get_input_tokens_batched(),
            batched_requests.get_first_token_indexes(),
            self.block_manager.get_partial_block_table(
                batched_requests.get_request_ids()
            ),
        )
        # only the leader of the last stage return valid output, i.e., generated tokens ids
        # self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])
        self.batches_ret_futures.append(remote_call)

        # output buffer
        step_outputs = []
        finished_reqs = []

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            generated_tokens_ids, _ = ray.get(self.batches_ret_futures[0])
            end_time = time.time()

            # LLaMA: add a space to the front for token begins with SPIECE_UNDERLINE("▁")
            if (self.model_config.hf_config.model_type == "llama"):
                SPIECE_UNDERLINE = "▁"
                if generated_tokens_ids != [] and (max(generated_tokens_ids) > self.model_config.hf_config.vocab_size or min(generated_tokens_ids) < 0):
                    print('Warning: generated token id exceeds vocab size', f'generated_tokens_ids: {generated_tokens_ids}')
                    generated_tokens_ids = [max(0, min(x, self.model_config.hf_config.vocab_size - 1)) for x in generated_tokens_ids]
                _tokenlist = self.tokenizer.convert_ids_to_tokens(generated_tokens_ids)
                generated_tokens = []
                for _token in _tokenlist:
                    newstr = self.tokenizer.convert_tokens_to_string([_token,])
                    if (_token.startswith(SPIECE_UNDERLINE)):
                        newstr = " " + newstr
                    generated_tokens.append(newstr)
            else:
                generated_tokens = [
                    self.tokenizer.decode(gen_token_id, skip_special_tokens=True)
                    for gen_token_id in generated_tokens_ids
                ]

            finished_batch = self.batches_in_pipeline[0]
            finished_batch.finish_one_iteration(
                generated_tokens, generated_tokens_ids, end_time
            )
            # logger.info(f"Finished batch: {finished_batch}")

            # construct outputs
            for request, new_token, new_token_id in zip(
                finished_batch.requests, generated_tokens, generated_tokens_ids
            ):
                step_outputs.append(StepOutput(request, new_token, new_token_id))
            finished_reqs = self.scheduler.pop_finished_requests()

            # free blocks for finished requests
            self.block_manager.free_blocks_batched(finished_reqs)
            self.remote_call_all_workers_async(
                "clear_request_resource_batched", finished_reqs
            )

            # pop the finished batch
            self.batches_in_pipeline.pop(0)
            self.batches_ret_futures.pop(0)

        # proactive swapping
        self.scheduler.post_process()

        return step_outputs, finished_reqs

    def get_num_unfinished_requests(self) -> int:
        return self.scheduler.get_total_num_requests()

    def abort_request(self, request_id: int):
        self.scheduler.abort_request(request_id)
        self.block_manager.free_blocks(request_id)
        self.remote_call_all_workers_async("clear_request_resource", request_id)

    def swap_in_request(self, requests: List[Request]):
        self.block_manager.swap_in_requests(requests)

    def swap_out_request(self, requests: List[Request]):
        self.block_manager.swap_out_requests(requests)
