import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from testserve.llm import AsyncLLM
from testserve.request import SamplingParams
from testserve.utils import random_uuid
from testserve.logger import init_logger

# import ray

logger = init_logger(__name__)

model_path = "/mnt/Data/austin/hf_models/opt-1.3b"
# model_path = "/mnt/Data/austin/hf_models/Llama-2-7b-chat-hf"
# model_path = "/mnt/Data/austin/hf_models/Meta-Llama-3-8B-Instruct"

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    logger.info("Received a request.")
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(
        request_id, prompt=prompt, sampling_params=sampling_params
    )

    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for step_output in results_generator:
                text_output = step_output.request.get_response()
                ret = {"text": text_output}
                yield (json.dumps(ret) + "\0").encode("utf-8")

        async def abort_request() -> None:
            await engine.abort(request_id)

        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)
    else:
        # Non-streaming case
        final_output = None
        async for step_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_output = step_output

        assert final_output is not None
        prompt = final_output.prompt
        text_output = prompt + final_output.request.get_response()
        ret = {"text": text_output}
        return JSONResponse(ret)


@app.get("/records")
async def records():
    engine.collect_all_workers_records()
    return {
        "status": "records done!"
    }


@app.get("/prebenchmarks")
async def get_prebenchmarks():
    return engine.collect_all_workers_prebenchmarks()


@app.get("/numfails")
async def get_numfails():
    return engine.get_num_fails()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--host", type=str, default="localhost")
    # parser.add_argument("--port", type=int, default=30320)
    # parser.add_argument("--model", type=str, default="/mnt/Data/austin/hf_models/opt-1.3b")
    # parser.add_argument("--tokenizer", type=str, default=None)
    # parser.add_argument("--trust-remote-code", action="store_true")
    # parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--pipeline-parallel-size", type=int, default=4)
    # parser.add_argument("--tensor-parallel-size", type=int, default=1)
    # parser.add_argument("--block-size", type=int, default=16)
    # parser.add_argument("--max-num-blocks-per-req", type=int, default=256)
    # parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    # parser.add_argument("--swap-space", type=int, default=16)
    # parser.add_argument("--sched-policy", type=str, default="fcfs")
    # parser.add_argument("--max-batch-size", type=int, default=256)
    # parser.add_argument("--max-tokens-per-batch", type=int, default=2048)
    # parser.add_argument("--profiling-file", type=str, default=None)
    # parser.add_argument("--use-dummy-weights", action="store_true")
    # parser.add_argument("--proactive-offloading", action="store_true")
    # parser.add_argument("--num-min-free-blocks-threshold", type=int, default=0)
    # parser.add_argument("--num-queues-for-prediction", type=int, default=2)
    # parser.add_argument("--use-skip-join", action="store_true")
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30320)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--deployments', type=str, default="['jetson-64g-4', 'jetson-16g-2', 'jetson-16g-8', 'jetson-8g-1']")
    parser.add_argument('--pipeline-distribution', type=str, default=None)
    args = parser.parse_args()

    deployments = eval(args.deployments)

    # ray.init()
    # llm = TestOfflineLLM_BS1(
    #     model=model_path,
    #     tensor_parallel_size=1,
    #     pipeline_parallel_size=pp_size,
    #     pipeline_distribution=[llm_config.num_hidden_layers // pp_size for _ in range(pp_size)],
    #     deployments=deployments,
    #     gpu_memory_utilization=0.7,
    #     max_batch_size=batch_size,
    #     enable_records=False,
    #     pre_benchmark_mode=True
    # )

    engine = AsyncLLM(
        model=model_path,
        tensor_parallel_size=1,
        pipeline_parallel_size=len(deployments),
        pipeline_distribution=[6, 6, 6, 6],
        deployments=deployments,
        gpu_memory_utilization=0.8,
        max_batch_size=args.batch_size,
        enable_records=args.record,
        pre_benchmark_mode=False,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
