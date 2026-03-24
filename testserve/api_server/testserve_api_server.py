import argparse
import json
from typing import AsyncGenerator, List, Tuple
import time

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from testserve.llm import AsyncLLM
from testserve.request import SamplingParams
from testserve.utils import random_uuid
from testserve.logger import init_logger
from testserve.engine import StepOutput

# import ray

logger = init_logger(__name__)

model_path = "/mnt/Data/austin/hf_models/opt-1.3b"
# model_path = "/mnt/Data/austin/hf_models/Llama-2-7b-chat-hf"
# model_path = "/mnt/Data/austin/hf_models/Meta-Llama-3-8B-Instruct"

TIMEOUT_KEEP_ALIVE = 300  # seconds.
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
        final_outputs: List[Tuple[StepOutput, float]] = []   # (step_output, timestamp)
        async for step_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_outputs.append((step_output, time.time()))

        text_output = prompt + ''.join([step_output[0].new_token for step_output in final_outputs])
        ret = {
            "text": text_output,
            "timestamps": [step_output[1] for step_output in final_outputs]
        }
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30320)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument('--deployments', type=str, required=True)
    # parser.add_argument('--pipeline-distribution', type=str, default=None)
    args = parser.parse_args()

    deployments = eval(args.deployments)

    engine = AsyncLLM(
        model=args.model,
        tensor_parallel_size=1,
        pipeline_parallel_size=len(deployments),
        # pipeline_distribution=[6, 6, 6, 6],
        deployments=deployments,
        gpu_memory_utilization=0.8,
        max_batch_size=args.max_batch_size,
        enable_records=False,
        pre_benchmark_mode=False,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
