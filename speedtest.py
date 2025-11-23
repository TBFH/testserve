from testserve.llm import TestOfflineLLM, SamplingParams, TestOfflineLLM_BS1
from testserve.csv_appender import append_to_csv, sum_up_lines

model_path = "/mnt/Data/austin/hf_models/opt-1.3b"
# model_path = "/mnt/Data/austin/hf_models/Llama-2-7b-chat-hf"
# model_path = "/mnt/Data/austin/hf_models/Meta-Llama-3-8B-Instruct"
max_input_tokens = 32
max_output_tokens = 32

import json
with open("/home/austin/repos/FastServe/prompts_20.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
prompts = data.get('test_data', [])
prompts = [' '.join(prompt.split()[:max_input_tokens]) for prompt in prompts]

req_num = 3

def swift_transformer():
    '''
    SwiftTransformer Metrics Testing
    '''
    global prompts
    prompts = prompts[:req_num]

    sampling_params = SamplingParams(
        temperature=0, top_p=1.0, max_tokens=max_output_tokens, stop=[]
    )
    # llm = TestOfflineLLM(
    #     model=model_path,
    #     tensor_parallel_size=1,
    #     pipeline_parallel_size=4,
    #     pipeline_distribution=[10,10,2,2],
    #     gpu_memory_utilization=0.01
    # )
    llm = TestOfflineLLM_BS1(
        model=model_path,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
        pipeline_distribution=[10,10,2,2],
        gpu_memory_utilization=0.01,
        max_batch_size=1
    )
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.get_response()
        output_len = output.get_output_len()
        print(
            f"Request {output.request_id}, Prompt: {prompt!r}, Generated text: {generated_text!r} ({output_len} tokens generated)."
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_length', type=int, default=32)
    parser.add_argument('-o', '--output_length', type=int, default=32)
    parser.add_argument('-r', '--req_num', type=int, default=4)
    args = parser.parse_args()

    max_input_tokens = args.input_length
    max_output_tokens = args.output_length
    req_num = args.req_num

    swift_transformer()