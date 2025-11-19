import torch

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

req_num = 4
prompts = prompts[:req_num]

def swift_transformer():
    '''
    SwiftTransformer Metrics Testing
    '''

    # append_to_csv(f"/home/austin/tools/utils/stats/SwiftTransformer_{model_path.split('/')[-1]}.csv", new_line=True)
    # append_to_csv(f"/home/austin/tools/utils/stats/SwiftTransformer_{model_path.split('/')[-1]}.csv", data=f"-i {max_input_tokens} -o {max_output_tokens}", new_line=True)

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
    # sum_up_lines(f"/home/austin/tools/utils/stats/SwiftTransformer_{model_path.split('/')[-1]}.csv", len(prompts))



def hf_transformers():
    '''
    Huggingface Transformers Metrics Testing
    '''

    append_to_csv(f"/home/austin/tools/utils/stats/Transformers_{model_path.split('/')[-1]}.csv", new_line=True)
    append_to_csv(f"/home/austin/tools/utils/stats/Transformers_{model_path.split('/')[-1]}.csv", data=f"-i {max_input_tokens} -o {max_output_tokens}", new_line=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16", use_cache=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    for prompt in prompts:
        model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        import time
        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=max_output_tokens,
                use_cache=True
            )
        latency = time.time() - start
        append_to_csv(f"/home/austin/tools/utils/stats/Transformers_{model_path.split('/')[-1]}.csv", data=latency, new_line=True)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(
            f"Prompt: {prompt!r}, Generated text: {response!r} ({max_output_tokens} tokens generated)."
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_length', type=int, default=32)
    parser.add_argument('-o', '--output_length', type=int, default=32)
    args = parser.parse_args()

    max_input_tokens = args.input_length
    max_output_tokens = args.output_length

    swift_transformer()
    # hf_transformers()