import numpy as np
import psutil
import random
import subprocess as sp
import torch
import uuid

GB = 1 << 30
MB = 1 << 20


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_gpu_memory_usage(gpu: int = 0):
    """
    Python equivalent of nvidia-smi, copied from https://stackoverflow.com/a/67722676
    and verified as being equivalent ✅
    """
    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"

    try:
        memory_use_info = output_to_list(
            sp.check_output(COMMAND.split(), stderr=sp.STDOUT)
        )[1:]

    except sp.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )

    return int(memory_use_info[gpu].split()[0])


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


import requests
import json
def mnserver_query_instant(query):
    base_url = 'http://219.222.20.79:31362/admin-api/ai/k8s-monitor/grafana/query'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "query": query,
        "type": 0
    }
    # 发送http请求
    response = requests.post(
        base_url,
        headers=headers,
        json=data,
        timeout=5
    )
    # 处理响应
    if response.status_code == 200:
        res = response.json()
        return json.loads(res["data"])
    elif response.status_code == 404:
        print("资源未找到")
        return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

def sort_gpu_data(fetched):
    res = {}
    if fetched['status'] != 'success':
        print('Fetched Not Success')
        return None
    for node in fetched['data']['result']:
        _, ram_in_bytes = node['value']
        res[node['metric']['instance']] = int(ram_in_bytes)
    return res

def profile_vram(devices):
    # 接口调用参数
    instances = "|".join(devices)
    free_vram_query = f'ram_kB{"{"}instance=~"({instances})", statistic="free"{"}"}'
    cached_vram_query = f'ram_kB{"{"}instance=~"({instances})", statistic="cached"{"}"}'
    # 获取显存数据并格式化
    fetched = mnserver_query_instant(free_vram_query)
    free_vram = sort_gpu_data(fetched)
    fetched = mnserver_query_instant(cached_vram_query)
    cached_vram = sort_gpu_data(fetched)
    # 返回数据
    return {key: free_vram[key] + cached_vram[key] for key in free_vram}