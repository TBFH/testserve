import ray
import time

# ray.init(address="auto")

@ray.remote
class ClusterMonitor:
    def wait_for_nodes_with_timeout(self, expected_nodes, timeout=30):
        """等待指定数量的节点加入，带超时限制"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_nodes = len(ray.nodes())
            if current_nodes >= expected_nodes:
                return True
            print(f"等待节点... 当前: {current_nodes}, 期望: {expected_nodes}")
            time.sleep(2)
        return False
    
    def wait_for_nodes(self, expected_nodes):
        """等待指定数量的节点加入，无超时"""
        while True:
            current_nodes = len(ray.nodes())
            if current_nodes < expected_nodes:
                time.sleep(1)
            else:
                break

# @ray.remote
# def remote_forward_async(stages, *args):
#     intermed = None
#     for stage in stages:
#         for worker in stage:
#             # generated_tokens_ids, intermediate = worker.step.remote(*args, intermed)
#             inter_ref = worker.step.remote(*args, intermed)
#             intermed = inter_ref
#     return inter_ref

def check_cluster_status():
    """检查集群节点状态"""
    nodes = ray.nodes()
    print(f"集群节点数量: {len(nodes)}")
    
    for i, node in enumerate(nodes):
        node_id = node["NodeID"]
        resources = node["Resources"]
        gpu_count = resources.get("GPU", 0)
        status = "活跃" if node["Alive"] else "离线"
        
        print(f"  节点 {i}: {node_id} | GPU: {gpu_count} | 状态: {status}")
        print(f"     资源: {resources}")


