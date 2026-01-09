import threading
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict

# ====================== 基础数据结构定义 ======================
@dataclass
class NodeState:
    """宿主机节点状态（核心：资源状态+分配状态）"""
    node_id: str          # 节点ID
    cpu_total: int = 100  # 总CPU资源（简化为整数）
    mem_total: int = 100  # 总内存资源
    cpu_used: int = 0     # 已使用CPU
    mem_used: int = 0     # 已使用内存
    is_allocating: bool = False  # 是否正在被分配
    queue_count: int = 0         # 排队分配计数
    load_level: str = "low"      # 负载等级：low/medium/high/extreme

@dataclass
class SchedulerRequest:
    """调度请求"""
    req_id: str
    cpu_need: int  # 请求需要的CPU
    mem_need: int  # 请求需要的内存

# ====================== 基线方案（Baseline） ======================
class BaselineScheduler:
    """传统调度器：仅基于资源剩余量选择节点，无并发状态感知"""
    def __init__(self, nodes: List[NodeState]):
        self.nodes = nodes
        self.conflict_count = 0  # 分配冲突次数
        self.retry_count = 0     # 重试次数
        self.success_count = 0   # 成功分配次数

    def filter_nodes_by_resource(self, req: SchedulerRequest) -> List[NodeState]:
        """仅过滤资源满足的节点"""
        return [
            node for node in self.nodes
            if (node.cpu_total - node.cpu_used) >= req.cpu_need
            and (node.mem_total - node.mem_used) >= req.mem_need
        ]

    def allocate(self, req: SchedulerRequest) -> bool:
        """分配逻辑：随机选资源满足的节点，无锁保护，易冲突"""
        eligible_nodes = self.filter_nodes_by_resource(req)
        if not eligible_nodes:
            return False
        
        # 随机选择一个节点（模拟传统调度器的简单决策）
        selected_node = random.choice(eligible_nodes)
        
        # 模拟并发冲突：多个线程同时修改同一节点
        if selected_node.is_allocating:
            self.conflict_count += 1
            self.retry_count += 1
            return False  # 冲突，返回失败（触发重试）
        
        # 模拟分配过程（加锁等待的开销）
        selected_node.is_allocating = True
        time.sleep(0.001)  # 模拟分配和锁等待耗时
        
        # 完成分配
        selected_node.cpu_used += req.cpu_need
        selected_node.mem_used += req.mem_need
        selected_node.is_allocating = False
        self.success_count += 1
        return True

# ====================== 优化方案（基于分配状态） ======================
class OptimizedScheduler:
    """优化调度器：基于共享节点状态表，规避高冲突节点"""
    def __init__(self, nodes: List[NodeState]):
        self.nodes = nodes
        self.node_state_lock = threading.Lock()  # 状态表写锁
        self.conflict_count = 0
        self.retry_count = 0
        self.success_count = 0
        # 调度器侧快表（同步节点状态，简化为直接引用+锁保护）
        self.node_state_table: Dict[str, NodeState] = {n.node_id: n for n in nodes}

    def filter_nodes_by_resource(self, req: SchedulerRequest) -> List[NodeState]:
        """基础资源过滤（同基线）"""
        return [
            node for node in self.nodes
            if (node.cpu_total - node.cpu_used) >= req.cpu_need
            and (node.mem_total - node.mem_used) >= req.mem_need
        ]

    def filter_nodes_by_state(self, eligible_nodes: List[NodeState]) -> List[NodeState]:
        """基于分配状态二次筛选：优先选空闲/低负载/排队少的节点"""
        with self.node_state_lock:  # 读状态表加锁，保证一致性
            return [
                node for node in eligible_nodes
                if node.load_level in ["low", "medium"]  # 排除高负载/极高负载
                and node.queue_count < 2                # 排队计数少
                and not node.is_allocating              # 未被分配
            ]

    def update_node_state(self, node: NodeState, success: bool, req: SchedulerRequest):
        """分配后更新节点状态"""
        with self.node_state_lock:
            if success:
                # 成功：先更新预估负载，再更新准确负载（简化为直接更新）
                node.cpu_used += req.cpu_need
                node.mem_used += req.mem_need
                # 根据使用率更新负载等级
                usage = (node.cpu_used / node.cpu_total)
                if usage < 0.3:
                    node.load_level = "low"
                elif usage < 0.7:
                    node.load_level = "medium"
                else:
                    node.load_level = "high"
            else:
                # 失败：标记为极高负载，避免后续分配
                node.load_level = "extreme"
            
            # 减少排队计数
            node.queue_count = max(0, node.queue_count - 1)
            node.is_allocating = False

    def allocate(self, req: SchedulerRequest) -> bool:
        """优化的分配逻辑"""
        # 1. 基础资源过滤
        eligible_nodes = self.filter_nodes_by_resource(req)
        if not eligible_nodes:
            return False
        
        # 2. 基于分配状态二次筛选
        state_eligible_nodes = self.filter_nodes_by_state(eligible_nodes)
        # 若二次筛选无节点，降级为基础筛选结果（避免无节点可选）
        selected_candidates = state_eligible_nodes if state_eligible_nodes else eligible_nodes
        if not selected_candidates:
            return False
        
        # 3. 选择节点并尝试分配
        selected_node = random.choice(selected_candidates)
        with self.node_state_lock:
            if selected_node.is_allocating:
                # 冲突：标记为极高负载，避免后续冲突
                selected_node.load_level = "extreme"
                self.conflict_count += 1
                self.retry_count += 1
                return False
            
            # 标记为正在分配，增加排队计数
            selected_node.is_allocating = True
            selected_node.queue_count += 1
        
        # 模拟分配耗时
        time.sleep(0.001)
        
        # 4. 模拟分配结果（简化为95%成功率）
        allocate_success = random.random() < 0.95
        self.update_node_state(selected_node, allocate_success, req)
        
        if allocate_success:
            self.success_count += 1
        else:
            self.retry_count += 1
        
        return allocate_success

# ====================== 模拟高并发测试 ======================
def simulate_high_concurrency(scheduler, request_num: int = 1000):
    """模拟高并发调度请求"""
    def worker(req: SchedulerRequest):
        # 模拟重试逻辑：最多重试3次
        retry = 0
        while retry < 3:
            if scheduler.allocate(req):
                break
            retry += 1

    # 生成高并发请求（模拟重尾分布：大部分小请求，少量大请求）
    requests = []
    for i in range(request_num):
        if random.random() < 0.8:  # 80%小请求
            cpu = random.randint(5, 10)
            mem = random.randint(5, 10)
        else:  # 20%大请求（易引发高负载）
            cpu = random.randint(20, 30)
            mem = random.randint(20, 30)
        requests.append(SchedulerRequest(req_id=f"req_{i}", cpu_need=cpu, mem_need=mem))
    
    # 启动多线程模拟并发调度器
    threads = []
    start_time = time.time()
    for req in requests:
        t = threading.Thread(target=worker, args=(req,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    end_time = time.time()
    
    # 输出统计结果
    print(f"\n===== {scheduler.__class__.__name__} 统计 =====")
    print(f"总请求数：{request_num}")
    print(f"成功分配数：{scheduler.success_count}")
    print(f"冲突次数：{scheduler.conflict_count}")
    print(f"总重试次数：{scheduler.retry_count}")
    print(f"冲突率：{scheduler.conflict_count/request_num:.2%}")
    print(f"耗时：{end_time - start_time:.2f}秒")

if __name__ == "__main__":
    # 初始化宿主机节点（5个节点）
    nodes = [NodeState(node_id=f"node_{i}") for i in range(5)]
    
    # 测试基线方案
    baseline_scheduler = BaselineScheduler(nodes.copy())
    simulate_high_concurrency(baseline_scheduler)
    
    # 重置节点状态，测试优化方案
    nodes = [NodeState(node_id=f"node_{i}") for i in range(5)]
    optimized_scheduler = OptimizedScheduler(nodes)
    simulate_high_concurrency(optimized_scheduler)