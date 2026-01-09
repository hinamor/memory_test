# src/models/machine.py
from datetime import timedelta
from typing import List, Tuple

class Machine:
    """机器类（新增任务释放逻辑+用户任务统计+近期任务统计）"""
    def __init__(self, total_cpu: int, total_memory: int, total_disk: int, is_enabled: bool):
        self.total_cpu = total_cpu
        self.used_cpu = 0
        self.free_cpu = total_cpu
        self.total_memory = total_memory    # MB
        self.used_memory = 0
        self.free_memory = total_memory
        self.memory_free_blocks = [total_memory]  # 内存空闲块列表
        self.total_disk = total_disk        # GB
        self.used_disk = 0
        self.free_disk = total_disk
        self.tasks = {}                     # {task_id: task_info}
        self.is_enabled = is_enabled

        self.user_task_count = {}           # {user_id: count}
        self.recent_tasks: List[Tuple[str, 'datetime']] = []

    def allocate_task(self, task_id: str, task_cpu: int, task_memory: int, task_disk: int,
                      user_id: str, submit_time) -> bool:
        if (task_cpu > self.free_cpu or 
            task_memory > self.free_memory or 
            task_disk > self.free_disk):
            return False

        target_block_idx = -1
        min_valid_block_size = float('inf')
        for idx, block_size in enumerate(self.memory_free_blocks):
            if block_size >= task_memory and block_size < min_valid_block_size:
                min_valid_block_size = block_size
                target_block_idx = idx
        if target_block_idx == -1:
            return False

        target_block_size = self.memory_free_blocks[target_block_idx]
        remaining = target_block_size - task_memory
        del self.memory_free_blocks[target_block_idx]
        if remaining > 0:
            self.memory_free_blocks.append(remaining)

        self.used_cpu += task_cpu
        self.free_cpu -= task_cpu
        self.used_memory += task_memory
        self.free_memory -= task_memory
        self.used_disk += task_disk
        self.free_disk -= task_disk

        self.tasks[task_id] = {
            'cpu': task_cpu,
            'memory': task_memory,
            'disk': task_disk,
            'user_id': user_id,
            'submit_time': submit_time
        }

        self.user_task_count[user_id] = self.user_task_count.get(user_id, 0) + 1
        self.recent_tasks.append((task_id, submit_time))
        return True

    def release_task(self, task_id: str):
        if task_id not in self.tasks:
            log_content = f"任务{task_id}不存在，释放失败"
            print(log_content)
            return False, log_content

        task_info = self.tasks.pop(task_id)
        task_cpu = task_info['cpu']
        task_memory = task_info['memory']
        task_disk = task_info['disk']
        user_id = task_info['user_id']
        submit_time = task_info['submit_time']

        self.used_cpu -= task_cpu
        self.free_cpu += task_cpu
        self.used_memory -= task_memory
        self.free_memory += task_memory
        self.used_disk -= task_disk
        self.free_disk += task_disk

        self.memory_free_blocks.append(task_memory)

        self.user_task_count[user_id] -= 1
        if self.user_task_count[user_id] == 0:
            del self.user_task_count[user_id]

        self.recent_tasks = [(tid, st) for tid, st in self.recent_tasks if tid != task_id]

        log_content = f"任务{task_id}（用户{user_id}）释放成功！释放内存：{task_memory/1024:.1f}GB"
        # print(log_content)
        return True, log_content

    def get_cpu_usage_rate(self) -> float:
        if self.total_cpu == 0:
            return 0.0
        return round((self.used_cpu / self.total_cpu) * 100, 2)

    def calculate_memory_fragmentation_rate(self, target_memory_mb: int) -> float:
        if self.total_memory == 0:
            return 0.0
        if target_memory_mb <= 0:
            unusable = self.free_memory
        else:
            unusable = self.free_memory % target_memory_mb
        return round((unusable / self.total_memory) * 100, 2)
    
    def calculate_cpu_fragmentation_rate(self, target_cpu_cores: int) -> float:
        if target_cpu_cores <= 0 or self.total_cpu <= 0:
            return 0.0
        remainder = self.free_cpu % target_cpu_cores
        frag_rate = (remainder / self.total_cpu) * 100
        return round(frag_rate, 2)

    def get_user_task_count(self, user_id: str) -> int:
        return self.user_task_count.get(user_id, 0)

    def get_recent_task_count(self, current_time, time_window_seconds: int) -> int:
        threshold = current_time - timedelta(seconds=time_window_seconds)
        return sum(1 for _, st in self.recent_tasks if st >= threshold)

    def __str__(self) -> str:
        status = "启用" if self.is_enabled else "禁用"
        cpu_rate = self.calculate_cpu_fragmentation_rate(64)
        frag_rate = self.calculate_memory_fragmentation_rate(8192)  # 默认值，实际由调度器传入
        user_count = len(self.user_task_count)
        recent_count = len(self.recent_tasks)
        return (
            f"状态：{status} | CPU(总{self.total_cpu}/剩{self.free_cpu}/碎片率{cpu_rate}%) | "
            f"内存(总{self.total_memory/1024:.1f}GB/剩{self.free_memory/1024:.1f}GB/碎片率{frag_rate}%) | "
            f"外存(总{self.total_disk}/剩{self.free_disk}) | 任务数{len(self.tasks)} | "
            f"承载用户数{user_count} | 近期任务数{recent_count}"
        )