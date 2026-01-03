import yaml
import os
import random
from datetime import datetime, timedelta

class Machine:
    """机器类（新增任务释放逻辑+用户任务统计+近期任务统计）"""
    def __init__(self, total_cpu, total_memory, total_disk, is_enabled):
        self.total_cpu = total_cpu          # 总CPU核心数
        self.used_cpu = 0                   # 已使用CPU
        self.free_cpu = total_cpu           # 剩余CPU
        self.total_memory = total_memory    # 总内存（MB）
        self.used_memory = 0                # 已使用内存
        self.free_memory = total_memory     # 剩余内存（总空闲）
        self.memory_free_blocks = [total_memory]  # 内存空闲块列表
        self.total_disk = total_disk        # 总外存（GB）
        self.used_disk = 0                  # 已使用外存
        self.free_disk = total_disk         # 剩余外存
        self.tasks = {}                     # 任务字典 {任务ID: 任务详情}
        self.is_enabled = is_enabled        # 节点启用状态

        # 新增：用户任务统计（用于同一用户分散）
        self.user_task_count = {}           # {用户ID: 该用户在本机的任务数}
        # 新增：近期任务统计（用于机器近期限流）
        self.recent_tasks = []              # 近期任务列表，存储(task_id, submit_time)

    def allocate_task(self, task_id, task_cpu, task_memory, task_disk, user_id, submit_time):
        """分配任务（修正为真正的最佳适配算法，易产生碎片）"""
        # 基础资源校验
        if (task_cpu > self.free_cpu or 
            task_memory > self.free_memory or 
            task_disk > self.free_disk):
            return False

        # 内存空闲块分配：真正的最佳适配（找到最小的可用空闲块，易产生碎片）
        target_block_idx = -1
        min_valid_block_size = float('inf')  # 初始化最小有效块为无穷大
        for idx, block_size in enumerate(self.memory_free_blocks):
            if block_size >= task_memory and block_size < min_valid_block_size:
                min_valid_block_size = block_size
                target_block_idx = idx  # 更新为更小的有效块索引
        if target_block_idx == -1:
            return False

        # 分割空闲块
        target_block_size = self.memory_free_blocks[target_block_idx]
        remaining_block_size = target_block_size - task_memory
        del self.memory_free_blocks[target_block_idx]
        if remaining_block_size > 0:
            self.memory_free_blocks.append(remaining_block_size)

        # 更新资源状态
        self.used_cpu += task_cpu
        self.free_cpu -= task_cpu
        self.used_memory += task_memory
        self.free_memory -= task_memory
        self.used_disk += task_disk
        self.free_disk -= task_disk
        # 记录任务
        self.tasks[task_id] = {
            'cpu': task_cpu,
            'memory': task_memory,
            'disk': task_disk,
            'user_id': user_id,
            'submit_time': submit_time
        }

        # 新增：更新用户任务统计
        if user_id in self.user_task_count:
            self.user_task_count[user_id] += 1
        else:
            self.user_task_count[user_id] = 1

        # 新增：更新近期任务列表
        self.recent_tasks.append((task_id, submit_time))

        return True

    def release_task(self, task_id):
        """【核心新增】释放任务资源（产生零散内存空闲块，触发碎片）"""
        # 校验任务是否存在
        if task_id not in self.tasks:
            log_content = f"任务{task_id}不存在，释放失败"
            print(log_content)
            return False, log_content

        # 获取任务信息
        task_info = self.tasks.pop(task_id)
        task_cpu = task_info['cpu']
        task_memory = task_info['memory']
        task_disk = task_info['disk']
        user_id = task_info['user_id']
        submit_time = task_info['submit_time']

        # 1. 恢复机器核心资源
        self.used_cpu -= task_cpu
        self.free_cpu += task_cpu
        self.used_memory -= task_memory
        self.free_memory += task_memory
        self.used_disk -= task_disk
        self.free_disk += task_disk

        # 2. 关键：释放的内存作为独立空闲块添加（不合并相邻块，强制产生碎片）
        self.memory_free_blocks.append(task_memory)

        # 3. 更新用户任务统计（若用户任务数为0则删除）
        if user_id in self.user_task_count:
            self.user_task_count[user_id] -= 1
            if self.user_task_count[user_id] == 0:
                del self.user_task_count[user_id]

        # 4. 移除近期任务列表中的该任务
        self.recent_tasks = [(tid, st) for tid, st in self.recent_tasks if tid != task_id]

        log_content = f"任务{task_id}（用户{user_id}）释放成功！释放内存：{task_memory/1024:.1f}GB"
        print(log_content)
        return True, log_content

    def _merge_adjacent_free_blocks(self, new_block_size):
        """【可选】合并相邻空闲块（真实场景使用，测试碎片时建议关闭）"""
        self.memory_free_blocks.append(new_block_size)
        merged_blocks = {}
        for block in self.memory_free_blocks:
            if block in merged_blocks:
                merged_blocks[block] += 1
            else:
                merged_blocks[block] = 1
        # 重新构建空闲块列表
        self.memory_free_blocks = []
        for block_size, count in merged_blocks.items():
            self.memory_free_blocks.extend([block_size] * count)

    def get_cpu_usage_rate(self):
        """计算CPU使用率（百分比）"""
        if self.total_cpu == 0:
            return 0.0
        return round((self.used_cpu / self.total_cpu) * 100, 2)

    def calculate_memory_fragmentation_rate(self, target_memory_mb):
        """计算单个节点内存碎片率（新算法）"""
        total_memory = self.total_memory
        free_memory = self.free_memory
        
        if total_memory == 0:
            return 0.0
        
        # 计算可用于目标规格的资源量
        usable_memory = 0
        for block in self.memory_free_blocks:
            if block >= target_memory_mb:
                usable_memory += block
        
        # 计算碎片率
        fragmentation_rate = (free_memory - usable_memory) / total_memory * 100
        return round(fragmentation_rate, 2)

    def get_user_task_count(self, user_id):
        """获取指定用户在本机的任务数"""
        return self.user_task_count.get(user_id, 0)

    def get_recent_task_count(self, current_time, time_window_seconds):
        """获取本机在指定时间窗口内的近期任务数"""
        time_threshold = current_time - timedelta(seconds=time_window_seconds)
        recent_count = 0
        for (task_id, submit_time) in self.recent_tasks:
            if submit_time >= time_threshold:
                recent_count += 1
        return recent_count

    def __str__(self):
        """机器状态输出（含碎片率+用户任务+近期任务概览）"""
        status = "启用" if self.is_enabled else "禁用"
        cpu_usage = self.get_cpu_usage_rate()
        # 临时使用8192作为默认目标规格，实际使用时会从策略获取
        frag_rate = self.calculate_memory_fragmentation_rate(8192)
        user_count = len(self.user_task_count)
        recent_task_count = len(self.recent_tasks)
        return (f"状态：{status} | CPU(总{self.total_cpu}/剩{self.free_cpu}/使用率{cpu_usage}%) | "
                f"内存(总{self.total_memory/1024:.1f}GB/剩{self.free_memory/1024:.1f}GB/碎片率{frag_rate}%) | "
                f"外存(总{self.total_disk}/剩{self.free_disk}) | 任务数{len(self.tasks)} | "
                f"承载用户数{user_count} | 近期任务数{recent_task_count}")

class TaskScheduler:
    """任务调度器（新增任务批量释放+策略配置+用户分散+机器限流）"""
    def __init__(self, 
                 machine_config_name="machine_config.yaml", 
                 task_config_name="task_config.yaml",
                 strategy_config_name="strategy_config.yaml",
                 log_filename=None):
        # 配置文件名（同目录）
        self.machine_config_path = machine_config_name
        self.task_config_path = task_config_name
        self.strategy_config_path = strategy_config_name

        # 初始化日志文件
        if log_filename is None:
            # 自动生成带时间戳的日志文件名，避免覆盖
            self.log_filename = f"scheduler_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        else:
            self.log_filename = log_filename
        # 确保日志文件目录存在
        self._ensure_log_dir_exists()

        # 读取三大配置
        self.machine_config = self.load_yaml(self.machine_config_path)
        self.task_config = self.load_yaml(self.task_config_path)
        self.strategy_config = self.load_yaml(self.strategy_config_path)

        # 初始化策略参数
        self._init_strategy_params()

        # 初始化机器
        self.machines = self.init_machines()
        # 任务缓存
        self.task_cache = {}

    def _ensure_log_dir_exists(self):
        """确保日志文件所在目录存在"""
        log_dir = os.path.dirname(self.log_filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def write_log_to_file(self, content, add_timestamp=True):
        """
        将日志内容写入文件
        :param content: 要写入的日志内容
        :param add_timestamp: 是否添加时间戳前缀
        """
        try:
            # 添加时间戳
            if add_timestamp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"[{timestamp}] {content}\n"
            else:
                log_line = f"{content}\n"
            
            # 以追加模式写入文件（不会覆盖原有内容）
            with open(self.log_filename, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception as e:
            print(f"写入日志文件失败：{e}")

    def load_yaml(self, yaml_filename):
        """读取同目录下的YAML文件"""
        current_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
        yaml_full_path = os.path.join(current_dir, yaml_filename)
        if not os.path.exists(yaml_full_path):
            error_msg = f"同目录下未找到YAML文件：{yaml_filename}（路径：{yaml_full_path}）"
            print(error_msg)
            self.write_log_to_file(error_msg)
            raise FileNotFoundError(error_msg)
        try:
            with open(yaml_full_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            error_msg = f"读取YAML文件{yaml_filename}失败：{e}"
            print(error_msg)
            self.write_log_to_file(error_msg)
            raise Exception(error_msg)

    def _init_strategy_params(self):
        """初始化策略权重与配置参数"""
        # 核心目标权重
        target_weights = self.strategy_config.get('target_weights', {})
        self.memory_frag_weight = target_weights.get('memory_fragment_min', 0.5)
        self.user_distribute_weight = target_weights.get('user_task_distribute', 0.3)
        self.machine_recent_weight = target_weights.get('machine_recent_limit', 0.2)

        # 用户分散配置
        user_dist_config = self.strategy_config.get('user_distribute_config', {})
        self.max_user_task_per_machine = user_dist_config.get('max_task_per_user_per_machine', 2)

        # 机器近期限流配置
        machine_recent_config = self.strategy_config.get('machine_recent_config', {})
        self.recent_time_window = machine_recent_config.get('recent_time_window_seconds', 3600)
        self.max_recent_tasks_per_machine = machine_recent_config.get('max_recent_tasks_per_machine', 5)

        # 新增：碎片率计算用的目标资源规格
        frag_config = self.strategy_config.get('fragment_calculation', {})
        self.target_memory_mb = frag_config.get('target_memory_mb', 8192)  # 默认8GB

        # 权重归一化（确保总和为1，避免权重失衡）
        total_weight = self.memory_frag_weight + self.user_distribute_weight + self.machine_recent_weight
        if total_weight != 0:
            self.memory_frag_weight /= total_weight
            self.user_distribute_weight /= total_weight
            self.machine_recent_weight /= total_weight

        # 记录权重初始化日志
        weight_log = (
            f"权重初始化完成（归一化后）：\n"
            f"  内存碎片最小化权重：{self.memory_frag_weight:.2f}\n"
            f"  同一用户任务分散权重：{self.user_distribute_weight:.2f}\n"
            f"  机器近期调度限流权重：{self.machine_recent_weight:.2f}\n"
            f"  碎片率计算目标内存规格：{self.target_memory_mb}MB（{self.target_memory_mb/1024:.1f}GB）"
        )
        self.write_log_to_file(weight_log, add_timestamp=False)

    def init_machines(self):
        """初始化机器列表（精简10台）"""
        machines = []
        count = self.machine_config.get('machine_count', 0)
        cpu = self.machine_config.get('cpu_per_machine', 0)
        memory = self.machine_config.get('memory_per_machine', 0)
        disk = self.machine_config.get('disk_per_machine', 0)
        disabled_ratio = self.machine_config.get('disabled_machine_ratio', 0.0)

        disabled_count = int(count * disabled_ratio)
        enabled_count = count - disabled_count

        # 创建启用节点
        for _ in range(enabled_count):
            machines.append(Machine(cpu, memory, disk, is_enabled=True))
        # 创建禁用节点
        for _ in range(disabled_count):
            machines.append(Machine(cpu, memory, disk, is_enabled=False))
        # 打乱顺序
        random.shuffle(machines)

        # 记录机器初始化日志
        machine_log = f"机器初始化完成：总数量{len(machines)}台 | 启用{enabled_count}台 | 禁用{disabled_count}台"
        print(machine_log)
        self.write_log_to_file(machine_log)
        return machines

    def _calculate_machine_score(self, machine, task_cpu, task_memory, task_disk, user_id, submit_time):
        """计算机器适配得分（综合三大目标，得分越高越优先）"""
        # 1. 内存碎片优化得分（使用新碎片率计算方式）
        mem_frag_rate = machine.calculate_memory_fragmentation_rate(self.target_memory_mb)
        mem_frag_score = 1 - (mem_frag_rate / 100)  # 碎片率0%得1分，100%得0分

        # 预计算：分配任务后的内存碎片率
        if machine.free_memory >= task_memory:
            temp_free_mem = machine.free_memory - task_memory
            target_block_idx = -1
            min_valid_block_size = float('inf')
            for idx, b in enumerate(machine.memory_free_blocks):
                if b >= task_memory and b < min_valid_block_size:
                    min_valid_block_size = b
                    target_block_idx = idx
            if target_block_idx != -1:
                temp_block_size = machine.memory_free_blocks[target_block_idx]
                temp_remaining = temp_block_size - task_memory
                temp_free_blocks = machine.memory_free_blocks.copy()
                del temp_free_blocks[target_block_idx]
                if temp_remaining > 0:
                    temp_free_blocks.append(temp_remaining)
                
                # 计算分配后的碎片率
                temp_usable = 0
                for block in temp_free_blocks:
                    if block >= self.target_memory_mb:
                        temp_usable += block
                temp_frag_rate = (temp_free_mem - temp_usable) / machine.total_memory * 100 if temp_free_mem > 0 else 0
                mem_frag_score = 1 - (temp_frag_rate / 100)

        # 2. 用户分散得分
        user_task_count = machine.get_user_task_count(user_id)
        if user_task_count >= self.max_user_task_per_machine:
            user_dist_score = 0.0
        else:
            user_dist_score = (self.max_user_task_per_machine - user_task_count) / self.max_user_task_per_machine

        # 3. 机器近期限流得分
        recent_task_count = machine.get_recent_task_count(submit_time, self.recent_time_window)
        if recent_task_count >= self.max_recent_tasks_per_machine:
            machine_recent_score = 0.0
        else:
            machine_recent_score = (self.max_recent_tasks_per_machine - recent_task_count) / self.max_recent_tasks_per_machine

        # 综合得分
        total_score = (mem_frag_score * self.memory_frag_weight) + \
                      (user_dist_score * self.user_distribute_weight) + \
                      (machine_recent_score * self.machine_recent_weight)

        return round(total_score, 4)

    def strategy_based_allocation(self, task_cpu, task_memory, task_disk, user_id, submit_time):
        """基于自定义策略的分配算法"""
        eligible_machines = []

        # 筛选基础合格的机器
        for machine in self.machines:
            if not machine.is_enabled:
                continue
            if machine.get_cpu_usage_rate() >= self.machine_config.get('cpu_overload_threshold', 90):
                continue
            if not self.strategy_config.get('resource_overcommit_allowed', False):
                if (task_cpu > machine.total_cpu or
                    task_memory > machine.total_memory or
                    task_disk > machine.total_disk):
                    continue
            if (machine.free_cpu < task_cpu or
                machine.free_memory < task_memory or
                machine.free_disk < task_disk):
                continue
            if machine.get_user_task_count(user_id) >= self.max_user_task_per_machine:
                continue
            recent_task_count = machine.get_recent_task_count(submit_time, self.recent_time_window)
            if recent_task_count >= self.max_recent_tasks_per_machine:
                continue

            eligible_machines.append(machine)

        if not eligible_machines:
            return None

        # 计算综合得分，选择最高分机器
        machine_score_dict = {}
        for machine in eligible_machines:
            score = self._calculate_machine_score(machine, task_cpu, task_memory, task_disk, user_id, submit_time)
            machine_score_dict[machine] = score

        sorted_machines = sorted(machine_score_dict.items(), key=lambda x: x[1], reverse=True)
        best_machine = sorted_machines[0][0]

        return best_machine

    def parse_time(self, time_str):
        """解析时间字符串"""
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            error_msg = f"时间解析失败：{e}"
            print(error_msg)
            self.write_log_to_file(error_msg)
            return None

    def submit_task(self, task_item):
        """提交单个任务（基于自定义策略）"""
        # 提取核心字段
        task_id = task_item.get('task_id')
        cpu_demand = task_item.get('cpu_demand', 0)
        memory_demand = task_item.get('memory_demand_mb', 0)
        disk_demand = task_item.get('disk_demand_gb', 0)
        user_id = task_item.get('user_id', "")
        start_time_str = task_item.get('start_time')
        duration = task_item.get('duration_seconds', 0)

        # 基础校验
        if cpu_demand <= 0 or memory_demand <= 0 or disk_demand <= 0:
            error_msg = f"任务{task_id}（用户{user_id}）资源非法，分配失败"
            print(error_msg)
            self.write_log_to_file(error_msg)
            return False

        # 解析提交时间
        submit_time = self.parse_time(start_time_str)
        if not submit_time:
            error_msg = f"任务{task_id}（用户{user_id}）时间格式非法，分配失败"
            print(error_msg)
            self.write_log_to_file(error_msg)
            return False

        # 寻找最优机器
        target_machine = self.strategy_based_allocation(cpu_demand, memory_demand, disk_demand, user_id, submit_time)
        if not target_machine:
            error_msg = f"任务{task_id}（用户{user_id}）无可用机器（策略过滤后），分配失败"
            print(error_msg)
            self.write_log_to_file(error_msg)
            return False

        # 分配任务
        if target_machine.allocate_task(task_id, cpu_demand, memory_demand, disk_demand, user_id, submit_time):
            machine_idx = self.machines.index(target_machine)
            # 缓存任务信息
            self.task_cache[task_id] = {
                'user_id': user_id,
                'submit_time': submit_time,
                'machine_idx': machine_idx,
                'cpu': cpu_demand,
                'memory': memory_demand,
                'disk': disk_demand
            }
            success_msg = (
                f"任务{task_id}（用户{user_id}）分配成功！\n"
                f"  资源：CPU{cpu_demand}核 | 内存{memory_demand/1024:.1f}GB | 外存{disk_demand}GB\n"
                f"  分配机器：索引{machine_idx} | 机器综合得分：{self._calculate_machine_score(target_machine, cpu_demand, memory_demand, disk_demand, user_id, submit_time)}"
            )
            print(success_msg)
            self.write_log_to_file(success_msg, add_timestamp=False)
            return True
        else:
            error_msg = f"任务{task_id}（用户{user_id}）分配异常，失败"
            print(error_msg)
            self.write_log_to_file(error_msg)
            return False

    def submit_all_tasks(self):
        """批量提交所有大量任务（按开始时间排序）"""
        task_list = self.task_config.get('tasks', [])
        if not task_list:
            msg = "无任务可提交"
            print(msg)
            self.write_log_to_file(msg)
            return

        # 按开始时间排序
        task_list.sort(key=lambda x: self.parse_time(x.get('start_time')) or datetime.min)

        title_msg = f"开始批量提交{len(task_list)}个任务（自定义策略：用户分散+机器限流+内存碎片最小化）"
        print("="*60)
        print(title_msg)
        print("="*60)
        self.write_log_to_file("="*60, add_timestamp=False)
        self.write_log_to_file(title_msg, add_timestamp=False)
        self.write_log_to_file("="*60, add_timestamp=False)

        for task in task_list:
            self.submit_task(task)
            print("-"*40)
            self.write_log_to_file("-"*40, add_timestamp=False)

        finish_msg = "所有任务提交完成！"
        print("="*60)
        print(finish_msg)
        print("="*60)
        self.write_log_to_file("="*60, add_timestamp=False)
        self.write_log_to_file(finish_msg, add_timestamp=False)
        self.write_log_to_file("="*60, add_timestamp=False)

    def release_single_task(self, task_id):
        """释放单个任务"""
        if task_id not in self.task_cache:
            error_msg = f"任务{task_id}未分配，无法释放"
            print(error_msg)
            self.write_log_to_file(error_msg)
            return False

        task_info = self.task_cache[task_id]
        machine_idx = task_info['machine_idx']
        machine = self.machines[machine_idx]

        # 释放机器上的任务
        success, log_content = machine.release_task(task_id)
        if success:
            del self.task_cache[task_id]
            self.write_log_to_file(log_content)
            return True
        else:
            self.write_log_to_file(log_content)
            return False

    def release_random_tasks(self, release_ratio=0.3):
        """【核心新增】随机释放指定比例的任务，用于测试碎片率"""
        if not self.task_cache:
            msg = "无已分配任务，无法随机释放"
            print(msg)
            self.write_log_to_file(msg)
            return

        task_ids = list(self.task_cache.keys())
        release_count = max(1, int(len(task_ids) * release_ratio))
        release_task_ids = random.sample(task_ids, release_count)

        title_msg = f"开始随机释放{len(release_task_ids)}个任务（释放比例：{release_ratio*100}%）"
        print("\n" + "="*60)
        print(title_msg)
        print("="*60)
        self.write_log_to_file("")
        self.write_log_to_file("="*60, add_timestamp=False)
        self.write_log_to_file(title_msg, add_timestamp=False)
        self.write_log_to_file("="*60, add_timestamp=False)

        success_count = 0
        for task_id in release_task_ids:
            if self.release_single_task(task_id):
                success_count += 1
            print("-"*40)
            self.write_log_to_file("-"*40, add_timestamp=False)

        finish_msg = f"随机释放完成！成功释放{success_count}个任务，剩余任务{len(self.task_cache)}个"
        print(finish_msg)
        print("="*60 + "\n")
        self.write_log_to_file(finish_msg, add_timestamp=False)
        self.write_log_to_file("="*60, add_timestamp=False)
        self.write_log_to_file("")

    def show_task_status(self):
        """查看所有任务状态"""
        if not self.task_cache:
            msg = "无已分配任务"
            print(msg)
            self.write_log_to_file(msg)
            return

        title_msg = f"当前已分配任务总数：{len(self.task_cache)}"
        print("\n" + "="*80)
        print(title_msg)
        print("="*80)
        self.write_log_to_file("")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file(title_msg, add_timestamp=False)
        self.write_log_to_file("="*80, add_timestamp=False)

        for task_id, info in self.task_cache.items():
            task_msg = (
                f"任务ID：{task_id} | 用户ID：{info['user_id']}\n"
                f"  分配机器：索引{info['machine_idx']} | 资源：CPU{info['cpu']}核 | 内存{info['memory']/1024:.1f}GB | 外存{info['disk']}GB\n"
                f"  提交时间：{info['submit_time'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(task_msg)
            self.write_log_to_file(task_msg, add_timestamp=False)

        print("="*80 + "\n")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file("")

    def show_strategy_params(self):
        """查看当前策略配置参数（便于对比调整）"""
        title_msg = "当前调度策略配置参数"
        print("\n" + "="*80)
        print(title_msg)
        print("="*80)
        self.write_log_to_file("")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file(title_msg, add_timestamp=False)
        self.write_log_to_file("="*80, add_timestamp=False)

        param_msg = (
            f"核心目标权重（归一化后）：\n"
            f"  内存碎片最小化权重：{self.memory_frag_weight:.2f}\n"
            f"  同一用户任务分散权重：{self.user_distribute_weight:.2f}\n"
            f"  机器近期调度限流权重：{self.machine_recent_weight:.2f}\n"
            f"\n用户分散配置：\n"
            f"  单个用户单台机器最大任务数：{self.max_user_task_per_machine}\n"
            f"\n机器近期限流配置：\n"
            f"  近期时间窗口：{self.recent_time_window}秒（{self.recent_time_window/3600:.1f}小时）\n"
            f"  单台机器近期最大任务数：{self.max_recent_tasks_per_machine}\n"
            f"\n碎片率计算配置：\n"
            f"  目标内存规格：{self.target_memory_mb}MB（{self.target_memory_mb/1024:.1f}GB）"
        )
        print(param_msg)
        self.write_log_to_file(param_msg, add_timestamp=False)

        print("="*80 + "\n")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file("")

    def calculate_global_memory_fragmentation_rate(self):
        """计算全局内存碎片率（使用新算法）"""
        enabled_machines = [m for m in self.machines if m.is_enabled]
        if not enabled_machines:
            return 0.0, []

        frag_rates = [m.calculate_memory_fragmentation_rate(self.target_memory_mb) for m in enabled_machines]
        avg_frag_rate = round(sum(frag_rates) / len(frag_rates), 2)
        return avg_frag_rate, frag_rates

    def print_fragmentation_stats(self, percentiles=[50, 90]):
        """输出指定百分位碎片率（用于策略对比）"""
        avg_frag_rate, frag_rates = self.calculate_global_memory_fragmentation_rate()
        enabled_machine_count = len([m for m in self.machines if m.is_enabled])

        title_msg = "内存碎片率统计（用于策略对比）"
        print("\n" + "="*80)
        print(title_msg)
        print("="*80)
        self.write_log_to_file("")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file(title_msg, add_timestamp=False)
        self.write_log_to_file("="*80, add_timestamp=False)

        stats_msg = (
            f"统计范围：{enabled_machine_count} 台启用节点\n"
            f"目标内存规格：{self.target_memory_mb}MB（{self.target_memory_mb/1024:.1f}GB）\n"
            f"平均内存碎片率：{avg_frag_rate}%\n"
            f"各节点碎片率列表：{frag_rates}"
        )
        print(stats_msg)
        self.write_log_to_file(stats_msg, add_timestamp=False)

        if frag_rates:
            frag_rates_sorted = sorted(frag_rates)
            for p in percentiles:
                if not (0 <= p <= 100):
                    p_msg = f"  无效百分位：{p}"
                    print(p_msg)
                    self.write_log_to_file(p_msg, add_timestamp=False)
                    continue
                idx = int((p / 100) * len(frag_rates_sorted))
                idx = min(idx, len(frag_rates_sorted) - 1)
                p_frag_rate = frag_rates_sorted[idx]
                p_msg = f"  {p}%分位碎片率：{p_frag_rate}%（{p}%的节点碎片率≤{p_frag_rate}%）"
                print(p_msg)
                self.write_log_to_file(p_msg, add_timestamp=False)

        print("="*80 + "\n")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file("")

    def show_all_machines_status(self):
        """查看所有机器状态（便于策略效果验证）"""
        title_msg = f"所有机器状态（总数：{len(self.machines)}台）"
        print("\n" + "="*80)
        print(title_msg)
        print("="*80)
        self.write_log_to_file("")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file(title_msg, add_timestamp=False)
        self.write_log_to_file("="*80, add_timestamp=False)

        enabled_count = sum(1 for m in self.machines if m.is_enabled)
        disabled_count = len(self.machines) - enabled_count
        node_stats = (
            f"节点统计：启用{enabled_count}台 | 禁用{disabled_count}台\n"
            f"碎片率计算基准：目标内存规格{self.target_memory_mb}MB（{self.target_memory_mb/1024:.1f}GB）"
        )
        print(node_stats)
        self.write_log_to_file(node_stats, add_timestamp=False)

        print("-"*80)
        self.write_log_to_file("-"*80, add_timestamp=False)

        for idx, machine in enumerate(self.machines):
            machine_msg = f"机器{idx}：{machine}"
            print(machine_msg)
            self.write_log_to_file(machine_msg, add_timestamp=False)

        print("="*80 + "\n")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file("")

    def verify_user_distribute_effect(self):
        """验证用户任务分散效果（便于策略对比）"""
        title_msg = "用户任务分散效果验证"
        print("\n" + "="*80)
        print(title_msg)
        print("="*80)
        self.write_log_to_file("")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file(title_msg, add_timestamp=False)
        self.write_log_to_file("="*80, add_timestamp=False)

        # 统计每个用户的任务分布
        user_task_dist = {}
        for task_id, info in self.task_cache.items():
            user_id = info['user_id']
            machine_idx = info['machine_idx']
            if user_id not in user_task_dist:
                user_task_dist[user_id] = {}
            if machine_idx not in user_task_dist[user_id]:
                user_task_dist[user_id][machine_idx] = 0
            user_task_dist[user_id][machine_idx] += 1

        # 输出每个用户的任务分布
        for user_id, machine_task_count in user_task_dist.items():
            total_task = sum(machine_task_count.values())
            user_msg = f"用户{user_id}：总任务数{total_task} | 分布在{len(machine_task_count)}台机器上"
            print(user_msg)
            self.write_log_to_file(user_msg, add_timestamp=False)
            for machine_idx, task_count in machine_task_count.items():
                machine_msg = f"  机器{machine_idx}：{task_count}个任务（是否超限：{task_count > self.max_user_task_per_machine}）"
                print(machine_msg)
                self.write_log_to_file(machine_msg, add_timestamp=False)

        # 统计超限数量
        over_limit_count = 0
        for user_id, machine_task_count in user_task_dist.items():
            for machine_idx, task_count in machine_task_count.items():
                if task_count > self.max_user_task_per_machine:
                    over_limit_count += 1
        over_limit_msg = f"\n用户任务超限总数：{over_limit_count}（理想值：0）"
        print(over_limit_msg)
        self.write_log_to_file(over_limit_msg, add_timestamp=False)

        print("="*80 + "\n")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file("")

    def verify_machine_recent_limit_effect(self):
        """验证机器近期限流效果（便于策略对比）"""
        title_msg = "机器近期调度限流效果验证"
        print("\n" + "="*80)
        print(title_msg)
        print("="*80)
        self.write_log_to_file("")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file(title_msg, add_timestamp=False)
        self.write_log_to_file("="*80, add_timestamp=False)

        enabled_machines = [m for m in self.machines if m.is_enabled]
        current_time = datetime.now()

        over_limit_machine_count = 0
        for idx, machine in enumerate(enabled_machines):
            recent_task_count = machine.get_recent_task_count(current_time, self.recent_time_window)
            is_over_limit = recent_task_count > self.max_recent_tasks_per_machine
            if is_over_limit:
                over_limit_machine_count += 1
            machine_msg = f"机器{idx}：近期任务数{recent_task_count} | 最大限制{self.max_recent_tasks_per_machine} | 是否超限：{is_over_limit}"
            print(machine_msg)
            self.write_log_to_file(machine_msg, add_timestamp=False)

        over_limit_msg = f"\n近期任务超限机器数：{over_limit_machine_count}（理想值：0）"
        print(over_limit_msg)
        self.write_log_to_file(over_limit_msg, add_timestamp=False)

        print("="*80 + "\n")
        self.write_log_to_file("="*80, add_timestamp=False)
        self.write_log_to_file("")

# 测试代码（含文件输出功能，可直接运行）
if __name__ == "__main__":
    try:
        # 初始化调度器，可自定义日志文件名，默认自动生成带时间戳的日志
        scheduler = TaskScheduler(
            machine_config_name="machine_config.yaml",
            task_config_name="task_config.yaml",
            strategy_config_name="strategy_config.yaml"
            # 自定义日志文件名示例：log_filename="my_scheduler_log.txt"
        )

        # 打印并记录日志文件名
        log_file_msg = f"本次日志将写入文件：{scheduler.log_filename}"
        print(log_file_msg)
        scheduler.write_log_to_file(log_file_msg)

        # 1. 查看当前策略参数
        scheduler.show_strategy_params()

        # 2. 批量提交所有任务
        scheduler.submit_all_tasks()

        # 3. 提交后查看机器状态
        print("\n【提交任务后，未释放任务的机器状态】")
        scheduler.write_log_to_file("\n【提交任务后，未释放任务的机器状态】")
        scheduler.show_all_machines_status()
        scheduler.print_fragmentation_stats(percentiles=[50, 90])

        # 4. 随机释放30%的任务
        scheduler.release_random_tasks(release_ratio=0.3)

        # 5. 释放后查看机器状态
        print("\n【释放任务后，产生碎片的机器状态】")
        scheduler.write_log_to_file("\n【释放任务后，产生碎片的机器状态】")
        scheduler.show_all_machines_status()
        scheduler.print_fragmentation_stats(percentiles=[50, 90])

        # 6. 后续验证逻辑
        scheduler.show_task_status()
        scheduler.verify_user_distribute_effect()
        scheduler.verify_machine_recent_limit_effect()

        # 最终日志提示
        final_msg = f"程序执行完成，所有日志已保存至：{scheduler.log_filename}"
        print(final_msg)
        scheduler.write_log_to_file(final_msg)

    except Exception as e:
        error_msg = f"程序异常：{e}"
        print(error_msg)
        # 若调度器已初始化，写入异常日志
        try:
            scheduler.write_log_to_file(error_msg)
        except:
            pass