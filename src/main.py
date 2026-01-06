# src/main.py
from .scheduler.task_scheduler import TaskScheduler
from datetime import datetime
from pathlib import Path

# === 新增：确定项目根目录（src 的父目录）===
PROJECT_ROOT = Path(__file__).parent.parent

def main():
    try:
        # === 修改：构建完整的配置文件路径 ===
        scheduler = TaskScheduler(
            machine_config_name=PROJECT_ROOT / "config" / "machine_config.yaml",
            task_config_name=PROJECT_ROOT / "config" / "task_config.yaml",
            strategy_config_name=PROJECT_ROOT / "config" / "strategy_config.yaml"
        )

        log_file_msg = f"本次日志将写入文件：{scheduler.log_filename}"
        print(log_file_msg)
        scheduler.write_log_to_file(log_file_msg)

        scheduler.show_strategy_params()
        scheduler.submit_all_tasks()

        print("\n【提交任务后，未释放任务的机器状态】")
        scheduler.write_log_to_file("\n【提交任务后，未释放任务的机器状态】")
        scheduler.show_all_machines_status()
        scheduler.print_fragmentation_stats(percentiles=[50, 90])

        scheduler.release_random_tasks(release_ratio=0.3)

        print("\n【释放任务后，产生碎片的机器状态】")
        scheduler.write_log_to_file("\n【释放任务后，产生碎片的机器状态】")
        scheduler.show_all_machines_status()
        scheduler.print_fragmentation_stats(percentiles=[50, 90])

        scheduler.show_task_status()
        scheduler.verify_user_distribute_effect()
        scheduler.verify_machine_recent_limit_effect()

        final_msg = f"程序执行完成，所有日志已保存至：{scheduler.log_filename}"
        print(final_msg)
        scheduler.write_log_to_file(final_msg)

    except Exception as e:
        error_msg = f"程序异常：{e}"
        print(error_msg)
        try:
            scheduler.write_log_to_file(error_msg)
        except:
            pass

# 注意：由于使用了相对导入 from .scheduler...，
# 必须通过 python -m src.main 运行，此时 __name__ != "__main__"
# 所以下面这个 if 实际不会触发，但保留也无妨
if __name__ == "__main__":
    main()