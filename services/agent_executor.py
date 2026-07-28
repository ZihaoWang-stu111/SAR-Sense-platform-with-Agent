from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable


class AgentExecutor:
    """用固定大小的线程池承载同步 Agent 任务。"""

    def __init__(self, max_workers: int = 3):
        if max_workers < 1:
            raise ValueError("max_workers 必须大于 0")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="agent-worker",
        )

    def submit(self, task: Callable, *args, **kwargs) -> Future:
        return self._executor.submit(task, *args, **kwargs)

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
