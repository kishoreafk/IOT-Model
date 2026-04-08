import uuid
import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Background task representation."""
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    _thread: Optional[threading.Thread] = None


class TaskTracker:
    """Background task queue and tracker."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()

    def create(self, task_id: str, task_type: str):
        """Create a new task entry."""
        with self._lock:
            self.tasks[task_id] = Task(task_id=task_id, task_type=task_type)

    def complete(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as completed with result."""
        with self._lock:
            task = self.tasks.get(task_id)
            if task:
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                task.completed_at = time.time()

    def fail(self, task_id: str, error: str):
        """Mark a task as failed with error message."""
        with self._lock:
            task = self.tasks.get(task_id)
            if task:
                task.error = error
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()

    def create_task(self, task_type: str, func: Callable, *args, **kwargs) -> str:
        """Create and start a background task."""
        task_id = str(uuid.uuid4())

        with self._lock:
            task = Task(task_id=task_id, task_type=task_type)
            self.tasks[task_id] = task

        def run_task():
            task.status = TaskStatus.RUNNING
            try:
                result = func(*args, **kwargs)
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
            finally:
                task.completed_at = time.time()

        thread = threading.Thread(target=run_task, daemon=True)
        task._thread = thread
        thread.start()

        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and result."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None

            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "task_type": task.task_type,
                "progress": task.progress,
                "result": task.result,
                "error": task.error,
                "created_at": task.created_at,
                "completed_at": task.completed_at,
            }

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and result (alias for get_task)."""
        return self.get_task(task_id)

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all tasks."""
        with self._lock:
            return {
                task_id: {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "task_type": task.task_type,
                    "progress": task.progress,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                }
                for task_id, task in self.tasks.items()
            }

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for task completion."""
        start_time = time.time()

        while True:
            task_data = self.get_task(task_id)
            if not task_data:
                return None

            if task_data["status"] in ["completed", "failed"]:
                return task_data

            if timeout and (time.time() - start_time) > timeout:
                return task_data

            time.sleep(0.1)

    def clear_completed(self):
        """Clear completed tasks."""
        with self._lock:
            completed_ids = [
                tid for tid, task in self.tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            ]
            for tid in completed_ids:
                del self.tasks[tid]