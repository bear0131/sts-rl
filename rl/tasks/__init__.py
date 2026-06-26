"""Task registry for the unified trainer.

Add a task: implement a Task subclass in this package and register it here.
"""

from __future__ import annotations

from rl.tasks.base import Task
from rl.tasks.combat import CombatTask
from rl.tasks.wholerun import WholeRunTask

TASKS: dict[str, Task] = {
    t.name: t for t in (CombatTask(), WholeRunTask())
}

__all__ = ["Task", "TASKS"]
