"""
Incremental Planning Module

This module provides the core logic for incremental planning in the FastMCP builder.
It is designed to break down complex planning tasks into smaller, manageable subtasks,
track progress, and adaptively update the plan as new information or requirements emerge.

TODO: 
- Integrate with the main planning engine.
- Add persistence for plan state.
- Add unit tests for all planning functions.
"""

from typing import List, Dict, Optional, Any, Callable
from enum import Enum, auto
import logging

# --- Types and Interfaces ---

class TaskStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    BLOCKED = auto()

class PlanStep:
    """
    Represents a single step in an incremental plan.
    """
    def __init__(
        self, 
        description: str, 
        status: TaskStatus = TaskStatus.PENDING, 
        subtasks: Optional[List['PlanStep']] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.description = description
        self.status = status
        self.subtasks = subtasks if subtasks is not None else []
        self.metadata = metadata if metadata is not None else {}

    def is_completed(self) -> bool:
        if self.status != TaskStatus.COMPLETED:
            return False
        return all(sub.is_completed() for sub in self.subtasks)

    def mark_completed(self):
        self.status = TaskStatus.COMPLETED
        for sub in self.subtasks:
            sub.mark_completed()

    def add_subtask(self, subtask: 'PlanStep'):
        self.subtasks.append(subtask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "status": self.status.name,
            "subtasks": [sub.to_dict() for sub in self.subtasks],
            "metadata": self.metadata
        }

    def __repr__(self):
        return f"PlanStep(description={self.description!r}, status={self.status}, subtasks={len(self.subtasks)})"

class IncrementalPlan:
    """
    Represents an incremental plan consisting of multiple steps.
    """
    def __init__(self, title: str, steps: Optional[List[PlanStep]] = None):
        self.title = title
        self.steps = steps if steps is not None else []
        self.current_index = 0

    def add_step(self, step: PlanStep):
        self.steps.append(step)

    def get_next_step(self) -> Optional[PlanStep]:
        while self.current_index < len(self.steps):
            step = self.steps[self.current_index]
            if step.status != TaskStatus.COMPLETED:
                return step
            self.current_index += 1
        return None

    def mark_step_completed(self, index: int):
        if 0 <= index < len(self.steps):
            self.steps[index].mark_completed()
        else:
            logging.warning(f"Attempted to mark invalid step index {index} as completed.")

    def is_completed(self) -> bool:
        return all(step.is_completed() for step in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "steps": [step.to_dict() for step in self.steps]
        }

    def __repr__(self):
        return f"IncrementalPlan(title={self.title!r}, steps={len(self.steps)})"

# --- Core Planning Functions ---

def break_down_task(
    task_description: str, 
    breakdown_fn: Callable[[str], List[str]]
) -> PlanStep:
    """
    Breaks down a high-level task into subtasks using the provided breakdown function.

    Args:
        task_description: The high-level task to break down.
        breakdown_fn: A function that takes a task description and returns a list of subtask descriptions.

    Returns:
        PlanStep: The root PlanStep with subtasks.
    """
    root = PlanStep(description=task_description)
    subtask_descriptions = breakdown_fn(task_description)
    for desc in subtask_descriptions:
        root.add_subtask(PlanStep(description=desc))
    return root

def update_plan_with_feedback(
    plan: IncrementalPlan, 
    feedback: Dict[str, Any]
) -> None:
    """
    Updates the plan based on feedback (e.g., progress, blockers, new requirements).

    Args:
        plan: The IncrementalPlan to update.
        feedback: A dictionary containing feedback information.
    """
    # TODO: Expand this logic to handle more complex feedback structures.
    for idx, step in enumerate(plan.steps):
        if feedback.get("completed_steps") and idx in feedback["completed_steps"]:
            step.mark_completed()
        if feedback.get("blockers") and idx in feedback["blockers"]:
            step.status = TaskStatus.BLOCKED
            logging.info(f"Step {idx} marked as BLOCKED due to feedback.")

def print_plan(plan: IncrementalPlan) -> None:
    """
    Pretty-prints the plan to the console for review.

    Args:
        plan: The IncrementalPlan to print.
    """
    print(f"Plan: {plan.title}")
    for idx, step in enumerate(plan.steps):
        status = step.status.name
        print(f"  [{idx}] {step.description} - {status}")
        for sub_idx, sub in enumerate(step.subtasks):
            sub_status = sub.status.name
            print(f"    [{idx}.{sub_idx}] {sub.description} - {sub_status}")

# --- Example Usage (for dev/test only, remove in prod) ---

if __name__ == "__main__":
    # Example breakdown function
    def simple_breakdown(task: str) -> List[str]:
        # TODO: Replace with NLP-based or domain-specific breakdown logic
        return [f"{task} - Subtask {i+1}" for i in range(3)]

    # Create a plan
    plan = IncrementalPlan(title="Build Incremental Planning Module")
    root_step = break_down_task("Implement core planning logic", simple_breakdown)
    plan.add_step(root_step)
    plan.add_step(PlanStep(description="Write unit tests"))
    plan.add_step(PlanStep(description="Integrate with FastMCP builder"))

    # Simulate feedback
    feedback = {"completed_steps": [0], "blockers": [2]}
    update_plan_with_feedback(plan, feedback)

    # Print plan
    print_plan(plan)
