"""Task and evaluation-criteria models (items 6 + 7).

Mirrors tau2's ``data_model/tasks.py`` but trimmed to what items 4-7 need:
a user scenario (persona + task description), evaluation criteria (gold actions
+ NL assertions), and a per-task initial-state ``Delta`` (item 7).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from eops_gym.environment.delta import Delta


class UserProfile(BaseModel):
    """The persona the user simulator role-plays (item 4 input)."""

    name: str
    personality: str


class UserScenario(BaseModel):
    """Everything passed to the user simulator."""

    persona: UserProfile
    task_description: str


class Action(BaseModel):
    """A gold tool call replayed to compute the expected DB state (item 6)."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class EvaluationCriteria(BaseModel):
    """How a task is scored (item 6): DB match via actions + NL assertions."""

    actions: List[Action] = Field(default_factory=list)
    nl_assertions: List[str] = Field(default_factory=list)


class Task(BaseModel):
    """A single benchmark task (items 6 + 7)."""

    id: str
    user_scenario: UserScenario
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
    # item 7: collection -> record_id -> {set|create|delete}
    initial_state_delta: Optional[Delta] = None
