"""Thin ITSM data model (scaffold for items 1-2, out of scope).

Just enough of the ITSM schema to develop and end-to-end test items 4-7.
The worked example in the spec is an incident workload-rebalance task, so
``Incident`` is modelled with real fields; ``User``/``Group`` are minimal.
Replace with the full SQLite->JSON port (item 2) later.
"""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

from eops_gym.environment.db import DB

IncidentStatus = Literal["new", "in_progress", "on_hold", "resolved", "closed"]
Priority = Literal["low", "moderate", "high", "critical"]
Severity = Literal["low", "medium", "high"]


class Incident(BaseModel):
    incident_id: str
    number: str
    short_description: str
    caller_id: str
    assigned_to: Optional[str] = None
    assignment_group: Optional[str] = None
    status: IncidentStatus = "new"
    category: str = "software"
    impact: Severity = "low"
    urgency: Severity = "low"
    priority: Priority = "low"
    created_at: Optional[str] = None
    org_id: Optional[str] = None


class User(BaseModel):
    user_id: str
    name: str


class Group(BaseModel):
    group_id: str
    name: str


class ItsmDB(DB):
    """In-memory ITSM database. Each collection is ``dict[id, record]``."""

    incidents: Dict[str, Incident] = Field(default_factory=dict)
    users: Dict[str, User] = Field(default_factory=dict)
    groups: Dict[str, Group] = Field(default_factory=dict)
