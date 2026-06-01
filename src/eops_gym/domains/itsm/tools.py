"""Thin ITSM tool stubs (scaffold for item 1, out of scope).

A handful of CRUD tools over the in-memory ITSM DB — enough to drive and
evaluate incident tasks. All writes mutate ``self.db`` in place.
"""

from typing import Optional

from eops_gym.domains.itsm.data_model import Incident, ItsmDB
from eops_gym.environment.toolkit import ToolKitBase, ToolType, is_tool


class ItsmTools(ToolKitBase):
    """CRUD tools over the ITSM database."""

    db: ItsmDB

    def __init__(self, db: ItsmDB) -> None:
        super().__init__(db)

    def _get_incident(self, incident_id: str) -> Incident:
        if incident_id not in self.db.incidents:
            raise ValueError(f"Incident {incident_id} not found.")
        return self.db.incidents[incident_id]

    @is_tool(ToolType.READ)
    def get_incident(self, incident_id: str) -> Incident:
        """Return the incident with the given id."""
        return self._get_incident(incident_id)

    @is_tool(ToolType.WRITE)
    def update_incident(
        self,
        incident_id: str,
        assigned_to: Optional[str] = None,
        assignment_group: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        impact: Optional[str] = None,
        urgency: Optional[str] = None,
        priority: Optional[str] = None,
        short_description: Optional[str] = None,
    ) -> Incident:
        """Update fields on an existing incident. Only provided fields change."""
        incident = self._get_incident(incident_id)
        updates = {
            "assigned_to": assigned_to,
            "assignment_group": assignment_group,
            "status": status,
            "category": category,
            "impact": impact,
            "urgency": urgency,
            "priority": priority,
            "short_description": short_description,
        }
        for field, value in updates.items():
            if value is not None:
                setattr(incident, field, value)
        return incident

    @is_tool(ToolType.WRITE)
    def create_incident(
        self,
        incident_id: str,
        number: str,
        short_description: str,
        caller_id: str,
        assigned_to: Optional[str] = None,
        assignment_group: Optional[str] = None,
        status: str = "new",
        category: str = "software",
        impact: str = "low",
        urgency: str = "low",
        priority: str = "low",
        created_at: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> Incident:
        """Create a new incident and add it to the database."""
        if incident_id in self.db.incidents:
            raise ValueError(f"Incident {incident_id} already exists.")
        incident = Incident(
            incident_id=incident_id,
            number=number,
            short_description=short_description,
            caller_id=caller_id,
            assigned_to=assigned_to,
            assignment_group=assignment_group,
            status=status,  # type: ignore[arg-type]
            category=category,
            impact=impact,  # type: ignore[arg-type]
            urgency=urgency,  # type: ignore[arg-type]
            priority=priority,  # type: ignore[arg-type]
            created_at=created_at,
            org_id=org_id,
        )
        self.db.incidents[incident_id] = incident
        return incident

    @is_tool(ToolType.READ)
    def list_incidents(self) -> list[Incident]:
        """List all incidents in the database."""
        return list(self.db.incidents.values())
