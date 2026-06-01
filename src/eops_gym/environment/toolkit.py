"""Toolkit base. Trimmed mirror of tau2's ``environment/toolkit.py``.

A ToolKit owns a domain ``DB`` and exposes methods decorated with ``@is_tool``.
Read tools query the DB; write tools mutate ``self.db`` in place. The evaluator
replays gold actions through these same tools, so behaviour is identical between
a live run and gold-state computation.
"""

import inspect
from enum import Enum
from typing import Any, Callable, Dict, Optional

from eops_gym.environment.db import DB
from eops_gym.utils.hash_utils import get_dict_hash

TOOL_ATTR = "__tool__"
TOOL_TYPE_ATTR = "__tool_type__"


class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"


def is_tool(tool_type: ToolType = ToolType.READ):
    """Mark a ToolKit method as a callable tool."""

    def decorator(func: Callable) -> Callable:
        setattr(func, TOOL_ATTR, True)
        setattr(func, TOOL_TYPE_ATTR, tool_type)
        return func

    return decorator


class ToolKitType(type):
    """Metaclass that collects ``@is_tool``-decorated methods (incl. inherited)."""

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        tool_names: dict[str, None] = {}
        for base in reversed(cls.__mro__):
            for attr_name, member in vars(base).items():
                if callable(member) and getattr(member, TOOL_ATTR, False):
                    tool_names[attr_name] = None
        cls._tool_names = tuple(tool_names.keys())


class ToolKitBase(metaclass=ToolKitType):
    """Base class for domain toolkits."""

    _tool_names: tuple[str, ...] = ()

    def __init__(self, db: Optional[DB] = None):
        self.db: Optional[DB] = db

    @property
    def tools(self) -> Dict[str, Callable]:
        """Bound tool methods keyed by name."""
        return {name: getattr(self, name) for name in self._tool_names}

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_names

    def tool_type(self, tool_name: str) -> ToolType:
        return getattr(getattr(self, tool_name), TOOL_TYPE_ATTR)

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a tool by name."""
        if not self.has_tool(tool_name):
            raise ValueError(f"Tool {tool_name!r} not found.")
        return self.tools[tool_name](**kwargs)

    def get_tool_schemas(self) -> list[dict]:
        """Minimal OpenAI-style schemas (param names from the signature).

        Enough for wiring an agent later; full typing/docs are item 1's concern.
        """
        schemas = []
        for name in self._tool_names:
            sig = inspect.signature(getattr(self, name))
            props = {p: {"type": "string"} for p in sig.parameters}
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (getattr(self, name).__doc__ or "").strip(),
                        "parameters": {"type": "object", "properties": props},
                    },
                }
            )
        return schemas

    def get_db_hash(self) -> str:
        if self.db is None:
            raise ValueError("Database has not been initialized.")
        return get_dict_hash(self.db.model_dump())
