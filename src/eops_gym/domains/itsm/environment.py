"""ITSM environment + task loader factory. Mirrors tau2 domain ``environment.py``."""

from pathlib import Path
from typing import Optional

from eops_gym.data_model.tasks import Task
from eops_gym.domains.itsm.data_model import ItsmDB
from eops_gym.domains.itsm.tools import ItsmTools
from eops_gym.environment.delta import Delta, apply_delta
from eops_gym.environment.environment import Environment
from eops_gym.utils.io_utils import load_file

_DATA_DIR = Path(__file__).resolve().parents[3].parent / "data" / "itsm"
ITSM_DB_PATH = _DATA_DIR / "db.json"
ITSM_POLICY_PATH = _DATA_DIR / "policy.md"
ITSM_TASKS_PATH = _DATA_DIR / "tasks.json"

DOMAIN_NAME = "itsm"


def get_environment(db_delta: Optional[Delta | dict] = None) -> Environment:
    """Build a fresh ITSM environment: load seed DB, apply the task delta (item 7)."""
    db = ItsmDB.load(ITSM_DB_PATH)
    db = apply_delta(db, db_delta)
    policy = ITSM_POLICY_PATH.read_text(encoding="utf-8") if ITSM_POLICY_PATH.exists() else ""
    return Environment(domain_name=DOMAIN_NAME, policy=policy, tools=ItsmTools(db))


def get_tasks() -> list[Task]:
    """Load and validate the ITSM tasks from tasks.json (item 6)."""
    raw = load_file(ITSM_TASKS_PATH)
    return [Task.model_validate(t) for t in raw]
