from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterable
import os

from config import WANDB_ENABLED, WANDB_PROJECT, WANDB_ENTITY, WANDB_TAGS

@dataclass
class _NoOpRun:
    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    _closed: bool = False
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        pass
    @property
    def summary(self) -> Dict[str, Any]:
        return {}
    def finish(self):
        self._closed = True

def init_run(name: str, config: Dict[str, Any] | None = None, group: str | None = None, tags: Iterable[str] | None = None):
    """Initialize a safe wandb run or a no-op stub."""
    enabled_env = os.getenv("WANDB_DISABLED", "").lower() not in {"true", "1", "yes"}
    if not WANDB_ENABLED or not enabled_env:
        run = _NoOpRun(name=name, config=config or {})
        return run

    try:
        import wandb  # type: ignore
    except Exception:
        run = _NoOpRun(name=name, config=config or {})
        return run

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=name,
        config=config or {},
        group=group,
        tags=list(tags or WANDB_TAGS),
        reinit=True,
    )
    return run
