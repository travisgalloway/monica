"""Multi-session state map (DEFERRED stub).

Maps session_id -> that session's fixed-size recurrent state. Multi-session serving
is the easy regime for Mamba: constant memory per session, so
max_concurrent = memory_budget / per_session_state. Serialize within a session
(sequential dependency); parallelize across sessions. No batching at POC scale.
"""

from __future__ import annotations

from typing import Dict

from ..model.interface import ModelInterface, State


class SessionStore:  # pragma: no cover - deferred
    def __init__(self, model: ModelInterface):
        self.model = model
        self._states: Dict[str, State] = {}
        raise NotImplementedError("Serving layer is deferred; build after the core loop.")
