from __future__ import annotations

from typing import Any

from hunyuan_video.actions import MethodDecision


class ReuseMethod:
    name = "reuse"

    def __init__(self, action: Any):
        self.action = action

    def reset(self) -> None:
        self.action.reset()

    def decide(self, **_kwargs: Any) -> MethodDecision:
        full, reason = self.action.decide_full()
        return MethodDecision(full=full, reason=reason)
