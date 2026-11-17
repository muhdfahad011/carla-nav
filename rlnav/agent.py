"""Policies: LLM agent (Anthropic API), heuristic baseline, random baseline."""
from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, Optional

from .types import Action, clip

SYSTEM_PROMPT = (
    "You drive a car toward a goal. Each turn you get a JSON observation. Reply with ONLY a JSON object "
    '{"throttle": 0..1, "steer": -1..1, "brake": 0..1}. Steer in the direction of heading_error_deg '
    "(positive => positive steer). Keep speed below speed_limit, slow down near the goal, never collide."
)
SAFE_ACTION = Action(throttle=0.0, steer=0.0, brake=1.0)


def parse_action(text: str) -> Action:
    """Robustly parse model output. Anything malformed -> safe braking action; values clamped."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return SAFE_ACTION
    try:
        d = json.loads(m.group(0))
        return Action(float(d.get("throttle", 0.0)), float(d.get("steer", 0.0)), float(d.get("brake", 0.0))).clamped()
    except (ValueError, TypeError, AttributeError):
        return SAFE_ACTION


class LLMPolicy:
    """Calls the LLM every `decision_interval` steps and repeats the action in between (cost control)."""

    def __init__(self, client: Any = None, model: Optional[str] = None, decision_interval: int = 5) -> None:
        if client is None:
            import anthropic  # lazy import

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("RLNAV_MODEL", "claude-sonnet-5-5")
        self.interval = max(1, decision_interval)
        self._n = 0
        self._last = SAFE_ACTION

    def act(self, obs: Dict[str, Any]) -> Action:
        if self._n % self.interval == 0:
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=100, system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": json.dumps(obs)}],
                )
                self._last = parse_action(resp.content[0].text)
            except Exception:  # API failure must not crash a rollout
                self._last = SAFE_ACTION
        self._n += 1
        return self._last


class HeuristicPolicy:
    """Proportional steering + speed governor. Reference solution / sanity baseline."""

    def act(self, obs: Dict[str, Any]) -> Action:
        steer = clip(obs["heading_error_deg"] / 30.0, -1.0, 1.0)
        target = min(obs["speed_limit"] * 0.6, 5.0) if obs["dist_to_goal"] > 10 else 2.0
        if obs["speed"] < target:
            return Action(throttle=0.5, steer=steer)
        return Action(throttle=0.0, steer=steer, brake=0.3 if obs["speed"] > target + 1 else 0.0)


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, obs: Dict[str, Any]) -> Action:
        return Action(self.rng.random(), self.rng.uniform(-1, 1), 0.0)
