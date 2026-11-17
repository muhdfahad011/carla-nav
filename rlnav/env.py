"""Gym-like environment over any backend (FakeBackend or CarlaBackend)."""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

from .reward import RewardConfig, compute_reward
from .types import Action, State, Task, dist_to_goal, heading_error_deg


class Backend(Protocol):
    def reset(self, task: Task) -> State: ...
    def step(self, action: Action) -> State: ...
    def close(self) -> None: ...


def observe(s: State, task: Task, step: int) -> Dict[str, Any]:
    return {
        "dist_to_goal": round(dist_to_goal(s, task), 2),
        "heading_error_deg": round(heading_error_deg(s, task), 1),  # +ve => steer positive
        "speed": round(s.speed, 2),
        "speed_limit": task.speed_limit,
        "collided": s.collided,
        "lane_invasions": s.lane_invasions,
        "step": step,
    }


class NavEnv:
    def __init__(self, backend: Backend, task: Task, reward_cfg: RewardConfig = RewardConfig()) -> None:
        self.backend, self.task, self.cfg = backend, task, reward_cfg
        self.trajectory: List[State] = []

    def reset(self) -> Dict[str, Any]:
        s = self.backend.reset(self.task)
        self.trajectory = [s]
        return observe(s, self.task, 0)

    def step(self, action: Action) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        prev = self.trajectory[-1]
        cur = self.backend.step(action)
        self.trajectory.append(cur)
        n = len(self.trajectory) - 1
        reached = dist_to_goal(cur, self.task) <= self.task.goal_radius
        done = reached or cur.collided or n >= self.task.max_steps
        return observe(cur, self.task, n), compute_reward(prev, cur, self.task, self.cfg), done, {"reached": reached}
