"""Training reward. Potential-based on distance-to-goal, so loops net to zero.

Deliberately NOT rewarded: speed, distance travelled, time alive.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import State, Task, dist_to_goal


@dataclass(frozen=True)
class RewardConfig:
    progress_k: float = 1.0
    step_penalty: float = 0.01
    collision: float = -10.0
    lane_invasion: float = -0.2
    goal_bonus: float = 20.0


def compute_reward(prev: State, cur: State, task: Task, cfg: RewardConfig = RewardConfig()) -> float:
    d0, d1 = dist_to_goal(prev, task), dist_to_goal(cur, task)
    r = cfg.progress_k * (d0 - d1) - cfg.step_penalty  # telescopes: circling sums to <= 0
    r += cfg.lane_invasion * (cur.lane_invasions - prev.lane_invasions)
    if cur.collided and not prev.collided:
        r += cfg.collision
    if d1 <= task.goal_radius < d0:
        r += cfg.goal_bonus
    return r
