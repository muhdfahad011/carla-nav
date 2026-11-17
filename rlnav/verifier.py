"""Independent grader. Judges a trajectory with hard, binary checks.

Kept separate from the training reward on purpose: reward = dense signal for learning,
verifier = strict pass/fail that must not be gameable. Every check maps to a known hack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .types import State, Task, dist_to_goal, wrap_deg

MAX_PHYSICAL_SPEED = 30.0  # m/s; faster per-step motion means teleporting
MAX_TOTAL_TURN_DEG = 540.0  # cumulative |yaw change|; blocks spinning/looping
MAX_PATH_RATIO = 3.0  # path length vs straight line; blocks loop-farming
SPEED_TOLERANCE = 1.1


@dataclass
class VerifierResult:
    passed: bool
    score: float  # binary 0/1 (use metrics["progress"] for diagnostics only)
    failures: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def verify(traj: Sequence[State], task: Task) -> VerifierResult:
    fails: List[str] = []
    if len(traj) < 2:
        return VerifierResult(False, 0.0, ["empty_trajectory"])

    sx, sy, _ = task.start
    if math.hypot(traj[0].x - sx, traj[0].y - sy) > 1.0:
        fails.append("bad_start")  # fabricated/teleported start

    steps = len(traj) - 1
    if steps > task.max_steps:
        fails.append("too_many_steps")

    path = turn = 0.0
    max_step = 0.0
    for a, b in zip(traj, traj[1:]):
        d = math.hypot(b.x - a.x, b.y - a.y)
        path += d
        max_step = max(max_step, d)
        turn += abs(wrap_deg(b.yaw - a.yaw))
        if b.t < a.t:
            fails.append("time_went_backwards")
            break

    if max_step > MAX_PHYSICAL_SPEED * task.dt:
        fails.append("teleport")
    if abs((traj[-1].t - traj[0].t) - steps * task.dt) > 0.05 * steps * task.dt + 1e-6:
        fails.append("bad_timestamps")

    max_speed = max(s.speed for s in traj)
    if max_speed > task.speed_limit * SPEED_TOLERANCE:
        fails.append("speed_limit")
    if any(s.collided for s in traj):
        fails.append("collision")
    if traj[-1].lane_invasions > task.max_lane_invasions:
        fails.append("lane_invasions")
    if turn > MAX_TOTAL_TURN_DEG:
        fails.append("excessive_turning")

    straight = math.hypot(task.goal[0] - sx, task.goal[1] - sy)
    if path > MAX_PATH_RATIO * straight + 10.0:
        fails.append("inefficient_path")

    final_d = dist_to_goal(traj[-1], task)
    if final_d > task.goal_radius:
        fails.append("goal_not_reached")

    progress = max(0.0, min(1.0, 1.0 - final_d / straight)) if straight > 0 else 1.0
    metrics = {"steps": float(steps), "path": path, "turn_deg": turn, "max_speed": max_speed,
               "final_dist": final_d, "progress": progress}
    return VerifierResult(not fails, 0.0 if fails else 1.0, fails, metrics)
