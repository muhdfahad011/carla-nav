"""Core data types and geometry helpers (no simulator dependency)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class State:
    x: float
    y: float
    yaw: float  # degrees
    speed: float  # m/s
    t: float = 0.0
    collided: bool = False
    lane_invasions: int = 0


@dataclass(frozen=True)
class Action:
    throttle: float = 0.0
    steer: float = 0.0  # +1 turns toward increasing yaw (right in CARLA)
    brake: float = 0.0

    def clamped(self) -> "Action":
        return Action(clip(self.throttle, 0.0, 1.0), clip(self.steer, -1.0, 1.0), clip(self.brake, 0.0, 1.0))


@dataclass(frozen=True)
class Task:
    start: Tuple[float, float, float]  # x, y, yaw(deg)
    goal: Tuple[float, float]
    goal_radius: float = 3.0
    max_steps: int = 400
    speed_limit: float = 8.0  # m/s
    dt: float = 0.1
    max_lane_invasions: int = 3
    obstacles: Tuple[Tuple[float, float, float], ...] = ()  # (x, y, radius); fake backend only


def dist_to_goal(s: State, task: Task) -> float:
    return math.hypot(task.goal[0] - s.x, task.goal[1] - s.y)


def heading_error_deg(s: State, task: Task) -> float:
    bearing = math.degrees(math.atan2(task.goal[1] - s.y, task.goal[0] - s.x))
    return wrap_deg(bearing - s.yaw)
