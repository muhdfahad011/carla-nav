"""Kinematic-bicycle stand-in for CARLA. Lets tests, CI and K8s jobs run with no simulator."""
from __future__ import annotations

import math

from .types import Action, State, Task, clip

WHEELBASE, MAX_STEER_DEG, ACCEL, BRAKE, DRAG = 2.5, 35.0, 3.0, 8.0, 0.05


class FakeBackend:
    def __init__(self) -> None:
        self.task: Task
        self.s: State

    def reset(self, task: Task) -> State:
        self.task = task
        x, y, yaw = task.start
        self.s = State(x, y, yaw, 0.0, 0.0)
        return self.s

    def step(self, action: Action) -> State:
        a, s, dt = action.clamped(), self.s, self.task.dt
        v = max(0.0, s.speed + (a.throttle * ACCEL - a.brake * BRAKE - DRAG * s.speed) * dt)
        yaw = math.radians(s.yaw) + v / WHEELBASE * math.tan(math.radians(a.steer * MAX_STEER_DEG)) * dt
        x, y = s.x + v * math.cos(yaw) * dt, s.y + v * math.sin(yaw) * dt
        hit = any(math.hypot(x - ox, y - oy) <= r for ox, oy, r in self.task.obstacles)
        self.s = State(x, y, math.degrees(yaw), v, s.t + dt, s.collided or hit, s.lane_invasions)
        return self.s

    def close(self) -> None:
        pass
