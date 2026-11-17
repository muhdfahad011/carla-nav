"""Real CARLA backend (0.9.13+ API). Imported lazily so the rest of the package needs no CARLA.

Not exercised by the unit tests (they use FakeBackend); needs a running CARLA server.
"""
from __future__ import annotations

import math
from typing import Any, List

from .types import Action, State, Task


class CarlaBackend:
    def __init__(self, host: str = "localhost", port: int = 2000, town: str = "Town01", dt: float = 0.1) -> None:
        import carla  # lazy import

        self.carla = carla
        self.dt = dt
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        st = self.world.get_settings()
        st.synchronous_mode, st.fixed_delta_seconds = True, dt
        self.world.apply_settings(st)
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle: Any = None
        self.sensors: List[Any] = []
        self._collided, self._lane, self._t = False, 0, 0.0

    def make_task(self, start_idx: int, goal_idx: int, **kw: Any) -> Task:
        s, g = self.spawn_points[start_idx], self.spawn_points[goal_idx]
        return Task(start=(s.location.x, s.location.y, s.rotation.yaw),
                    goal=(g.location.x, g.location.y), dt=self.dt, **kw)

    def reset(self, task: Task) -> State:
        c = self.carla
        self._destroy()
        self._collided, self._lane, self._t = False, 0, 0.0
        lib = self.world.get_blueprint_library()
        x, y, yaw = task.start
        self.vehicle = self.world.spawn_actor(lib.find("vehicle.tesla.model3"),
                                              c.Transform(c.Location(x=x, y=y, z=0.5), c.Rotation(yaw=yaw)))
        col = self.world.spawn_actor(lib.find("sensor.other.collision"), c.Transform(), attach_to=self.vehicle)
        lane = self.world.spawn_actor(lib.find("sensor.other.lane_invasion"), c.Transform(), attach_to=self.vehicle)
        col.listen(lambda _e: setattr(self, "_collided", True))
        lane.listen(lambda _e: setattr(self, "_lane", self._lane + 1))
        self.sensors = [col, lane]
        self.world.tick()
        return self._state()

    def step(self, action: Action) -> State:
        a = action.clamped()
        self.vehicle.apply_control(self.carla.VehicleControl(throttle=a.throttle, steer=a.steer, brake=a.brake))
        self.world.tick()
        self._t += self.dt
        return self._state()

    def _state(self) -> State:
        tf, v = self.vehicle.get_transform(), self.vehicle.get_velocity()
        return State(tf.location.x, tf.location.y, tf.rotation.yaw,
                     math.sqrt(v.x**2 + v.y**2 + v.z**2), self._t, self._collided, self._lane)

    def _destroy(self) -> None:
        for a in [*self.sensors, self.vehicle]:
            if a is not None:
                a.destroy()
        self.sensors, self.vehicle = [], None

    def close(self) -> None:
        self._destroy()
        st = self.world.get_settings()
        st.synchronous_mode, st.fixed_delta_seconds = False, None
        self.world.apply_settings(st)
