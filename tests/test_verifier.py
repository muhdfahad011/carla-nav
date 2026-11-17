"""Each test is a known reward hack. The verifier must reject all of them and accept the honest run."""
import math
from typing import List

from rlnav.agent import HeuristicPolicy
from rlnav.backend_fake import FakeBackend
from rlnav.reward import compute_reward
from rlnav.rollout import run_episode
from rlnav.types import State, Task
from rlnav.verifier import verify

TASK = Task(start=(0.0, 0.0, 0.0), goal=(40.0, 0.0), max_steps=600)
DT = TASK.dt


def circle(loops: float, speed: float = 5.0, r: float = 10.0) -> List[State]:
    n = int(loops * 2 * math.pi * r / (speed * DT))
    out = []
    for k in range(n + 1):
        th = -math.pi / 2 + k * speed * DT / r
        out.append(State(r * math.cos(th), r + r * math.sin(th), math.degrees(th) + 90, speed, k * DT))
    return out


def line(x0: float, x1: float, speed: float, k0: int = 0) -> List[State]:
    n = max(1, int(abs(x1 - x0) / (speed * DT)))
    return [State(x0 + (x1 - x0) * i / n, 0.0, 0.0, speed, (k0 + i) * DT) for i in range(n + 1)]


def test_honest_run_passes():
    r = run_episode(HeuristicPolicy(), TASK, FakeBackend())
    assert r["passed"], r["failures"]


def test_circling_is_rejected():
    assert "goal_not_reached" in verify(circle(1.0), TASK).failures


def test_loops_then_goal_rejected():
    traj = circle(2.0) + line(0.0, 40.0, 5.0, k0=len(circle(2.0)))
    f = verify(traj, TASK).failures
    assert "excessive_turning" in f and "inefficient_path" in f


def test_teleport_rejected():
    traj = [State(0, 0, 0, 0, 0.0), State(40, 0, 0, 0, DT)]
    assert "teleport" in verify(traj, TASK).failures


def test_speed_farming_rejected():
    assert "speed_limit" in verify(line(0.0, 40.0, 20.0), TASK).failures


def test_standing_still_rejected():
    traj = [State(0, 0, 0, 0, i * DT) for i in range(50)]
    assert "goal_not_reached" in verify(traj, TASK).failures


def test_fabricated_start_rejected():
    traj = [State(40.0, 0.0, 0, 0, i * DT) for i in range(5)]
    assert "bad_start" in verify(traj, TASK).failures


def test_faked_timestamps_rejected():
    traj = [State(s.x, s.y, s.yaw, s.speed, 0.0) for s in line(0.0, 40.0, 5.0)]
    assert "bad_timestamps" in verify(traj, TASK).failures


def test_collision_rejected():
    t = Task(start=(0, 0, 0), goal=(40, 0), obstacles=((20.0, 0.0, 3.0),))
    r = run_episode(HeuristicPolicy(), t, FakeBackend())
    assert not r["passed"] and "collision" in r["failures"]


def test_circling_earns_no_reward():
    traj = circle(1.0)
    assert sum(compute_reward(a, b, TASK) for a, b in zip(traj, traj[1:])) <= 0.0


def test_speed_alone_earns_no_reward():
    traj = [State(0, 0, 0, 25.0, i * DT) for i in range(20)]  # fast but going nowhere
    assert sum(compute_reward(a, b, TASK) for a, b in zip(traj, traj[1:])) < 0.0
