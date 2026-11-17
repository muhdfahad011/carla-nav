"""Rollout orchestration: run episodes (optionally in parallel) and grade each with the verifier."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Sequence

from .env import Backend, NavEnv
from .types import Task
from .verifier import verify


def run_episode(policy: Any, task: Task, backend: Backend) -> Dict[str, Any]:
    env = NavEnv(backend, task)
    obs, total, done = env.reset(), 0.0, False
    while not done:
        obs, r, done, _ = env.step(policy.act(obs))
        total += r
    res = verify(env.trajectory, task)
    return {"passed": res.passed, "return": round(total, 3), "failures": res.failures, **res.metrics}


def run_batch(policy_factory: Callable[[int], Any], tasks: Sequence[Task],
              backend_factory: Callable[[], Backend], workers: int = 4) -> List[Dict[str, Any]]:
    def one(i: int) -> Dict[str, Any]:
        backend = backend_factory()
        try:
            return run_episode(policy_factory(i), tasks[i], backend)
        finally:
            backend.close()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, range(len(tasks))))


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    return {"episodes": n, "pass_rate": sum(r["passed"] for r in results) / n if n else 0.0,
            "mean_return": sum(r["return"] for r in results) / n if n else 0.0}
