"""CLI: python -m rlnav.evaluate --policy heuristic|random|llm --backend fake|carla --episodes N"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, List

from .agent import HeuristicPolicy, LLMPolicy, RandomPolicy
from .backend_fake import FakeBackend
from .rollout import run_batch, summarize
from .types import Task


def make_tasks(n: int, seed: int = 0) -> List[Task]:
    rng, tasks = random.Random(seed), []
    for _ in range(n):
        d, b = rng.uniform(30, 60), math.radians(rng.uniform(-180, 180))
        gx, gy = d * math.cos(b), d * math.sin(b)
        decoy = (gx / 2 - 10 * math.sin(b), gy / 2 + 10 * math.cos(b), 3.0)  # off the straight path
        tasks.append(Task(start=(0.0, 0.0, rng.uniform(-180, 180)), goal=(gx, gy), obstacles=(decoy,)))
    return tasks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="heuristic", choices=["heuristic", "random", "llm"])
    p.add_argument("--backend", default="fake", choices=["fake", "carla"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    seed = a.seed + int(os.environ.get("JOB_COMPLETION_INDEX", "0")) * 1000  # K8s indexed-job sharding
    tasks = make_tasks(a.episodes, seed)

    def backend_factory() -> Any:
        if a.backend == "carla":
            from .backend_carla import CarlaBackend

            return CarlaBackend(os.environ.get("CARLA_HOST", "localhost"), int(os.environ.get("CARLA_PORT", "2000")))
        return FakeBackend()

    def policy_factory(i: int) -> Any:
        return {"heuristic": HeuristicPolicy, "llm": LLMPolicy}.get(a.policy, lambda: RandomPolicy(seed + i))()

    results = run_batch(policy_factory, tasks, backend_factory, a.workers if a.backend == "fake" else 1)
    print(json.dumps({"policy": a.policy, "seed": seed, **summarize(results)}))


if __name__ == "__main__":
    main()
