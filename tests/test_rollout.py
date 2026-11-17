from rlnav.evaluate import make_tasks
from rlnav.agent import HeuristicPolicy, RandomPolicy
from rlnav.backend_fake import FakeBackend
from rlnav.rollout import run_batch, summarize


def test_heuristic_beats_random():
    tasks = make_tasks(10, seed=1)
    h = summarize(run_batch(lambda i: HeuristicPolicy(), tasks, FakeBackend))
    r = summarize(run_batch(lambda i: RandomPolicy(i), tasks, FakeBackend))
    assert h["pass_rate"] >= 0.8
    assert r["pass_rate"] == 0.0
