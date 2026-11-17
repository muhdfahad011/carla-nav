"""Runs the agent's /app/policy.py on HIDDEN tasks with the trusted env + verifier. Writes reward."""
import importlib.util
import sys
from pathlib import Path

from rlnav.backend_fake import FakeBackend
from rlnav.evaluate import make_tasks
from rlnav.rollout import run_episode
from rlnav.types import Action

HIDDEN_SEED, N = 987654, 10  # not visible to the agent (/tests is mounted only at verification time)


class FilePolicy:
    def __init__(self, fn):
        self.fn = fn

    def act(self, obs):
        try:
            d = self.fn(dict(obs))
            return Action(float(d["throttle"]), float(d["steer"]), float(d["brake"])).clamped()
        except Exception:
            return Action(brake=1.0)  # broken policy behaviour = stop, never crash the grader


def main() -> None:
    reward = 0.0
    try:
        spec = importlib.util.spec_from_file_location("policy", "/app/policy.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        res = [run_episode(FilePolicy(mod.act), t, FakeBackend()) for t in make_tasks(N, HIDDEN_SEED)]
        reward = sum(r["passed"] for r in res) / N
        print({"pass_rate": reward, "failures": [r["failures"] for r in res if not r["passed"]]})
    except Exception as e:
        print("policy failed to load:", e, file=sys.stderr)
    Path("/logs/verifier/reward.txt").write_text(str(reward))


if __name__ == "__main__":
    main()
