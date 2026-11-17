from types import SimpleNamespace

from rlnav.agent import SAFE_ACTION, LLMPolicy, parse_action

OBS = {"dist_to_goal": 30.0, "heading_error_deg": 10.0, "speed": 1.0, "speed_limit": 8.0}


def test_parse_plain_json():
    a = parse_action('{"throttle": 0.5, "steer": -0.2, "brake": 0}')
    assert (a.throttle, a.steer, a.brake) == (0.5, -0.2, 0.0)


def test_parse_json_inside_prose_and_fences():
    assert parse_action('Sure!\n```json\n{"throttle": 1, "steer": 0, "brake": 0}\n```').throttle == 1.0


def test_parse_clamps_out_of_range():
    a = parse_action('{"throttle": 99, "steer": -5, "brake": 3}')
    assert (a.throttle, a.steer, a.brake) == (1.0, -1.0, 1.0)


def test_garbage_falls_back_to_brake():
    assert parse_action("I refuse") == SAFE_ACTION
    assert parse_action('{"throttle": "abc"}') == SAFE_ACTION


class FakeClient:
    def __init__(self, text: str = '{"throttle": 0.4, "steer": 0.1, "brake": 0}', fail: bool = False) -> None:
        self.calls, self.text, self.fail = 0, text, fail
        self.messages = self

    def create(self, **_kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("api down")
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def test_llm_policy_respects_decision_interval():
    c = FakeClient()
    p = LLMPolicy(client=c, decision_interval=5)
    for _ in range(10):
        p.act(OBS)
    assert c.calls == 2


def test_llm_api_failure_does_not_crash():
    assert LLMPolicy(client=FakeClient(fail=True)).act(OBS) == SAFE_ACTION
