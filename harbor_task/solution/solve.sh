#!/bin/bash
cat > /app/policy.py <<'PY'
def act(obs):
    steer = max(-1.0, min(1.0, obs["heading_error_deg"] / 30.0))
    target = min(obs["speed_limit"] * 0.6, 5.0) if obs["dist_to_goal"] > 10 else 2.0
    if obs["speed"] < target:
        return {"throttle": 0.5, "steer": steer, "brake": 0.0}
    return {"throttle": 0.0, "steer": steer, "brake": 0.3 if obs["speed"] > target + 1 else 0.0}
PY
