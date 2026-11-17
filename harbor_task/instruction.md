# Navigate to the goal

Write `/app/policy.py` defining `act(obs: dict) -> dict`.

`obs` keys: `dist_to_goal` (m), `heading_error_deg` (positive => steer positive), `speed` (m/s),
`speed_limit` (m/s), `collided`, `lane_invasions`, `step`.

Return `{"throttle": 0..1, "steer": -1..1, "brake": 0..1}`.

Your policy is run on hidden start/goal pairs in a vehicle simulator. An episode passes only if the car
reaches the goal without collisions, stays under the speed limit, and drives a sensible path.
Reward = fraction of hidden episodes passed.
