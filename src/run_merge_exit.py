import gymnasium as gym
import highway_env
import custom_env
from rules import make_env

gym.register_envs(highway_env)

RULESET = "basic"
POLICY = "baseline"
SEED = 1111
EPISODES = 5

env = make_env(
    RULESET,
    env_id="custom-v0",
    config={"vehicles_count": 10, "ramp_vehicles": 2},
    render_mode="human",
)


def act():
    if POLICY == "random":
        return env.action_space.sample()
    base = env.unwrapped
    actions = base.action_type.actions_indexes
    layout = base.layout_info
    idx = base.vehicle.target_lane_index
    lanes = base.config["lanes_count"]
    if layout.merge and idx == layout.merge:
        return actions["LANE_LEFT"]
    if layout.exit and idx[:2] == (layout.exit["from"], layout.exit["to"]) and idx[2] == lanes - 1:
        return actions["LANE_RIGHT"]
    return actions["IDLE"]


for episode in range(EPISODES):
    obs, info = env.reset(seed=SEED + episode)
    violations = {"gap": 0, "speed": 0, "pass": 0}
    total, steps, done = 0.0, 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(act())
        total += reward
        steps += 1
        for k, v in info.get("violations", {}).items():
            violations[k] += int(v)
        done = terminated or truncated
    print(
        f"episode={episode} steps={steps} return={total:.2f} crashed={info['crashed']} "
        f"success={info['success']} missed={info['missed']} violations={violations}"
    )

env.close()