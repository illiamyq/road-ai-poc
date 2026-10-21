import sys
import gymnasium as gym
import highway_env
import custom_env
from rules import make_env

gym.register_envs(highway_env)

ruleset = sys.argv[1] if len(sys.argv) > 1 else "basic"
VEHICLES = 50

env = make_env(
    ruleset,
    env_id="custom-v0",
    config={"vehicles_count": VEHICLES},
    render_mode="human",
)

obs, info = env.reset(seed=0)
violations = {"gap": 0, "speed": 0, "pass": 0}
episodes = 0

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    for k, v in info.get("violations", {}).items():
        violations[k] += int(v)

    if terminated or truncated:
        episodes += 1
        obs, info = env.reset()

env.close()
print(ruleset, "episodes:", episodes, violations)