''' example from https://github.com/Farama-Foundation/HighwayEnv Usage, with custom ruleset in rules.py'''
import sys
import gymnasium as gym
from src.rules import make_env

ruleset = sys.argv[1] if len(sys.argv) > 1 else "basic"

env = make_env(ruleset, config={"lanes_count": 3}, render_mode="human")

obs, info = env.reset(seed=0)
violations = {"gap": 0, "speed": 0, "pass": 0}

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    for k, v in info.get("violations", {}).items():
        violations[k] += int(v)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
print(ruleset, violations)