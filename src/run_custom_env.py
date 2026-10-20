import sys
import gymnasium as gym
import imageio.v2 as imageio
import highway_env
import custom_env

gym.register_envs(highway_env)

env = gym.make("custom-v0", render_mode="human")
obs, info = env.reset(seed=0)
print("obs shape:", obs.shape, "| actions:", env.unwrapped.action_type.actions_indexes)

total = 0.0
for step in range(300):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        break
        
env.close()
