import gymnasium as gym
import highway_env
import numpy as np
gym.register_envs(highway_env)

RULESETS = {
    "none": None,
    "basic": {"max_speed": 25.0, "min_gap_s": 1.5, "w_gap": 0.2, "w_speed": 0.1, "w_pass": 0.1},
}


class RuleWrapper(gym.Wrapper):
    def __init__(self, env, max_speed, min_gap_s, w_gap, w_speed, w_pass):
        super().__init__(env)
        self.max_speed, self.min_gap_s = max_speed, min_gap_s
        self.w_gap, self.w_speed, self.w_pass = w_gap, w_speed, w_pass
        self.left = env.unwrapped.action_type.actions_indexes["LANE_LEFT"]

    def step(self, action):
        base = self.env.unwrapped
        ego = base.vehicle
        front, _ = base.road.neighbour_vehicles(ego)
        needs_pass = front is not None and front.speed < ego.speed and front.position[0] - ego.position[0] < 50
        lanes = base.config.get("lanes_count", 99)
        on_ramp = getattr(ego.lane, "forbidden", False) or ego.lane_index[2] >= lanes
        bad_pass = action == self.left and not needs_pass and not on_ramp

        obs, reward, terminated, truncated, info = self.env.step(action)

        ego = base.vehicle
        front, _ = base.road.neighbour_vehicles(ego)
        gap_s = (front.position[0] - ego.position[0]) / max(ego.speed, 1.0) if front is not None else np.inf
        v = {"gap": gap_s < self.min_gap_s, "speed": ego.speed > self.max_speed, "pass": bad_pass}

        reward -= self.w_gap * v["gap"] + self.w_speed * v["speed"] + self.w_pass * v["pass"]
        info["violations"] = v
        return obs, reward, terminated, truncated, info


def make_env(ruleset="none", env_id="highway-v0", **kwargs):
    env = gym.make(env_id, **kwargs)
    params = RULESETS[ruleset]
    return env if params is None else RuleWrapper(env, **params)