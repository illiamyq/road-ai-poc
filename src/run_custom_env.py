import os
import gymnasium as gym
import highway_env
import custom_env
from PIL import Image, ImageDraw
from rules import make_env

gym.register_envs(highway_env)

RULESET = "basic"
POLICY = "baseline"
MODE = "gif"
GIF_MODE = "combined"
GIF_EPISODES = 10
FPS = 30
FRAME_SKIP = 5
PAUSE = 2
SEED = 555
EPISODES = 10
OUT_DIR = "outputs"
TAG = f"{RULESET}_{POLICY}_seed{SEED}"

env = make_env(
    RULESET,
    env_id="custom-v0",
    config={"vehicles_count": 10, "ramp_vehicles": 1},
    render_mode={"human": "human", "gif": "rgb_array", "none": None}[MODE],
)
base = env.unwrapped
frames = []
label = ""
tick = 0
recording = False


def keep(frame):
    global tick
    if tick % FRAME_SKIP == 0:
        image = Image.fromarray(frame)
        ImageDraw.Draw(image).text((6, 4), label, fill=(255, 255, 255))
        frames.append(image)
    tick += 1


def capture():
    if recording and base.viewer is not None and base.enable_auto_render:
        keep(base.render())


def add_pause(text):
    last = frames[-1].copy()
    ImageDraw.Draw(last).text((6, 20), text, fill=(255, 255, 0))
    frames.extend(last.copy() for _ in range(PAUSE))


def save_gif(path):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
    print(f"saved {path}: {len(frames)} frames")


def outcome(info):
    if info["crashed"]:
        return "CRASHED"
    if info["success"]:
        return "SUCCESS"
    if info["missed"]:
        return "MISSED EXIT"
    return "TIMEOUT"


def act():
    if POLICY == "random":
        return env.action_space.sample()
    actions = base.action_type.actions_indexes
    layout = base.layout_info
    idx = base.vehicle.target_lane_index
    lanes = base.config["lanes_count"]
    if layout.merge and idx == layout.merge:
        return actions["LANE_LEFT"]
    if layout.exit and idx[:2] == (layout.exit["from"], layout.exit["to"]) and idx[2] == lanes - 1:
        return actions["LANE_RIGHT"]
    return actions["IDLE"]


if MODE == "gif":
    os.makedirs(OUT_DIR, exist_ok=True)
    base._automatic_rendering = capture

for episode in range(EPISODES):
    recording = MODE == "gif" and episode < GIF_EPISODES
    label = f"episode {episode}  seed {SEED + episode}"
    tick = 0
    obs, info = env.reset(seed=SEED + episode)
    base.enable_auto_render = False
    if recording:
        keep(base.render())
    violations = {"gap": 0, "speed": 0, "pass": 0}
    total, steps, done = 0.0, 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(act())
        if recording:
            keep(base.render())
        total += reward
        steps += 1
        for k, v in info.get("violations", {}).items():
            violations[k] += int(v)
        done = terminated or truncated
    result = outcome(info)
    print(f"episode={episode} steps={steps} return={total:.2f} result={result} violations={violations}")
    if recording:
        add_pause(result)
        if GIF_MODE == "separate":
            save_gif(f"{OUT_DIR}/{TAG}_episode{episode}.gif")
            frames.clear()

if MODE == "gif" and GIF_MODE == "combined" and frames:
    save_gif(f"{OUT_DIR}/{TAG}_combined.gif")

env.close()