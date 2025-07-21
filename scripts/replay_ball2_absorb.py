import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

# モジュールパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from environment.Ball2Absorb import Ball2AbsorbEnv

# XMLファイルのパス
here = os.path.dirname(__file__)
xml_path = os.path.abspath(os.path.join(here, "../../../../../../data/simulation_models/simple_model.xml"))

# モデル読み込み
model = PPO.load("ppo_ball2_absorb")

# 10エピソード実行
n_episodes = 10

for ep in range(n_episodes):
    print(f"\n▶️ Episode {ep+1}")
    env = Ball2AbsorbEnv(xml_path)
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            viewer.sync()
            time.sleep(env.timestep)
            step += 1

        print(f"✅ Episode {ep+1} finished in {step} steps. Total reward = {total_reward:.2f}")

