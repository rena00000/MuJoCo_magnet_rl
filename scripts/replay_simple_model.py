import sys
import os
import time
import json
import numpy as np
from stable_baselines3 import PPO
from mujoco import viewer  # 映像表示用

# --- パス設定 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- モジュールインポート ---
from environment.acromonk import AcroMonkEnv
from utils.make_mujoco_simulator import make_mujoco_simulator

# --- 設定ファイルとモデルパス ---
params_path = "../training/parameters_simple_model.json"
model_path = "../training/models/rl_model_2000000_steps"

# --- JSON読み込み ---
with open(params_path, "r") as f:
    parameters = json.load(f)

env_params = parameters["environment"]
reward_params = parameters["reward_setup"]
sim_params = parameters["simulation"]

# --- モデルXMLファイルの絶対パス解決（例：~/acromonk/data/simulation_models/simple_model.xml）
model_rel_path = sim_params["model_path"]
base_dir = os.path.expanduser("~/acromonk/data/simulation_models")
model_xml_path = os.path.join(base_dir, model_rel_path)

# --- シミュレータ・環境構築 ---
simulator = make_mujoco_simulator(model_xml_path, sim_params)
env = AcroMonkEnv(simulator, env_params, reward_params)

# --- モデル読み込み ---
model = PPO.load(model_path)

# --- 映像付きで10エピソード分を再生 ---
with viewer.launch_passive(env.sim.model, env.sim.data) as v:
    for ep in range(10):
        print(f"\n=== Episode {ep + 1} ===")
        obs = env.reset()
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            print(f"[Step {step}] Reward: {reward:.2f}")
            time.sleep(0.01)

            if done:
                print("Episode finished.")
                break

