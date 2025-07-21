import os
import sys
import numpy as np
from stable_baselines3 import PPO
# モジュールパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from environment.Ball2Absorb import Ball2AbsorbEnv

# モジュールパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# XMLファイルのパス
here = os.path.dirname(__file__)
xml_path = os.path.abspath(os.path.join(here, "../../../../../../data/simulation_models/simple_model.xml"))

# モデル読み込み
model = PPO.load("ppo_ball2_absorb")

# 評価設定
n_episodes = 100
success_count = 0

for ep in range(n_episodes):
    env = Ball2AbsorbEnv(xml_path)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # Ball2が吸着に成功（r_absorbで+50）したら success にカウント
        if reward >= 50:
            success_count += 1
            break

print(f"\n✅ 成功回数: {success_count}/{n_episodes} ({success_count/n_episodes * 100:.1f}%)")

