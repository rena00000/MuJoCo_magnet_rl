import os
import sys
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

# モジュールパス追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from environment.Ball2Absorb import Ball2AbsorbEnv

# XMLファイルのパス
here = os.path.dirname(__file__)
xml_path = os.path.abspath(os.path.join(here, "../../../../../../data/simulation_models/simple_model.xml"))

# 環境作成
env = Ball2AbsorbEnv(xml_path)
#check_env(env, warn=True)  # Gym APIに準拠しているか確認

# 評価環境の設定
eval_env = Ball2AbsorbEnv(xml_path)
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)

# モデル作成（MLPポリシー）
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

# 学習開始
model.learn(total_timesteps=20000, callback=eval_callback)

# モデル保存
model.save("ppo_ball2_absorb")

print("\nTraining complete! Model saved as 'ppo_ball2_absorb'.")

