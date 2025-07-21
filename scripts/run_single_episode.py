import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# 👇 モジュールパスを1階層上に追加（scripts → environment が読み込めるように）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from environment.Ball2Absorb import Ball2AbsorbEnv

# 📄 XMLファイルのパス（simple_model.xml）
here = os.path.dirname(__file__)
xml_path = os.path.abspath(os.path.join(here, "../../../../../../data/simulation_models/simple_model.xml"))

# 🌱 環境初期化
env = Ball2AbsorbEnv(xml_path)
obs, _ = env.reset()

# 🎥 viewer起動（リアルタイム表示）
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    done = False
    step = 0

    while viewer.is_running() and not done:
        # sin波トルクを各関節に適用
        action = np.array([
            1.0 * np.sin(2 * np.pi * 0.1 * step * env.timestep),
            2.0 * np.sin(2 * np.pi * 0.1 * step * env.timestep + np.pi / 4),
           -2.0 * np.sin(2 * np.pi * 0.1 * step * env.timestep + np.pi / 2)
        ])

        obs, reward, done, _ = env.step(action)
        viewer.sync()

        # Ball2の高さ・報酬・done表示
        ball2_z = obs[2]
        print(f"step={step:4d}, z={ball2_z:.3f}, reward={reward:.2f}, done={done}")

        step += 1
        time.sleep(env.timestep)  # 実時間と同期

print("\nEpisode finished!")

