# train_simple_model.py
import json
import os
import sys
sys.path.append('../')

from utils.paths_handling import rl_root_dir
from training.trainer import acromonk_training

# === パラメータファイルの指定 ===
param_path = os.path.join(rl_root_dir, 'training', 'parameters_simple_model.json')

# === 学習実行 ===
with open(param_path) as config:
    parameters = json.load(config)

acromonk_training(parameters)

#model.save("models/rl_model_2000000_steps")

