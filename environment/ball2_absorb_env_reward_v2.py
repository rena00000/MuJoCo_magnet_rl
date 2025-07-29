
import numpy as np
from gymnasium import Env, spaces

class Ball2AbsorbEnv(Env):
    """Reward = forward progress + absorb success + ceiling proximity bonus."""

    def __init__(self, simulator, env_params, reward_params):
        self.sim = simulator
        self.max_steps = env_params["max_steps"]
        self.ceiling_z = env_params["ceiling_z"]
        self.reward_weights = reward_params

        self.action_space = spaces.Box(
            low=np.array(env_params["action_low"], dtype=np.float32),
            high=np.array(env_params["action_high"], dtype=np.float32),
            dtype=np.float32
        )

        obs_dim = 6
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.radius = 0.05
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.sim.reset()
        self.step_count = 0
        self.has_absorbed = False
        self.prev_x = self.sim.get_body_com("Ball2")[0]
        return self._get_obs(), {}

    def _get_obs(self):
        qpos, qvel = self.sim.get_state()
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def step(self, action):
        self.sim.set_action(action)
        self.sim.step()
        self.step_count += 1

        obs = self._get_obs()
        reward = self._calc_reward()

        done = self.has_absorbed or self.step_count >= self.max_steps
        return obs, reward, done, False, {}

    def _calc_reward(self):
        pos1 = self.sim.get_body_com("Ball1")
        pos2 = self.sim.get_body_com("Ball2")

        # 吸着成功チェック
        ds = self.ceiling_z - (pos2[2] + self.radius)
        success = (0 <= ds <= 0.02) and (pos2[0] > pos1[0])
        r_absorb = 1.0 if success else 0.0
        if success:
            self.has_absorbed = True

        # 右方向の移動量
        dx = max(pos2[0] - self.prev_x, 0.0)
        r_forward = dx
        self.prev_x = pos2[0]

        # 天井接近報酬（0 <= ds < 0.07 の範囲のみ）
        if 0 <= ds < 0.07:
            r_ball2_ceiling = 30.0 * ((ds / 0.07) - 1.0) ** 2
        else:
            r_ball2_ceiling = 0.0

        reward = (
            self.reward_weights.get("absorb", 100.0) * r_absorb +
            self.reward_weights.get("forward", 10.0) * r_forward +
            self.reward_weights.get("ball2_ceiling_bonus", 1.0) * r_ball2_ceiling
        )
        return reward
