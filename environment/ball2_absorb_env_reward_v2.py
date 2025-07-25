import numpy as np
from gymnasium import Env, spaces


class Ball2AbsorbEnv(Env):
    def __init__(self, simulator, env_params, reward_params):
        self.sim = simulator
        self.max_steps = env_params["max_steps"]
        self.ceiling_z = env_params["ceiling_z"]
        self.action_space = self._make_action_space(env_params)
        self.reward_weights = reward_params

        self.radius = 0.05  # Ball2 半径[m]
        self.step_count = 0
        self.has_absorbed = False  # 吸着済みフラグ

        obs_dim = 6
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.reset()

    def _make_action_space(self, env_params):
        low = np.array(env_params["action_low"], dtype=np.float32)
        high = np.array(env_params["action_high"], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.sim.reset()
        self.step_count = 0
        self.has_absorbed = False
        self.prev_pos = self.sim.get_body_com("Ball2")
        return self._get_obs(), {}

    def _get_obs(self):
        qpos, qvel = self.sim.get_state()
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def step(self, action):
        self.sim.set_action(action)
        self.sim.step()
        self.step_count += 1

        obs = self._get_obs()
        reward = self._calculate_reward()

        pos1 = self.sim.get_body_com("Ball1")
        pos2 = self.sim.get_body_com("Ball2")
        ds = self.ceiling_z - pos2[2]
        done = False

        # --- episode termination ---
        if self.has_absorbed:
            done = True
        elif ds < -0.1:  # 落下など
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        pos1 = self.sim.get_body_com("Ball1")
        pos2 = self.sim.get_body_com("Ball2")
        ceiling_z = self.ceiling_z
        radius = self.radius

        ds = ceiling_z - pos2[2]
        dc = ds - radius
        d12 = np.linalg.norm(pos1 - pos2)

        # 吸着報酬（Ball1より右でかつ近ければ）
        if not self.has_absorbed and 0 <= ds <= 0.07 and pos2[0] > pos1[0]:
            r1 = 30 * ((ds / 0.07) - 1) ** 2
            self.has_absorbed = True
        else:
            r1 = 0.0

        # 衝突ペナルティ（ds < 0）
        r_col = -20 - 100 * abs(ds) if ds < 0 else 0.0

        # 支点近接ペナルティ
        r12 = -5 * (0.10 - d12) if d12 < 0.10 else 0.0

        # 静止ペナルティ
        v2 = self.sim.get_body_velocity("Ball2")
        speed = np.linalg.norm(v2)
        r_static = -0.3 if speed < 0.1 else 0.0

        total = (
            self.reward_weights["absorb"] * r1 +
            self.reward_weights["overlap_penalty"] * r12 +
            self.reward_weights["collision"] * r_col +
            self.reward_weights["static"] * r_static
        )

        print(f"[debug] ds = {ds:.4f}, dc = {dc:.4f}, d12 = {d12:.4f}, speed = {speed:.4f}, r1 = {r1:.2f}, r_col = {r_col:.2f}, r12 = {r12:.2f}, r_static = {r_static:.2f}")
        return total

