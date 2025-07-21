import gym
from gym import spaces
import numpy as np
from mujoco import MjModel, MjData, mj_step, mj_forward


class Ball2AbsorbEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path):
        super().__init__()

        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)
        self.timestep = self.model.opt.timestep
        self.max_steps = 2000
        self.ceiling_z = 1.2

        self.ball2_id = self.model.body("Ball2").id
        self.ball2_joint_names = ["ball1_joint", "joint1", "joint2"]
        self.joint_ids = [self.model.joint(name).dofadr[0] for name in self.ball2_joint_names]
        self.qpos_indices = [self.model.joint(name).qposadr for name in self.ball2_joint_names]

        self.action_space = spaces.Box(low=np.array([-1.5, -2.5, -2.5]),
                                       high=np.array([1.5, 2.5, 2.5]), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.step_count = 0
        self.prev_x = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data = MjData(self.model)

        # 初期角度をランダムに設定（各関節 ±0.3 rad）
        for qpos_index in self.qpos_indices:
            self.data.qpos[qpos_index] = np.random.uniform(-0.3, 0.3)

        self.data.qvel[:] = np.random.uniform(-0.05, 0.05, size=self.data.qvel.shape)
        mj_forward(self.model, self.data)
        self.step_count = 0
        self.prev_x = self.data.xpos[self.ball2_id][0]
        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.data.xpos[self.ball2_id]
        vel = self.data.cvel[self.ball2_id][:3]
        return np.concatenate([pos, vel])

    def _compute_reward(self):
        pos = self.data.xpos[self.ball2_id]
        x, y, z = pos
        dc = np.sqrt(x ** 2 + y ** 2 + (z - self.ceiling_z) ** 2)
        ds = dc - 0.07

        r1 = 30 * ((ds / 0.07 - 1) ** 2) if 0 < ds < 0.07 else 0
        r_col = -20 - 100 * abs(ds) if ds < 0 else 0

        speed = np.linalg.norm(self.data.cvel[self.ball2_id][:3])
        r_static = -10 if speed < 1e-3 else 0

        delta_x = self.data.xpos[self.ball2_id][0] - self.prev_x
        self.prev_x = self.data.xpos[self.ball2_id][0]
        r_move = max(0, 5 * delta_x)

        r_absorb = 50 if ds < 0.05 and speed < 0.05 else 0

        return r1 + r_col + r_static + r_move + r_absorb, ds, speed, r_absorb > 0

    def step(self, action):
        for i, a in zip(self.joint_ids, action):
            self.data.qfrc_applied[i] = a

        mj_step(self.model, self.data)
        self.step_count += 1

        reward, ds, speed, absorbed = self._compute_reward()
        done = absorbed or self.step_count >= self.max_steps or self.data.xpos[self.ball2_id][2] < 0.2

        return self._get_obs(), reward, done, False, {}

    def render(self):
        pass

    def close(self):
        del self.data

