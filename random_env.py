from tf_agents.environments.py_environment import PyEnvironment
import numpy as np
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec


class RandomEnvironment(PyEnvironment):
	def __init__(self, action_dim=6, obs_dim=12, horizon=125, seed=7):
		np.random.seed(seed)
		self.horizon = horizon
		self.idx = 0
		self._episode_ended = False
		self.action_dim = action_dim
		self.obs_dim = obs_dim
		self._action_spec = array_spec.BoundedArraySpec(shape=(self.action_dim, ), dtype=np.float32, minimum=-1.0,
                                                    maximum=1.0)
		self._obs_spec = array_spec.BoundedArraySpec(shape=(self.obs_dim, ), dtype=np.float32, minimum=-5.0, maximum=5.0)
		super().__init__()

	def action_spec(self):
		return self._action_spec

	def observation_spec(self):
		return self._obs_spec

	def _reset(self) -> ts.TimeStep:
		self._episode_ended = False
		self.idx = 0
		return ts.restart(np.random.uniform(-5.0, 5.0, size=self.obs_dim).astype(np.float32))

	def _step(self, action) -> ts.TimeStep:
		if self._episode_ended:
			return self.reset()

		self.idx += 1
		if self.idx >= self.horizon:
			self._episode_ended = True

		obs = np.random.uniform(-5.0, 5.0, size=self.obs_dim).astype(np.float32)
		reward = np.random.uniform(-1.0, 1.0)
		if self._episode_ended:
			return ts.termination(obs, reward)
		else:
			return ts.transition(obs, reward, 0.99)
