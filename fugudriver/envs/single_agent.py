"""Single-agent environments."""

from typing import Dict

import gymnasium as gym
import numpy as np
from pufferlib.spaces import joint_space
from pufferlib.ocean.drive.drive import Drive


class FuguSingleAgent(gym.Env):
    """
    A single-agent environment wrapping a PufferDrive environment.
    """
    def __init__(self, pd_config: Dict):
        super().__init__()
        if (na := pd_config.get("num_agents", 0) != 1):
            # log warning
            pd_config["num_agents"] = 1
        self.drive_env = Drive(**pd_config)

        # fix observation space bug
        self.drive_env.single_observation_space = gym.spaces.Box(-2, 2, (self.drive_env.num_obs,), dtype=np.float32)
        self.drive_env.observation_space = joint_space(self.drive_env.single_observation_space, self.drive_env.num_agents)


    @property
    def observation_space(self) -> gym.Space:
        return self.drive_env.single_observation_space


    @property
    def action_space(self) -> gym.Space:
        return self.drive_env.single_action_space


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs, _ = self.drive_env.reset(seed=(seed if seed is not None else 0)) # Note: PD always returns empty info from reset
        return obs[0], {}


    def step(self, action):
        obs, rewards, terms, truncs, infos = self.drive_env.step(action)
        info = infos[0] if len(infos) > 0 else {}
        return obs[0], rewards[0], terms[0], truncs[0], info


    def render(self):
        self.drive_env.render()


    def close(self):
        self.drive_env.close()

    @property
    def render_mode(self):
        return self.drive_env.render_mode
