"""Multi-agent environments."""

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from pufferlib.spaces import joint_space
from pufferlib.ocean.drive.drive import Drive


class FuguMultiAgent(MultiAgentEnv):
    """
    A RLlib MultiAgentEnv wrapping a PufferDrive environment.

    Compatible with both old and new RLlib API stacks.
    """
    def __init__(self, pd_config: Dict):
        """
        :param pd_config: Config parameters for underlying PufferDrive env
        """
        n_agents = pd_config.get("num_agents", 0)
        if not n_agents > 1:
            raise ValueError(f"num_agents must be greater than 1. Got {n_agents} instead.")

        self.drive_env = Drive(**pd_config)

        # RLlib @OldAPIStack compatibility
        self._agent_ids = set(range(n_agents))
        self.observation_space = gym.spaces.Dict({aid: self.drive_env.single_observation_space for aid in self._agent_ids})
        self.action_spaces = gym.spaces.Dict({aid: self.drive_env.single_action_space for aid in self._agent_ids})
        # end @OldAPIStack section

        # RLlib new API
        self.possible_agents = list(range(n_agents))
        self.agents = self.possible_agents.copy()

        super().__init__()


    def get_observation_space(self, agent_id) -> gym.Space:
        return self.drive_env.single_observation_space


    def get_action_space(self, agent_id) -> gym.Space:
        return self.drive_env.single_action_space


    def reset(self, seed=None, options=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        super().reset(seed=seed, options=options)
        _obs, _ = self.drive_env.reset(seed=(seed if seed is not None else 0)) # Note: PD always returns empty info from reset
        # Convert obs to MultiAgentDict
        obs = {aid: ob for aid, ob in enumerate(_obs)}
        return obs, {}


    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        actions = np.array(list(action_dict.values()))
        _obs, _rewards, _terms, _truncs, _infos = self.drive_env.step(actions)
        # Convert outputs to MultiAgentDicts
        obs = {aid: ob for aid, ob in enumerate(_obs)}
        rewards = {aid: reward for aid, reward in enumerate(_rewards)}
        terms = {aid: term for aid, term in enumerate(_terms)} | {"__all__": _terms.all()}
        truncs = {aid: trunc for aid, trunc in enumerate(_truncs)} | {"__all__": _truncs.all()}
        infos = {0: _infos[0]} if len(_infos) > 0  else {}
        return obs, rewards, terms, truncs, infos


    def render(self):
        self.drive_env.render()


    def close(self):
        self.drive_env.close()


    @property
    def render_mode(self):
        return self.drive_env.render_mode
