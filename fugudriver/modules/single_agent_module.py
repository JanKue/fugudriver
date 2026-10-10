"""Default single-agent module."""

import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchMultiCategorical
from ray.rllib.utils.annotations import override


class DefaultSingleAgentModule(TorchRLModule, ValueFunctionAPI):
    """Re-implementation as RLlib RLModule of the default model in `PufferDrive/pufferlib/ocean/torch.py`"""
    @override(TorchRLModule)
    def setup(self):
        self.hidden_dim = self.model_config.get("hidden_dim")
        single_enc_dim = self.model_config.get("single_enc_dim")

        # These values are hardcoded in drive.h
        self.n_partner_objects = 31
        self.partner_features = 7
        self.n_road_objects = 128
        self.road_features = 7
        self.road_n_classes = 7
        road_features_with_onehot = (self.road_features - 1) + self.road_n_classes # last feature is expanded to onehot classes
        total_non_ego_dim = self.n_partner_objects * self.partner_features + self.n_road_objects * self.road_features
        self.ego_features = self.observation_space.shape[0] - total_non_ego_dim

        # Handle different actions spaces, depending on action_type
        self.action_is_continuous = isinstance(self.action_space, gym.spaces.Box)
        if self.action_is_continuous:
            self.action_dim = self.action_space.shape[0] * 2
        else:
            self.action_dim = self.action_space.nvec[0]
            self.action_dist_cls = TorchMultiCategorical.get_partial_dist_cls(space=self.action_space, input_lens=[self.action_dim])

        # Define encoder submodules
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_features, single_enc_dim),
            nn.LayerNorm(single_enc_dim),
            nn.Linear(single_enc_dim, single_enc_dim),
        )

        self.partner_encoder = nn.Sequential(
            nn.Linear(self.partner_features, single_enc_dim),
            nn.LayerNorm(single_enc_dim),
            nn.Linear(single_enc_dim, single_enc_dim)
        )

        self.road_encoder = nn.Sequential(
            nn.Linear(road_features_with_onehot, single_enc_dim),
            nn.LayerNorm(single_enc_dim),
            nn.Linear(single_enc_dim, single_enc_dim)
        )

        self.shared_emb = nn.Sequential(
            nn.GELU(),
            nn.Linear(3 * single_enc_dim, self.hidden_dim),
        )

        # Define decoding submodules
        self.value_fn = nn.Linear(self.hidden_dim, 1)
        self.actor = nn.Linear(self.hidden_dim, self.action_dim)


    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        embeddings = self._encode_observations(observations=batch[Columns.OBS])
        action_emb = self.actor(embeddings)
        return {
            Columns.EMBEDDINGS: embeddings,
            Columns.ACTION_DIST_INPUTS: action_emb,
        }


    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self._encode_observations(observations=batch[Columns.OBS])
        values = self.value_fn(embeddings)
        return values


    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Perform a single complete encoder pass for a given observation batch.
        
        :param observation: Tensor of shape [n_batch, n_obs]
        :returns: Tensor of shape [n_batch, hidden_dim]
        """
        ego_obs = observations[..., :self.ego_features].view(-1, self.ego_features) # [n_batch, ego_dim]
        ego_emb = self.ego_encoder(ego_obs) # [n_batch, single_enc_dim]

        partner_dim = self.n_partner_objects * self.partner_features
        partner_obs = observations[..., self.ego_features : self.ego_features+partner_dim] \
            .view(-1, self.n_partner_objects, self.partner_features) # [n_batch, n_partner_objects, partner_features]
        partner_emb = torch.amax(input=self.partner_encoder(partner_obs), dim=-2) # max-pool to [n_batch, single_enc_dim]

        road_dim = self.n_road_objects * self.road_features
        road_obs = observations[..., self.ego_features+partner_dim : self.ego_features+partner_dim+road_dim] \
            .view(-1, self.n_road_objects, self.road_features) # [n_batch, n_road_objects, road_features]
        road_continuous = road_obs[..., :-1] # [n_batch, n_road_objects, road_features-1]
        road_categorical = road_obs[..., -1] # [n_batch, n_road_objects, 1]
        road_onehot = F.one_hot(road_categorical.long(), self.road_n_classes) # [n_batch, n_road_objects, road_n_classes]
        road_obs_with_onehot = torch.cat([road_continuous, road_onehot], dim=-1) # [n_batch, n_road_objects, road_features_after_onehot]
        road_emb = torch.amax(input=self.road_encoder(road_obs_with_onehot), dim=-2) # max-pool to [n_batch, single_enc_dim]

        concat_emb = torch.cat([ego_emb, partner_emb, road_emb], dim=-1) # [n_batch, 3*single_enc_dim]
        return self.shared_emb(concat_emb)
