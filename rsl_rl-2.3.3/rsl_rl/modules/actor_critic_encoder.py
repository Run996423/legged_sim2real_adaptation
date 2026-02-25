# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticEncoder(nn.Module):
    is_recurrent = False

    def __init__(
            self,
            num_env_obs,
            num_critic_obs,
            num_actions,
            num_encoder_obs,
            num_latent,
            encoder_hidden_dims,
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[256, 256, 256],
            activation="elu",
            init_noise_std=1.0,
            noise_std_type: str = "scalar",
            **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.num_encoder_obs = num_encoder_obs
        self.num_latent = num_latent
        self.num_env_obs = num_env_obs
        num_actor_obs = num_env_obs - num_encoder_obs + num_latent
        mlp_input_dim_e = num_encoder_obs
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Encoder
        encoder_layers = [nn.Linear(mlp_input_dim_e, encoder_hidden_dims[0]), activation]
        for layer_index in range(len(encoder_hidden_dims)):
            if layer_index == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], num_latent))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent = None

        # Policy
        actor_layers = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), activation]
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), activation]
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # assume that the head num_encoder_obs elem of observations is for encoder
        # compute latent
        latent = self.encoder(observations[..., 0:self.num_encoder_obs])
        self.latent = latent
        # compute mean
        mean = self.actor(torch.cat((latent, observations[..., self.num_encoder_obs:self.num_env_obs]), dim=-1))
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, requires_grad=False, **kwargs):
        self.update_distribution(observations)
        if requires_grad:
            # 使用 rsample (reparameterization trick) 保持梯度
            return self.distribution.rsample()
        else:
            # 普通采样，不记录梯度
            return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        self.latent = self.encoder(observations[..., 0:self.num_encoder_obs])
        mean = self.actor(torch.cat((self.latent, observations[..., self.num_encoder_obs:self.num_env_obs]), dim=-1))
        return mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
