# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class RolloutStorageForPPOVAE:
    class Transition:
        def __init__(self):
            self.obs_history = None
            self.current_obs = None
            self.privileged_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
            self,
            num_envs,
            num_transitions_per_env,
            obs_history_shape,
            current_obs_shape,
            privileged_obs_shape,
            actions_shape,
            device="cpu",
    ):
        # store inputs
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_history_shape = obs_history_shape
        self.current_obs_shape = current_obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        header = (num_transitions_per_env, num_envs)
        self.obs_histories = torch.zeros(*header, *obs_history_shape, device=self.device)
        self.current_observations = torch.zeros(*header, *current_obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(*header, *privileged_obs_shape, device=self.device)
        self.rewards = torch.zeros(*header, 1, device=self.device)
        self.actions = torch.zeros(*header, *actions_shape, device=self.device)
        self.dones = torch.zeros(*header, 1, device=self.device).byte()
        self.values = torch.zeros(*header, 1, device=self.device)
        self.actions_log_prob = torch.zeros(*header, 1, device=self.device)
        self.mu = torch.zeros(*header, *actions_shape, device=self.device)
        self.sigma = torch.zeros(*header, *actions_shape, device=self.device)
        self.returns = torch.zeros(*header, 1, device=self.device)
        self.advantages = torch.zeros(*header, 1, device=self.device)

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.obs_histories[self.step].copy_(transition.obs_history)
        self.current_observations[self.step].copy_(transition.current_obs)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        # increment the counter
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        obs_histories = self.obs_histories.flatten(0, 1)
        current_observations = self.current_observations.flatten(0, 1)
        privileged_observations = self.privileged_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_history_batch = obs_histories[batch_idx]
                current_observations_batch = current_observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # yield the mini-batch
                yield (obs_history_batch, current_observations_batch,
                       privileged_observations_batch, actions_batch,
                       target_values_batch, advantages_batch, returns_batch,
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch)
