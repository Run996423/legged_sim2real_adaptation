from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.modules import ActorCriticSupervisedLatent
from rsl_rl.storage import RolloutStorageForSupervisedLatent


class PPOSupervisedLatent:
    policy: ActorCriticAE

    def __init__(
            self,
            policy,
            value_loss_coef=1.0,
            entropy_coef=0.01,
            latent_loss_coef=0.01,
            use_clipped_value_loss=True,
            clip_param=0.2,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            device="cpu",
    ):
        # device-related parameters
        self.device = device
        self.gpu_global_rank = 0
        self.gpu_world_size = 1

        # PPO components
        policy.to(self.device)
        self.policy = policy
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer_params = []
        for group in self.optimizer.param_groups:
            self.optimizer_params.extend(group['params'])
        # Create rollout storage
        self.storage: RolloutStorageForSupervisedLatent = None  # type: ignore
        self.transition = RolloutStorageForSupervisedLatent.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.latent_loss_coef = latent_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
            self, num_envs, num_transitions_per_env, obs_history_shape,
            current_obs_shape, privileged_obs_shape, supervised_latent_ground_truth_shape, actions_shape
    ):
        # create rollout storage
        self.storage = RolloutStorageForSupervisedLatent(
            num_envs,
            num_transitions_per_env,
            obs_history_shape,
            current_obs_shape,
            privileged_obs_shape,
            supervised_latent_ground_truth_shape,
            actions_shape,
            self.device,
        )

    def act(self, obs_dict, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        obs_history = obs_dict["obs_history"]  # (batch_size, history_length, obs_dim)
        current_obs = obs_dict["current_obs"]  # (batch_size, obs_dim)
        supervised_latent_ground_truth = obs_dict["supervised_latent_ground_truth"]
        # compute the actions and values
        action_sampled, _ = self.policy.act(obs_dict, compute_recon_loss=False)
        self.transition.actions = action_sampled.detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.obs_history = obs_history
        self.transition.current_obs = current_obs
        self.transition.privileged_observations = critic_obs
        self.transition.supervised_latent_ground_truth = supervised_latent_ground_truth
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        """"""
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_latent_loss = 0

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (obs_history_batch, current_observations_batch,
             privileged_observations_batch, supervised_latent_ground_truths_batch, actions_batch,
             target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
             ) in generator:

            # original batch size
            original_batch_size = obs_history_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            action_sampled, latent_loss = self.policy.act(obs_dict={
                "obs_history": obs_history_batch,
                "current_obs": current_observations_batch,
                "supervised_latent_ground_truth": supervised_latent_ground_truths_batch,
            }, compute_latent_loss=True)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(privileged_observations_batch)
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif self.desired_kl / 2.0 > kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Compute the gradients
            loss = self.latent_loss_coef * latent_loss + surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.optimizer_params, self.max_grad_norm)
            self.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_latent_loss += latent_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_latent_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "latent_supervise": mean_latent_loss,
        }

        return loss_dict
