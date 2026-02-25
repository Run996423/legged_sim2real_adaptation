import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticSupervisedLatent(nn.Module):
    is_recurrent = False

    def __init__(
            self,
            num_env_obs,
            num_critic_obs,
            num_actions,
            history_length,
            num_latent,
            num_supervised_latent,  # 假设latent的前num_supervised_latent维度是受到ground truth监督的
            encoder_hidden_dims=(512, 256, 128),
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            init_noise_std=1.0,
            noise_std_type: str = "scalar",
            **kwargs,
    ):
        if kwargs:
            print("ActorCriticSupervisedLatent.__init__ got unexpected arguments, which will be ignored: " +
                  str([key for key in kwargs.keys()]))
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.num_supervised_latent = num_supervised_latent

        # encoder
        encoder_layers = [
            nn.Linear(num_env_obs * history_length, encoder_hidden_dims[0]), activation]
        for layer_index in range(len(encoder_hidden_dims)):
            if layer_index == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], num_latent))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # Actor
        actor_layers = [nn.Linear(num_env_obs + num_latent, actor_hidden_dims[0]), activation]
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation]
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder: {self.encoder}")
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

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

    def update_distribution(self, obs_dict, compute_latent_loss=False):
        obs_history = obs_dict["obs_history"]  # (batch_size, history_length, obs_dim)
        current_obs = obs_dict["current_obs"]  # (batch_size, obs_dim)
        supervised_latent_ground_truth = obs_dict["supervised_latent_ground_truth"]
        latent = self.encoder(obs_history.flatten(1, 2))
        mean = self.actor(torch.cat((current_obs, latent), dim=-1))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)
        if compute_latent_loss:
            latent_loss = torch.nn.functional.mse_loss(supervised_latent_ground_truth,
                                                       latent[:, 0:self.num_supervised_latent], reduction='mean')
            return latent_loss
        else:
            return None

    def act(self, obs_dict, compute_latent_loss=False, **kwargs):
        latent_loss = self.update_distribution(obs_dict=obs_dict, compute_latent_loss=compute_latent_loss)
        return self.distribution.rsample(), latent_loss

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        obs_history = obs_dict["obs_history"]  # (batch_size, history_length, obs_dim)
        current_obs = obs_dict["current_obs"]  # (batch_size, obs_dim)
        latent = self.encoder(obs_history.flatten(1, 2))
        mean = self.actor(torch.cat((current_obs, latent), dim=-1))
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
