from torch.optim import Adam
import torch

from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise


class DDPGAgent(object):
    def __init__(
        self,
        num_in_pol,
        num_out_pol,
        num_in_critic,
        hidden_dim=128,
        lr=3e-4,
        lr_critic=5e-4,
        lr_actor=1e-3,
        discrete_action=False,
        use_channel_attention=False,
        channel_attention_config=None,
    ):
        actor_hidden = int(hidden_dim)
        critic_hidden = int(max(hidden_dim * 2, hidden_dim))

        if use_channel_attention:
            raise ValueError("Channel-aware attention was a Chapter-5 communication module and has been removed.")
        self.policy = MLPNetwork(
            input_dim=num_in_pol,
            out_dim=num_out_pol,
            hidden_dim=actor_hidden,
            constrain_out=True,
            discrete_action=discrete_action,
            norm_in=True,
        )
        self.target_policy = MLPNetwork(
            input_dim=num_in_pol,
            out_dim=num_out_pol,
            hidden_dim=actor_hidden,
            constrain_out=True,
            discrete_action=discrete_action,
            norm_in=True,
        )

        self.critic1 = MLPNetwork(
            input_dim=num_in_critic,
            out_dim=1,
            hidden_dim=critic_hidden,
            constrain_out=False,
            discrete_action=discrete_action,
            norm_in=True,
        )
        self.target_critic1 = MLPNetwork(
            input_dim=num_in_critic,
            out_dim=1,
            hidden_dim=critic_hidden,
            constrain_out=False,
            discrete_action=discrete_action,
            norm_in=True,
        )
        self.critic2 = MLPNetwork(
            input_dim=num_in_critic,
            out_dim=1,
            hidden_dim=critic_hidden,
            constrain_out=False,
            discrete_action=discrete_action,
            norm_in=True,
        )
        self.target_critic2 = MLPNetwork(
            input_dim=num_in_critic,
            out_dim=1,
            hidden_dim=critic_hidden,
            constrain_out=False,
            discrete_action=discrete_action,
            norm_in=True,
        )
        self.rec_critic = MLPNetwork(
            input_dim=num_in_critic,
            out_dim=1,
            hidden_dim=critic_hidden,
            constrain_out=False,
            discrete_action=discrete_action,
            norm_in=True,
        )
        self.target_rec_critic = MLPNetwork(
            input_dim=num_in_critic,
            out_dim=1,
            hidden_dim=critic_hidden,
            constrain_out=False,
            discrete_action=discrete_action,
            norm_in=True,
        )
        self.safe_critic = MLPNetwork(
            input_dim=num_in_critic,
            out_dim=1,
            hidden_dim=critic_hidden,
            constrain_out=False,
            discrete_action=discrete_action,
            norm_in=True,
        )
        self.target_safe_critic = MLPNetwork(
            input_dim=num_in_critic,
            out_dim=1,
            hidden_dim=critic_hidden,
            constrain_out=False,
            discrete_action=discrete_action,
            norm_in=True,
        )

        self.initial_sigma = 0.2
        self.min_sigma = 0.02
        self.noise_decay = 2e-5
        self.current_sigma = self.initial_sigma
        self.noise = OUNoise(num_out_pol, sigma=self.initial_sigma)
        self.discrete_action = discrete_action

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic1, self.critic1)
        hard_update(self.target_critic2, self.critic2)
        hard_update(self.target_rec_critic, self.rec_critic)
        hard_update(self.target_safe_critic, self.safe_critic)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr_critic)
        self.rec_critic_optimizer = Adam(self.rec_critic.parameters(), lr=lr_critic)
        self.safe_critic_optimizer = Adam(self.safe_critic.parameters(), lr=lr_critic)

    def _policy_device(self):
        return next(self.policy.parameters()).device

    def sync_noise_device(self):
        self.noise.to(device=self._policy_device())

    def step(self, obs, explore=False):
        self.sync_noise_device()
        with torch.no_grad():
            action = self.policy(obs)
            if explore:
                noise = self.noise.sample().to(device=action.device, dtype=action.dtype)
                if action.ndim > 1:
                    noise = noise.unsqueeze(0).expand_as(action)
                action = torch.clamp(action + noise, -1.0, 1.0)
            if self.discrete_action:
                action = gumbel_softmax(action, hard=True) if explore else onehot_from_logits(action)
        return action

    def get_params(self):
        return {
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "policy_opt": self.policy_optimizer.state_dict(),
            "critic1": self.critic1.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "critic1_opt": self.critic1_optimizer.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "critic2_opt": self.critic2_optimizer.state_dict(),
            "rec_critic": self.rec_critic.state_dict(),
            "target_rec_critic": self.target_rec_critic.state_dict(),
            "rec_critic_opt": self.rec_critic_optimizer.state_dict(),
            "safe_critic": self.safe_critic.state_dict(),
            "target_safe_critic": self.target_safe_critic.state_dict(),
            "safe_critic_opt": self.safe_critic_optimizer.state_dict(),
        }

    def load_params(self, params):
        self.policy.load_state_dict(params["policy"])
        self.target_policy.load_state_dict(params["target_policy"])
        self.policy_optimizer.load_state_dict(params["policy_opt"])
        self.critic1.load_state_dict(params["critic1"])
        self.target_critic1.load_state_dict(params["target_critic1"])
        self.critic1_optimizer.load_state_dict(params["critic1_opt"])
        self.critic2.load_state_dict(params["critic2"])
        self.target_critic2.load_state_dict(params["target_critic2"])
        self.critic2_optimizer.load_state_dict(params["critic2_opt"])
        if "rec_critic" in params:
            self.rec_critic.load_state_dict(params["rec_critic"])
        if "target_rec_critic" in params:
            self.target_rec_critic.load_state_dict(params["target_rec_critic"])
        else:
            hard_update(self.target_rec_critic, self.rec_critic)
        if "rec_critic_opt" in params:
            self.rec_critic_optimizer.load_state_dict(params["rec_critic_opt"])
        if "safe_critic" in params:
            self.safe_critic.load_state_dict(params["safe_critic"])
        if "target_safe_critic" in params:
            self.target_safe_critic.load_state_dict(params["target_safe_critic"])
        else:
            hard_update(self.target_safe_critic, self.safe_critic)
        if "safe_critic_opt" in params:
            self.safe_critic_optimizer.load_state_dict(params["safe_critic_opt"])
        self.sync_noise_device()

    def scale_noise(self, sigma: float, multiply: bool = False):
        if multiply:
            self.noise.sigma *= sigma
        else:
            self.noise.sigma = sigma
        self.current_sigma = float(self.noise.sigma)

    @property
    def critic(self):
        return self.critic1

    @critic.setter
    def critic(self, val):
        self.critic1 = val

    @property
    def target_critic(self):
        return self.target_critic1

    @target_critic.setter
    def target_critic(self, val):
        self.target_critic1 = val

    @property
    def critic_optimizer(self):
        return self.critic1_optimizer

    @critic_optimizer.setter
    def critic_optimizer(self, val):
        self.critic1_optimizer = val
