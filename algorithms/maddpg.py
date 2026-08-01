
import os
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F

from utils.misc import soft_update, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent


class MADDPG(object):
    def __init__(
        self,
        agent_init_params,
        alg_types,
        gamma=0.95,
        tau=5e-3,
        lr=0.01,
        lr_actor=None,
        lr_critic=None,
        hidden_dim=64,
        discrete_action=False,
        agent_role_names=None,
        agent_noise_sigmas=None,
        residual_action_reg=1e-2,
        use_tail_weighted_training=False,
        use_tail_replay_priority=True,
        use_tail_critic_loss=True,
        use_tail_actor_loss=True,
        tail_alpha=0.20,
        lambda_tail_critic=0.75,
        lambda_tail_actor=0.35,
        tail_weight_strength=2.0,
    ):
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.lr_actor = lr if lr_actor is None else lr_actor
        self.lr_critic = lr if lr_critic is None else lr_critic
        self.hidden_dim = hidden_dim
        self.discrete_action = discrete_action
        self.agent_role_names = agent_role_names or [f"agent_{i}" for i in range(self.nagents)]
        self.residual_action_reg = float(residual_action_reg)
        self.use_tail_weighted_training = bool(use_tail_weighted_training)
        self.use_tail_replay_priority = bool(use_tail_replay_priority)
        self.use_tail_critic_loss = bool(use_tail_critic_loss)
        self.use_tail_actor_loss = bool(use_tail_actor_loss)
        self.tail_alpha = float(max(1e-3, min(1.0, tail_alpha)))
        self.lambda_tail_critic = float(max(0.0, lambda_tail_critic))
        self.lambda_tail_actor = float(max(0.0, lambda_tail_actor))
        self.tail_weight_strength = float(max(0.0, tail_weight_strength))
        self.last_tail_fraction = 0.0
        self.last_tail_score = 0.0
        self.last_tail_weight = 1.0

        self.agents = [
            DDPGAgent(
                lr=lr,
                lr_actor=self.lr_actor,
                lr_critic=self.lr_critic,
                discrete_action=discrete_action,
                hidden_dim=hidden_dim,
                **params,
            )
            for params in agent_init_params
        ]

        if agent_noise_sigmas is not None:
            for ag, sigma in zip(self.agents, agent_noise_sigmas):
                ag.scale_noise(float(sigma), multiply=False)

        self.device = torch.device("cpu")
        self.niter = 0
        self.init_dict = None
        self._cached_sample_key = None
        self._cached_sample = None

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        device_str = str(device).lower()
        if device_str == "gpu":
            device_str = "cuda"
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_str)

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=device, non_blocking=True)

    @staticmethod
    def _recursive_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: MADDPG._recursive_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MADDPG._recursive_to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(MADDPG._recursive_to_cpu(v) for v in obj)
        return obj

    @staticmethod
    def _clone_agent_params_to_cpu(agent: DDPGAgent):
        return {
            "policy": MADDPG._recursive_to_cpu(agent.policy.state_dict()),
            "target_policy": MADDPG._recursive_to_cpu(agent.target_policy.state_dict()),
            "policy_opt": MADDPG._recursive_to_cpu(agent.policy_optimizer.state_dict()),
            "critic1": MADDPG._recursive_to_cpu(agent.critic1.state_dict()),
            "target_critic1": MADDPG._recursive_to_cpu(agent.target_critic1.state_dict()),
            "critic1_opt": MADDPG._recursive_to_cpu(agent.critic1_optimizer.state_dict()),
            "critic2": MADDPG._recursive_to_cpu(agent.critic2.state_dict()),
            "target_critic2": MADDPG._recursive_to_cpu(agent.target_critic2.state_dict()),
            "critic2_opt": MADDPG._recursive_to_cpu(agent.critic2_optimizer.state_dict()),
            "rec_critic": MADDPG._recursive_to_cpu(agent.rec_critic.state_dict()),
            "target_rec_critic": MADDPG._recursive_to_cpu(agent.target_rec_critic.state_dict()),
            "rec_critic_opt": MADDPG._recursive_to_cpu(agent.rec_critic_optimizer.state_dict()),
            "safe_critic": MADDPG._recursive_to_cpu(agent.safe_critic.state_dict()),
            "target_safe_critic": MADDPG._recursive_to_cpu(agent.target_safe_critic.state_dict()),
            "safe_critic_opt": MADDPG._recursive_to_cpu(agent.safe_critic_optimizer.state_dict()),
        }

    def _move_all_to(self, device: torch.device) -> None:
        if self.device == device:
            return
        for a in self.agents:
            a.policy.to(device)
            a.target_policy.to(device)
            a.critic1.to(device)
            a.target_critic1.to(device)
            a.critic2.to(device)
            a.target_critic2.to(device)
            a.rec_critic.to(device)
            a.target_rec_critic.to(device)
            a.safe_critic.to(device)
            a.target_safe_critic.to(device)
            self._move_optimizer_state(a.policy_optimizer, device)
            self._move_optimizer_state(a.critic1_optimizer, device)
            self._move_optimizer_state(a.critic2_optimizer, device)
            self._move_optimizer_state(a.rec_critic_optimizer, device)
            self._move_optimizer_state(a.safe_critic_optimizer, device)
            if hasattr(a, "sync_noise_device"):
                a.sync_noise_device()
        self.device = device
        self._clear_sample_cache()

    def _clear_sample_cache(self):
        self._cached_sample_key = None
        self._cached_sample = None

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def reset_noise(self):
        for agent in self.agents:
            agent.noise.reset()

    def _ensure_obs_tensor(self, obs: Any) -> torch.Tensor:
        if torch.is_tensor(obs):
            tensor = obs.to(device=self.device, dtype=torch.float32, non_blocking=True)
        else:
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def step(self, observations, explore=False):
        with torch.no_grad():
            proc_obs = [self._ensure_obs_tensor(obs) for obs in observations]
            return [a.step(obs, explore=explore) for a, obs in zip(self.agents, proc_obs)]

    def step_residual(self, observations, explore=False):
        return self.step(observations, explore=explore)

    def _unpack_sample(self, sample):
        tail_scores = comm_risks = scenario_ids = None
        graph_meta = message_meta = belief_meta = None
        task_found_masks = executor_assigned_masks = post_found_masks = near_target_masks = executor_nav_distances = None
        recovery_costs = safety_costs = smooth_costs = None
        if len(sample) == 22:
            (
                obs, acs, rews, next_obs, dones, weights, indices, success_flags,
                tail_scores, comm_risks, scenario_ids, graph_meta, message_meta, belief_meta,
                task_found_masks, executor_assigned_masks, post_found_masks, near_target_masks,
                executor_nav_distances, recovery_costs, safety_costs, smooth_costs,
            ) = sample
        elif len(sample) == 17:
            obs, acs, rews, next_obs, dones, weights, indices, success_flags, tail_scores, comm_risks, scenario_ids, graph_meta, message_meta, belief_meta, recovery_costs, safety_costs, smooth_costs = sample
        elif len(sample) == 14:
            obs, acs, rews, next_obs, dones, weights, indices, success_flags, tail_scores, comm_risks, scenario_ids, graph_meta, message_meta, belief_meta = sample
        elif len(sample) == 11:
            obs, acs, rews, next_obs, dones, weights, indices, success_flags, tail_scores, comm_risks, scenario_ids = sample
        elif len(sample) == 8:
            obs, acs, rews, next_obs, dones, weights, indices, success_flags = sample
        elif len(sample) == 6:
            obs, acs, rews, next_obs, dones, weights = sample
            indices, success_flags = None, None
        else:
            raise ValueError(f"Unexpected sample length: {len(sample)}")
        return (
            obs, acs, rews, next_obs, dones, weights, indices, success_flags,
            tail_scores, comm_risks, scenario_ids, graph_meta, message_meta, belief_meta,
            task_found_masks, executor_assigned_masks, post_found_masks, near_target_masks,
            executor_nav_distances, recovery_costs, safety_costs, smooth_costs,
        )

    def _to_device(self, tensors, dtype=torch.float32):
        out = []
        for x in tensors:
            if torch.is_tensor(x):
                out.append(x.to(device=self.device, dtype=dtype, non_blocking=True))
            else:
                out.append(torch.as_tensor(x, dtype=dtype, device=self.device))
        return out

    def _prepare_sample(self, sample):
        key = (id(sample), self.device)
        if self._cached_sample_key == key and self._cached_sample is not None:
            return self._cached_sample

        (
            obs, acs, rews, next_obs, dones, weights, indices, success_flags,
            tail_scores, comm_risks, scenario_ids, graph_meta, message_meta, belief_meta,
            task_found_masks, executor_assigned_masks, post_found_masks, near_target_masks,
            executor_nav_distances, recovery_costs, safety_costs, smooth_costs,
        ) = self._unpack_sample(sample)
        obs = self._to_device(obs)
        acs = self._to_device(acs)
        rews = self._to_device(rews)
        next_obs = self._to_device(next_obs)
        dones = self._to_device(dones)

        if torch.is_tensor(weights):
            weights = weights.to(device=self.device, dtype=torch.float32, non_blocking=True)
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        weights = weights.view(-1, 1)
        batch_size = weights.shape[0]
        if tail_scores is None:
            tail_scores = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
        elif torch.is_tensor(tail_scores):
            tail_scores = tail_scores.to(device=self.device, dtype=torch.float32, non_blocking=True).view(-1)
        else:
            tail_scores = torch.as_tensor(tail_scores, dtype=torch.float32, device=self.device).view(-1)
        if comm_risks is None:
            comm_risks = tail_scores.clone()
        elif torch.is_tensor(comm_risks):
            comm_risks = comm_risks.to(device=self.device, dtype=torch.float32, non_blocking=True).view(-1)
        else:
            comm_risks = torch.as_tensor(comm_risks, dtype=torch.float32, device=self.device).view(-1)
        if scenario_ids is None:
            scenario_ids = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        elif torch.is_tensor(scenario_ids):
            scenario_ids = scenario_ids.to(device=self.device, dtype=torch.long, non_blocking=True).view(-1)
        else:
            scenario_ids = torch.as_tensor(scenario_ids, dtype=torch.long, device=self.device).view(-1)

        def prep_cost(cost):
            if cost is None:
                return torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
            if torch.is_tensor(cost):
                cost_t = cost.to(device=self.device, dtype=torch.float32, non_blocking=True).view(-1, 1)
            else:
                cost_t = torch.as_tensor(cost, dtype=torch.float32, device=self.device).view(-1, 1)
            if cost_t.shape[0] != batch_size:
                if cost_t.shape[0] == 1:
                    cost_t = cost_t.expand(batch_size, 1)
                else:
                    padded = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
                    n = min(batch_size, int(cost_t.shape[0]))
                    padded[:n].copy_(cost_t[:n])
                    cost_t = padded
            return torch.nan_to_num(cost_t, nan=0.0, posinf=0.0, neginf=0.0)

        def prep_mask(mask, default=1.0):
            if mask is None:
                return torch.full((batch_size, 1), float(default), dtype=torch.float32, device=self.device)
            if torch.is_tensor(mask):
                mask_t = mask.to(device=self.device, dtype=torch.float32, non_blocking=True).view(-1, 1)
            else:
                mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device).view(-1, 1)
            if mask_t.shape[0] != batch_size:
                if mask_t.shape[0] == 1:
                    mask_t = mask_t.expand(batch_size, 1)
                else:
                    padded = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
                    n = min(batch_size, int(mask_t.shape[0]))
                    padded[:n].copy_(mask_t[:n])
                    mask_t = padded
            return torch.nan_to_num(mask_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        mask_available = task_found_masks is not None and post_found_masks is not None and near_target_masks is not None
        task_found_masks = prep_mask(task_found_masks, default=1.0)
        executor_assigned_masks = prep_mask(executor_assigned_masks, default=1.0)
        post_found_masks = prep_mask(post_found_masks, default=1.0)
        near_target_masks = prep_mask(near_target_masks, default=1.0)
        executor_nav_distances = prep_cost(executor_nav_distances)
        recovery_costs = prep_cost(recovery_costs)
        safety_costs = prep_cost(safety_costs)
        smooth_costs = prep_cost(smooth_costs)

        def prep_meta(meta, dim=16):
            if meta is None:
                return torch.zeros((batch_size, dim), dtype=torch.float32, device=self.device)

            if torch.is_tensor(meta):
                t = meta.to(device=self.device, dtype=torch.float32, non_blocking=True)
            else:
                t = torch.as_tensor(meta, dtype=torch.float32, device=self.device)

            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

            if t.dim() == 1:
                if t.numel() == dim:
                    t = t.view(1, dim).expand(batch_size, dim)
                elif t.numel() == batch_size * dim:
                    t = t.view(batch_size, dim)
                else:
                    flat = t.reshape(-1)
                    out = torch.zeros((batch_size, dim), dtype=torch.float32, device=self.device)
                    n = min(dim, int(flat.numel()))
                    if n > 0:
                        out[:, :n] = flat[:n].view(1, n).expand(batch_size, n)
                    return out
            elif t.dim() == 2:
                if t.shape[0] != batch_size:
                    if t.shape[0] == 1:
                        t = t.expand(batch_size, t.shape[1])
                    else:
                        t = t[:batch_size]
                        if t.shape[0] < batch_size:
                            pad_rows = torch.zeros((batch_size - t.shape[0], t.shape[1]), dtype=torch.float32, device=self.device)
                            t = torch.cat([t, pad_rows], dim=0)
            else:
                t = t.reshape(batch_size, -1) if t.numel() % max(1, batch_size) == 0 else t.reshape(1, -1).expand(batch_size, -1)

            if t.shape[1] < dim:
                pad = torch.zeros((batch_size, dim - t.shape[1]), dtype=torch.float32, device=self.device)
                t = torch.cat([t, pad], dim=1)
            elif t.shape[1] > dim:
                t = t[:, :dim]

            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        graph_meta = prep_meta(graph_meta, 16)
        message_meta = prep_meta(message_meta, 16)
        belief_meta = prep_meta(belief_meta, 16)
        tail_weights, tail_mask = self._build_tail_weights(tail_scores)

        with torch.no_grad():
            next_acs = self._build_target_actions(next_obs)
            target_vf_in = torch.cat((*next_obs, *next_acs), dim=1)

        joint_vf_in = torch.cat((*obs, *acs), dim=1) if all(alg == "MADDPG" for alg in self.alg_types) else None

        packed = {
            "obs": obs,
            "acs": acs,
            "rews": rews,
            "next_obs": next_obs,
            "dones": dones,
            "weights": weights,
            "indices": indices,
            "success_flags": success_flags,
            "tail_scores": tail_scores.view(-1, 1),
            "comm_risks": comm_risks.view(-1, 1),
            "scenario_ids": scenario_ids,
            "graph_meta": graph_meta,
            "message_meta": message_meta,
            "belief_meta": belief_meta,
            "task_found_masks": task_found_masks,
            "executor_assigned_masks": executor_assigned_masks,
            "post_found_masks": post_found_masks,
            "near_target_masks": near_target_masks,
            "executor_nav_distances": executor_nav_distances,
            "recovery_costs": recovery_costs,
            "safety_costs": safety_costs,
            "smooth_costs": smooth_costs,
            "mask_available": bool(mask_available),
            "tail_weights": tail_weights.view(-1, 1),
            "tail_mask": tail_mask.view(-1, 1),
            "next_acs": next_acs,
            "target_vf_in": target_vf_in,
            "joint_vf_in": joint_vf_in,
        }
        self._cached_sample_key = key
        self._cached_sample = packed
        return packed

    def configure_tail_training(
        self,
        use_tail_weighted_training=None,
        use_tail_replay_priority=None,
        use_tail_critic_loss=None,
        use_tail_actor_loss=None,
        tail_alpha=None,
        lambda_tail_critic=None,
        lambda_tail_actor=None,
        tail_weight_strength=None,
    ):
        if use_tail_weighted_training is not None:
            self.use_tail_weighted_training = bool(use_tail_weighted_training)
        if use_tail_replay_priority is not None:
            self.use_tail_replay_priority = bool(use_tail_replay_priority)
        if use_tail_critic_loss is not None:
            self.use_tail_critic_loss = bool(use_tail_critic_loss)
        if use_tail_actor_loss is not None:
            self.use_tail_actor_loss = bool(use_tail_actor_loss)
        if tail_alpha is not None:
            self.tail_alpha = float(max(1e-3, min(1.0, tail_alpha)))
        if lambda_tail_critic is not None:
            self.lambda_tail_critic = float(max(0.0, lambda_tail_critic))
        if lambda_tail_actor is not None:
            self.lambda_tail_actor = float(max(0.0, lambda_tail_actor))
        if tail_weight_strength is not None:
            self.tail_weight_strength = float(max(0.0, tail_weight_strength))

    def _build_tail_weights(self, tail_scores):
        tail_scores = tail_scores.view(-1).clamp(0.0, 1.0)
        if (not self.use_tail_weighted_training) or self.tail_weight_strength <= 0.0 or tail_scores.numel() == 0:
            ones = torch.ones_like(tail_scores)
            zeros = torch.zeros_like(tail_scores)
            self.last_tail_fraction = 0.0
            self.last_tail_score = float(tail_scores.mean().item()) if tail_scores.numel() > 0 else 0.0
            self.last_tail_weight = 1.0
            return ones, zeros
        k = max(1, int(torch.ceil(torch.tensor(float(tail_scores.numel()) * self.tail_alpha, device=tail_scores.device)).item()))
        if k >= tail_scores.numel():
            threshold = tail_scores.min()
        else:
            threshold = torch.topk(tail_scores, k=k, largest=True).values[-1]
        tail_mask = (tail_scores >= threshold).to(dtype=tail_scores.dtype)
        tail_weights = 1.0 + self.tail_weight_strength * tail_mask * tail_scores
        self.last_tail_fraction = float(tail_mask.mean().item())
        self.last_tail_score = float(tail_scores.mean().item())
        self.last_tail_weight = float(tail_weights.mean().item())
        return tail_weights, tail_mask

    def _build_target_actions(self, next_obs):
        if self.discrete_action:
            return [onehot_from_logits(pi(no)) for pi, no in zip(self.target_policies, next_obs)]
        return [torch.clamp(pi(no), -1.0, 1.0) for pi, no in zip(self.target_policies, next_obs)]

    def _build_policy_action(self, agent_i, obs_i, explore_gumbel=True):
        ag = self.agents[agent_i]
        if self.discrete_action:
            logits = ag.policy(obs_i)
            act = gumbel_softmax(logits, hard=True) if explore_gumbel else onehot_from_logits(logits)
            return logits, act
        raw = torch.clamp(ag.policy(obs_i), -1.0, 1.0)
        return raw, raw

    def _compute_target_q(self, ag: DDPGAgent, target_vf_in, rew_i, done_i):
        with torch.no_grad():
            q1_next = ag.target_critic1(target_vf_in)
            q2_next = ag.target_critic2(target_vf_in)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rew_i.view(-1, 1) + self.gamma * min_q_next * (1 - done_i.view(-1, 1))
            return torch.clamp(target_q, -10.0, 10.0)

    def _critic_input(self, batch, agent_i):
        obs = batch["obs"]
        acs = batch["acs"]
        if self.alg_types[agent_i] == "MADDPG":
            return batch["joint_vf_in"] if batch["joint_vf_in"] is not None else torch.cat((*obs, *acs), dim=1)
        return torch.cat((obs[agent_i], acs[agent_i]), dim=1)

    def update_critic_only(self, sample, agent_i):
        batch = self._prepare_sample(sample)
        ag = self.agents[agent_i]
        target_q = self._compute_target_q(ag, batch["target_vf_in"], batch["rews"][agent_i], batch["dones"][agent_i])
        vf_in = self._critic_input(batch, agent_i)

        q1 = ag.critic1(vf_in)
        q2 = ag.critic2(vf_in)
        td_error = (target_q - q1).detach().squeeze(-1)

        loss1 = F.smooth_l1_loss(q1, target_q, reduction="none")
        loss2 = F.smooth_l1_loss(q2, target_q, reduction="none")
        td_loss = loss1 + loss2
        vf_loss = (td_loss * batch["weights"]).mean()
        if self.use_tail_weighted_training and self.use_tail_critic_loss and self.lambda_tail_critic > 0.0:
            tail_excess = (batch["tail_weights"] - 1.0).clamp_min(0.0)
            denom = tail_excess.sum().clamp_min(1e-6)
            tail_vf_loss = (td_loss * batch["weights"] * tail_excess).sum() / denom
            vf_loss = vf_loss + self.lambda_tail_critic * tail_vf_loss

        ag.critic1_optimizer.zero_grad(set_to_none=True)
        ag.critic2_optimizer.zero_grad(set_to_none=True)
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(ag.critic2.parameters(), 0.5)
        ag.critic1_optimizer.step()
        ag.critic2_optimizer.step()

        return float(vf_loss.item()), td_error

    def update(self, sample, agent_i, parallel=False, logger=None):
        batch = self._prepare_sample(sample)
        ag = self.agents[agent_i]
        target_q = self._compute_target_q(ag, batch["target_vf_in"], batch["rews"][agent_i], batch["dones"][agent_i])
        vf_in = self._critic_input(batch, agent_i)

        q1 = ag.critic1(vf_in)
        q2 = ag.critic2(vf_in)
        td_error = (target_q - q1).detach().squeeze(-1)

        loss1 = F.smooth_l1_loss(q1, target_q, reduction="none")
        loss2 = F.smooth_l1_loss(q2, target_q, reduction="none")
        td_loss = loss1 + loss2
        vf_loss = (td_loss * batch["weights"]).mean()
        if self.use_tail_weighted_training and self.use_tail_critic_loss and self.lambda_tail_critic > 0.0:
            tail_excess = (batch["tail_weights"] - 1.0).clamp_min(0.0)
            denom = tail_excess.sum().clamp_min(1e-6)
            tail_vf_loss = (td_loss * batch["weights"] * tail_excess).sum() / denom
            vf_loss = vf_loss + self.lambda_tail_critic * tail_vf_loss

        ag.critic1_optimizer.zero_grad(set_to_none=True)
        ag.critic2_optimizer.zero_grad(set_to_none=True)
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(ag.critic2.parameters(), 0.5)
        ag.critic1_optimizer.step()
        ag.critic2_optimizer.step()

        ag.policy_optimizer.zero_grad(set_to_none=True)
        curr_pol_out, curr_pol_act = self._build_policy_action(agent_i, batch["obs"][agent_i], explore_gumbel=True)

        all_pol_acs = []
        for idx, (pi, ob) in enumerate(zip(self.policies, batch["obs"])):
            if idx == agent_i:
                all_pol_acs.append(curr_pol_act)
            else:
                with torch.no_grad():
                    if self.discrete_action:
                        all_pol_acs.append(onehot_from_logits(pi(ob)))
                    else:
                        all_pol_acs.append(torch.clamp(pi(ob), -1.0, 1.0))

        vf_in_pol = torch.cat((*batch["obs"], *all_pol_acs), dim=1)
        q_pol = ag.critic1(vf_in_pol)
        if self.use_tail_weighted_training and self.use_tail_actor_loss and self.lambda_tail_actor > 0.0:
            actor_weights = 1.0 + self.lambda_tail_actor * (batch["tail_weights"] - 1.0).clamp_min(0.0)
            actor_weights = actor_weights / actor_weights.mean().clamp_min(1e-6)
            pol_loss = -(q_pol * actor_weights).mean()
        else:
            pol_loss = -q_pol.mean()

        # In residual-prior control, the actor output is a correction term rather
        # than the full acceleration command. Penalizing large residuals keeps the
        # learned policy from fighting the waypoint prior unless the critic supports it.
        if self.residual_action_reg > 0.0:
            pol_loss = pol_loss + (curr_pol_out ** 2).mean() * self.residual_action_reg

        if self.discrete_action:
            probs = F.softmax(curr_pol_out, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            pol_loss = pol_loss - 2e-3 * entropy

        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.policy.parameters(), 0.2)
        ag.policy_optimizer.step()

        if logger is not None:
            logger.add_scalars(
                f"agent{agent_i}/loss",
                {"critic": float(vf_loss.item()), "actor": float(pol_loss.item())},
                self.niter,
            )

        return float(vf_loss.item()), float(pol_loss.item()), td_error

    def update_rscu(
        self,
        sample,
        agent_i,
        lambda_rec=0.05,
        lambda_safe=0.02,
        lambda_smooth=0.001,
        rscu_actor_scope="executor",
        executor_idx=3,
        update_actor=True,
        rscu_postfound_only=True,
        rscu_mask_mode="post_found",
        rscu_min_mask_count=4,
        rscu_aux_critic_masked=True,
    ):
        batch = self._prepare_sample(sample)
        ag = self.agents[agent_i]
        batch_size = int(batch["weights"].shape[0])
        ones_mask = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        post_found_masks = batch["post_found_masks"].view(-1, 1)
        near_target_masks = batch["near_target_masks"].view(-1, 1)
        mask_mode = str(rscu_mask_mode).strip().lower()
        if not bool(rscu_postfound_only):
            mask_mode = "all"
        if mask_mode == "post_found":
            aux_mask = post_found_masks
        elif mask_mode == "near_target":
            aux_mask = near_target_masks
        elif mask_mode == "post_found_or_near":
            aux_mask = torch.clamp(post_found_masks + near_target_masks, 0.0, 1.0)
        elif mask_mode == "all":
            aux_mask = ones_mask
        else:
            aux_mask = post_found_masks
            mask_mode = "post_found"
        aux_mask = torch.nan_to_num(aux_mask, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        aux_mask_count = int((aux_mask > 0.5).sum().item())
        min_mask_count = int(max(0, rscu_min_mask_count))
        use_aux_penalty = aux_mask_count >= min_mask_count

        def weighted_mean(values, mask=None):
            values = values.view(batch_size, -1)
            if values.shape[1] != 1:
                values = values.mean(dim=1, keepdim=True)
            weights = batch["weights"]
            if mask is None:
                denom = weights.sum().clamp_min(1e-6)
                return (values * weights).sum() / denom
            mask = mask.view(batch_size, 1)
            denom = (weights * mask).sum().clamp_min(1e-6)
            return (values * weights * mask).sum() / denom

        target_q = self._compute_target_q(ag, batch["target_vf_in"], batch["rews"][agent_i], batch["dones"][agent_i])
        vf_in = self._critic_input(batch, agent_i)

        q1 = ag.critic1(vf_in)
        q2 = ag.critic2(vf_in)
        td_error = (target_q - q1).detach().squeeze(-1)
        task_loss1 = F.smooth_l1_loss(q1, target_q, reduction="none")
        task_loss2 = F.smooth_l1_loss(q2, target_q, reduction="none")
        task_td_loss = task_loss1 + task_loss2
        task_vf_loss = (task_td_loss * batch["weights"]).mean()
        if self.use_tail_weighted_training and self.use_tail_critic_loss and self.lambda_tail_critic > 0.0:
            tail_excess = (batch["tail_weights"] - 1.0).clamp_min(0.0)
            denom = tail_excess.sum().clamp_min(1e-6)
            tail_vf_loss = (task_td_loss * batch["weights"] * tail_excess).sum() / denom
            task_vf_loss = task_vf_loss + self.lambda_tail_critic * tail_vf_loss

        ag.critic1_optimizer.zero_grad(set_to_none=True)
        ag.critic2_optimizer.zero_grad(set_to_none=True)
        task_vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(ag.critic2.parameters(), 0.5)
        ag.critic1_optimizer.step()
        ag.critic2_optimizer.step()

        done_i = batch["dones"][agent_i].view(-1, 1)
        with torch.no_grad():
            target_rec = batch["recovery_costs"] + self.gamma * ag.target_rec_critic(batch["target_vf_in"]) * (1 - done_i)
            target_safe = batch["safety_costs"] + self.gamma * ag.target_safe_critic(batch["target_vf_in"]) * (1 - done_i)
        rec_q = ag.rec_critic(vf_in.detach())
        safe_q = ag.safe_critic(vf_in.detach())
        critic_mask = aux_mask if bool(rscu_aux_critic_masked) else ones_mask
        critic_mask_count = aux_mask_count if bool(rscu_aux_critic_masked) else batch_size
        rec_critic_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        safe_critic_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if critic_mask_count >= min_mask_count:
            rec_critic_loss = weighted_mean(F.smooth_l1_loss(rec_q, target_rec, reduction="none"), critic_mask)
            safe_critic_loss = weighted_mean(F.smooth_l1_loss(safe_q, target_safe, reduction="none"), critic_mask)

            ag.rec_critic_optimizer.zero_grad(set_to_none=True)
            rec_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(ag.rec_critic.parameters(), 0.5)
            ag.rec_critic_optimizer.step()

            ag.safe_critic_optimizer.zero_grad(set_to_none=True)
            safe_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(ag.safe_critic.parameters(), 0.5)
            ag.safe_critic_optimizer.step()

        actor_loss = None
        actor_task_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        actor_rec_penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        actor_safe_penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        actor_smooth_penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        rscu_actor_applied = False
        actor_allowed = bool(update_actor)
        if str(rscu_actor_scope).lower() == "executor":
            actor_allowed = actor_allowed and int(agent_i) == int(executor_idx)
        elif str(rscu_actor_scope).lower() != "all":
            actor_allowed = False

        if actor_allowed:
            ag.policy_optimizer.zero_grad(set_to_none=True)
            curr_pol_out, curr_pol_act = self._build_policy_action(agent_i, batch["obs"][agent_i], explore_gumbel=True)
            all_pol_acs = []
            for idx, (pi, ob) in enumerate(zip(self.policies, batch["obs"])):
                if idx == agent_i:
                    all_pol_acs.append(curr_pol_act)
                else:
                    with torch.no_grad():
                        if self.discrete_action:
                            all_pol_acs.append(onehot_from_logits(pi(ob)))
                        else:
                            all_pol_acs.append(torch.clamp(pi(ob), -1.0, 1.0))

            vf_in_pol = torch.cat((*batch["obs"], *all_pol_acs), dim=1)
            q_pol = ag.critic1(vf_in_pol)
            actor_task_loss = -(batch["weights"] * q_pol).mean()
            actor_loss = actor_task_loss
            if use_aux_penalty:
                q_rec_pi = ag.rec_critic(vf_in_pol)
                q_safe_pi = ag.safe_critic(vf_in_pol)
                actor_rec_penalty = weighted_mean(q_rec_pi, aux_mask)
                actor_safe_penalty = weighted_mean(q_safe_pi, aux_mask)
                action_norm = curr_pol_act.pow(2).mean(dim=1, keepdim=True)
                actor_smooth_penalty = weighted_mean(action_norm, aux_mask)
                actor_loss = (
                    actor_loss
                    + float(lambda_rec) * actor_rec_penalty
                    + float(lambda_safe) * actor_safe_penalty
                    + float(lambda_smooth) * actor_smooth_penalty
                )
                rscu_actor_applied = True
            if self.residual_action_reg > 0.0:
                actor_loss = actor_loss + (curr_pol_out ** 2).mean() * self.residual_action_reg
            if self.discrete_action:
                probs = F.softmax(curr_pol_out, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                actor_loss = actor_loss - 2e-3 * entropy
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(ag.policy.parameters(), 0.2)
            ag.policy_optimizer.step()

        critic_loss_total = task_vf_loss.detach() + rec_critic_loss.detach() + safe_critic_loss.detach()
        aux = {
            "rec_critic_loss": float(rec_critic_loss.detach().item()),
            "safe_critic_loss": float(safe_critic_loss.detach().item()),
            "actor_task_loss": float(actor_task_loss.detach().item()),
            "actor_rec_penalty": float(actor_rec_penalty.detach().item()),
            "actor_safe_penalty": float(actor_safe_penalty.detach().item()),
            "actor_smooth_penalty": float(actor_smooth_penalty.detach().item()),
            "mask_available": bool(batch.get("mask_available", False)),
            "aux_mask_frac": float(aux_mask.mean().detach().item()) if aux_mask.numel() else 0.0,
            "aux_mask_count": int(aux_mask_count),
            "post_found_frac": float(post_found_masks.mean().detach().item()) if post_found_masks.numel() else 0.0,
            "near_target_frac": float(near_target_masks.mean().detach().item()) if near_target_masks.numel() else 0.0,
            "rscu_actor_applied": bool(rscu_actor_applied),
            "rscu_mask_mode": mask_mode,
            "smooth_is_action_norm": True,
        }
        return critic_loss_total, None if actor_loss is None else actor_loss.detach(), td_error.detach(), aux

    def update_all_targets(self, compute_diff=False):
        diffs = {"policy": [], "critic1": [], "critic2": [], "rec_critic": [], "safe_critic": []} if compute_diff else None
        for a in self.agents:
            diff_c1 = soft_update(a.target_critic1, a.critic1, self.tau, return_diff=compute_diff)
            diff_c2 = soft_update(a.target_critic2, a.critic2, self.tau, return_diff=compute_diff)
            diff_rec = soft_update(a.target_rec_critic, a.rec_critic, self.tau, return_diff=compute_diff)
            diff_safe = soft_update(a.target_safe_critic, a.safe_critic, self.tau, return_diff=compute_diff)
            diff_pol = soft_update(a.target_policy, a.policy, self.tau, return_diff=compute_diff)
            if compute_diff:
                diffs["critic1"].append(diff_c1)
                diffs["critic2"].append(diff_c2)
                diffs["rec_critic"].append(diff_rec)
                diffs["safe_critic"].append(diff_safe)
                diffs["policy"].append(diff_pol)

        self._clear_sample_cache()
        self.niter += 1
        return diffs

    def scale_noise(self, factor: float, multiply: bool = True):
        for agent in self.agents:
            agent.scale_noise(factor, multiply=multiply)

    def prep_training(self, device="cuda"):
        target_device = self._resolve_device(device)
        self._move_all_to(target_device)
        for a in self.agents:
            a.policy.train()
            a.critic1.train()
            a.critic2.train()
            a.rec_critic.train()
            a.safe_critic.train()
            a.target_policy.train()
            a.target_critic1.train()
            a.target_critic2.train()
            a.target_rec_critic.train()
            a.target_safe_critic.train()
            if hasattr(a, "sync_noise_device"):
                a.sync_noise_device()

    def prep_rollouts(self, device=None):
        if device is not None:
            target_device = self._resolve_device(device)
            self._move_all_to(target_device)
        for a in self.agents:
            a.policy.eval()
            a.critic1.eval()
            a.critic2.eval()
            a.rec_critic.eval()
            a.safe_critic.eval()
            if hasattr(a, "sync_noise_device"):
                a.sync_noise_device()

    def save(self, filename):
        save_dict = {
            "init_dict": self.init_dict,
            "agent_params": [self._clone_agent_params_to_cpu(a) for a in self.agents],
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(
        cls,
        env,
        gamma=0.95,
        tau=0.01,
        lr=0.01,
        lr_actor=None,
        lr_critic=None,
        hidden_dim=64,
    ):
        agent_init_params = []
        alg_types = ["MADDPG" for _ in range(env.num_agents)]

        obs_dims = [env.observation_space[f"agent_{i}"].shape[0] for i in range(env.num_agents)]
        ac_dims = [env.action_space[f"agent_{i}"].shape[0] for i in range(env.num_agents)]
        total_critic_in = sum(obs_dims) + sum(ac_dims)

        for i in range(env.num_agents):
            num_in_pol = obs_dims[i]
            num_out_pol = ac_dims[i]
            num_in_critic = total_critic_in if alg_types[i] == "MADDPG" else (obs_dims[i] + ac_dims[i])
            params = {
                "num_in_pol": num_in_pol,
                "num_out_pol": num_out_pol,
                "num_in_critic": num_in_critic,
            }
            agent_init_params.append(params)

        role_names = getattr(env, "role_names", [f"agent_{i}" for i in range(env.num_agents)])
        if hasattr(env, "agent_specs") and len(env.agent_specs) == env.num_agents:
            noise_sigmas = []
            for spec in env.agent_specs:
                name = spec.get("name", "")
                if name == "search_fast":
                    noise_sigmas.append(0.18)
                elif name == "search_balanced":
                    noise_sigmas.append(0.14)
                elif name == "search_precise":
                    noise_sigmas.append(0.10)
                elif name == "executor":
                    noise_sigmas.append(0.08)
                else:
                    noise_sigmas.append(0.12)
        else:
            noise_sigmas = None

        effective_lr_actor = lr if lr_actor is None else lr_actor
        effective_lr_critic = lr if lr_critic is None else lr_critic

        init_dict = {
            "gamma": gamma,
            "tau": tau,
            "lr": lr,
            "lr_actor": effective_lr_actor,
            "lr_critic": effective_lr_critic,
            "hidden_dim": hidden_dim,
            "alg_types": alg_types,
            "agent_init_params": agent_init_params,
            "discrete_action": False,
            "agent_role_names": role_names,
            "agent_noise_sigmas": noise_sigmas,
            "residual_action_reg": 1e-2,
        }

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, device=None):
        load_device = cls._resolve_device(device)
        save_dict = torch.load(filename, map_location=load_device)
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)
        instance.prep_training(device=load_device)
        return instance
