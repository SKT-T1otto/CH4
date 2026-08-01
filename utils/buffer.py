import torch


class ReplayBuffer:
    """
    Device-aware PER replay buffer used by PSE/RBE training.

    Stored transition fields:
    - per-agent obs/action/reward/next_obs/done
    - success flag for task-relevant samples
    - optional tail_score / scenario_id fields for prioritized replay
    - global scalar recovery/safety/smoothness costs shared by all agents
    """

    def __init__(
        self,
        max_steps: int,
        num_agents: int,
        obs_dims,
        ac_dims,
        success_priority: float = 2.0,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        storage_device=None,
        pin_memory: bool = False,
        tail_priority_scale: float = 1.5,
        graph_meta_dim: int = 16,
        message_meta_dim: int = 16,
        belief_meta_dim: int = 16,
    ):
        self.max_steps = int(max_steps)
        self.num_agents = int(num_agents)
        self.obs_dims = obs_dims
        self.ac_dims = ac_dims
        self.success_priority = float(success_priority)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.tail_priority_scale = float(max(0.0, tail_priority_scale))
        self.graph_meta_dim = int(graph_meta_dim)
        self.message_meta_dim = int(message_meta_dim)
        self.belief_meta_dim = int(belief_meta_dim)
        self.frame = 1.0
        self.filled_i = 0
        self.next_idx = 0
        self.storage_device = torch.device(storage_device) if storage_device is not None else torch.device("cpu")
        self.pin_memory = bool(pin_memory and self.storage_device.type == "cpu")

        def alloc(shape, dtype=torch.float32):
            if self.storage_device.type == "cpu":
                return torch.empty(shape, dtype=dtype, pin_memory=self.pin_memory)
            return torch.empty(shape, dtype=dtype, device=self.storage_device)

        self.obs_buffs = [alloc((self.max_steps, d)) for d in obs_dims]
        self.ac_buffs = [alloc((self.max_steps, d)) for d in ac_dims]
        self.rew_buffs = [alloc((self.max_steps,)) for _ in range(self.num_agents)]
        self.next_obs_buffs = [alloc((self.max_steps, d)) for d in obs_dims]
        self.done_buffs = [alloc((self.max_steps,)) for _ in range(self.num_agents)]
        self.success_buffs = alloc((self.max_steps,), dtype=torch.bool)
        self.success_buffs.zero_()
        self.tail_score_buffs = alloc((self.max_steps,))
        self.tail_score_buffs.zero_()
        self.comm_risk_buffs = alloc((self.max_steps,))
        self.comm_risk_buffs.zero_()
        self.scenario_id_buffs = alloc((self.max_steps,), dtype=torch.long)
        self.scenario_id_buffs.zero_()
        self.graph_meta_buffs = alloc((self.max_steps, self.graph_meta_dim))
        self.graph_meta_buffs.zero_()
        self.message_meta_buffs = alloc((self.max_steps, self.message_meta_dim))
        self.message_meta_buffs.zero_()
        self.belief_meta_buffs = alloc((self.max_steps, self.belief_meta_dim))
        self.belief_meta_buffs.zero_()
        self.task_found_buffs = alloc((self.max_steps,))
        self.task_found_buffs.zero_()
        self.executor_assigned_buffs = alloc((self.max_steps,))
        self.executor_assigned_buffs.zero_()
        self.post_found_buffs = alloc((self.max_steps,))
        self.post_found_buffs.zero_()
        self.near_target_buffs = alloc((self.max_steps,))
        self.near_target_buffs.zero_()
        self.executor_nav_distance_buffs = alloc((self.max_steps,))
        self.executor_nav_distance_buffs.zero_()
        self.recovery_cost_buffs = alloc((self.max_steps,))
        self.recovery_cost_buffs.zero_()
        self.safety_cost_buffs = alloc((self.max_steps,))
        self.safety_cost_buffs.zero_()
        self.smooth_cost_buffs = alloc((self.max_steps,))
        self.smooth_cost_buffs.zero_()
        self.priorities = alloc((self.max_steps,))
        self.priorities.fill_(1.0)

    def _to_tensor(self, x, *, dtype=torch.float32, device=None, flatten=False):
        target_device = self.storage_device if device is None else torch.device(device)
        if torch.is_tensor(x):
            t = x.detach().to(device=target_device, dtype=dtype, non_blocking=True)
        else:
            t = torch.as_tensor(x, dtype=dtype, device=target_device)
        if flatten:
            t = t.reshape(-1)
        return t

    @staticmethod
    def _scalar_float(x, default=0.0):
        if x is None:
            return float(default)
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().float().mean().item())
        try:
            return float(x)
        except Exception:
            return float(default)

    def _meta_tensor(self, x, dim):
        out = torch.zeros((int(dim),), dtype=torch.float32, device=self.storage_device)
        if x is None:
            return out
        if isinstance(x, dict):
            values = []
            for key in sorted(x):
                value = x[key]
                try:
                    if torch.is_tensor(value):
                        values.append(float(value.detach().float().mean().item()))
                    else:
                        values.append(float(value))
                except Exception:
                    values.append(0.0)
            t = torch.as_tensor(values, dtype=torch.float32, device=self.storage_device).reshape(-1)
        elif torch.is_tensor(x):
            t = x.detach().to(device=self.storage_device, dtype=torch.float32, non_blocking=True).reshape(-1)
        else:
            t = torch.as_tensor(x, dtype=torch.float32, device=self.storage_device).reshape(-1)
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        n = min(int(dim), int(t.numel()))
        if n > 0:
            out[:n].copy_(t[:n])
        return out

    def push(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        success_flags,
        tail_score=None,
        tail_scores=None,
        comm_risk=None,
        comm_risks=None,
        scenario_id=None,
        scenario_ids=None,
        graph_meta=None,
        message_meta=None,
        belief_meta=None,
        recovery_cost=None,
        safety_cost=None,
        smooth_cost=None,
        task_found=None,
        executor_assigned=None,
        post_found=None,
        near_target=None,
        executor_nav_distance=None,
        **kwargs,
    ):
        del kwargs
        idx = self.next_idx
        is_success = bool(any(success_flags))
        self.success_buffs[idx] = is_success

        rewards_t = self._to_tensor(rewards, flatten=True)
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][idx].copy_(self._to_tensor(obs[agent_i], flatten=True))
            self.ac_buffs[agent_i][idx].copy_(self._to_tensor(actions[agent_i], flatten=True))
            self.rew_buffs[agent_i][idx] = rewards_t[agent_i]
            self.next_obs_buffs[agent_i][idx].copy_(self._to_tensor(next_obs[agent_i], flatten=True))
            self.done_buffs[agent_i][idx] = float(dones[agent_i])

        ts = self._scalar_float(tail_score if tail_score is not None else tail_scores, 0.0)
        cr = self._scalar_float(comm_risk if comm_risk is not None else comm_risks, ts)
        sid = int(self._scalar_float(scenario_id if scenario_id is not None else scenario_ids, 0.0))
        ts = float(max(0.0, min(1.0, ts)))
        cr = float(max(0.0, min(1.0, cr)))

        self.tail_score_buffs[idx] = ts
        self.comm_risk_buffs[idx] = cr
        self.scenario_id_buffs[idx] = sid
        self.graph_meta_buffs[idx].copy_(self._meta_tensor(graph_meta, self.graph_meta_dim))
        self.message_meta_buffs[idx].copy_(self._meta_tensor(message_meta, self.message_meta_dim))
        self.belief_meta_buffs[idx].copy_(self._meta_tensor(belief_meta, self.belief_meta_dim))
        self.task_found_buffs[idx] = 1.0 if bool(task_found) else 0.0
        self.executor_assigned_buffs[idx] = 1.0 if bool(executor_assigned) else 0.0
        self.post_found_buffs[idx] = 1.0 if bool(post_found) else 0.0
        self.near_target_buffs[idx] = 1.0 if bool(near_target) else 0.0
        self.executor_nav_distance_buffs[idx] = self._scalar_float(executor_nav_distance, 0.0)
        self.recovery_cost_buffs[idx] = self._scalar_float(recovery_cost, 0.0)
        self.safety_cost_buffs[idx] = self._scalar_float(safety_cost, 0.0)
        self.smooth_cost_buffs[idx] = self._scalar_float(smooth_cost, 0.0)

        priority = self.success_priority if is_success else 1.0
        priority *= (1.0 + self.tail_priority_scale * ts)
        self.priorities[idx] = float(priority)
        self.next_idx = (self.next_idx + 1) % self.max_steps
        self.filled_i = min(self.filled_i + 1, self.max_steps)

    def _beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def sample(self, N, to_gpu=False, norm_rews=True, device=None):
        if self.filled_i == 0:
            raise ValueError("ReplayBuffer 为空，无法采样")
        N = min(int(N), self.filled_i)

        target_device = self.storage_device if device is None else torch.device(device)
        probs = self.priorities[: self.filled_i].pow(self.alpha)
        prob_sum = probs.sum()
        if float(prob_sum.item()) <= 0:
            probs = torch.full((self.filled_i,), 1.0 / self.filled_i, dtype=torch.float32, device=self.storage_device)
        else:
            probs = probs / prob_sum

        inds = torch.multinomial(probs, num_samples=N, replacement=False)
        self.frame += 1.0 / max(1, self.num_agents)
        beta = self._beta_by_frame()
        weights = (self.filled_i * probs[inds]).pow(-beta)
        weights = weights / weights.max().clamp_min(1e-8)
        success_flags = self.success_buffs[inds]
        tail_scores = self.tail_score_buffs[inds]
        comm_risks = self.comm_risk_buffs[inds]
        scenario_ids = self.scenario_id_buffs[inds]
        graph_meta = self.graph_meta_buffs[inds]
        message_meta = self.message_meta_buffs[inds]
        belief_meta = self.belief_meta_buffs[inds]
        task_found_masks = self.task_found_buffs[inds]
        executor_assigned_masks = self.executor_assigned_buffs[inds]
        post_found_masks = self.post_found_buffs[inds]
        near_target_masks = self.near_target_buffs[inds]
        executor_nav_distances = self.executor_nav_distance_buffs[inds]
        recovery_costs = self.recovery_cost_buffs[inds]
        safety_costs = self.safety_cost_buffs[inds]
        smooth_costs = self.smooth_cost_buffs[inds]

        def maybe_move(x):
            if x.device == target_device:
                return x
            return x.to(device=target_device, non_blocking=True)

        if norm_rews:
            ret_rews = []
            for i in range(self.num_agents):
                rew = self.rew_buffs[i][: self.filled_i]
                mean = rew.mean()
                std = rew.std(unbiased=False).clamp_min(1e-6)
                ret_rews.append(maybe_move((self.rew_buffs[i][inds] - mean) / std))
        else:
            ret_rews = [maybe_move(self.rew_buffs[i][inds]) for i in range(self.num_agents)]

        return (
            [maybe_move(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
            [maybe_move(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
            ret_rews,
            [maybe_move(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
            [maybe_move(self.done_buffs[i][inds]) for i in range(self.num_agents)],
            maybe_move(weights),
            inds if inds.device == target_device else inds.to(device=target_device, non_blocking=True),
            success_flags if success_flags.device == target_device else success_flags.to(device=target_device, non_blocking=True),
            tail_scores if tail_scores.device == target_device else tail_scores.to(device=target_device, non_blocking=True),
            comm_risks if comm_risks.device == target_device else comm_risks.to(device=target_device, non_blocking=True),
            scenario_ids if scenario_ids.device == target_device else scenario_ids.to(device=target_device, non_blocking=True),
            graph_meta if graph_meta.device == target_device else graph_meta.to(device=target_device, non_blocking=True),
            message_meta if message_meta.device == target_device else message_meta.to(device=target_device, non_blocking=True),
            belief_meta if belief_meta.device == target_device else belief_meta.to(device=target_device, non_blocking=True),
            task_found_masks if task_found_masks.device == target_device else task_found_masks.to(device=target_device, non_blocking=True),
            executor_assigned_masks if executor_assigned_masks.device == target_device else executor_assigned_masks.to(device=target_device, non_blocking=True),
            post_found_masks if post_found_masks.device == target_device else post_found_masks.to(device=target_device, non_blocking=True),
            near_target_masks if near_target_masks.device == target_device else near_target_masks.to(device=target_device, non_blocking=True),
            executor_nav_distances if executor_nav_distances.device == target_device else executor_nav_distances.to(device=target_device, non_blocking=True),
            recovery_costs if recovery_costs.device == target_device else recovery_costs.to(device=target_device, non_blocking=True),
            safety_costs if safety_costs.device == target_device else safety_costs.to(device=target_device, non_blocking=True),
            smooth_costs if smooth_costs.device == target_device else smooth_costs.to(device=target_device, non_blocking=True),
        )

    def update_priorities(self, indices, td_errors, success_flags=None, tail_scores=None, eps=1e-5):
        if indices is None or td_errors is None or self.filled_i <= 0:
            return
        if torch.is_tensor(indices):
            idx = indices.detach().to(device=self.storage_device, dtype=torch.long, non_blocking=True).reshape(-1)
        else:
            idx = torch.as_tensor(indices, dtype=torch.long, device=self.storage_device).reshape(-1)
        if torch.is_tensor(td_errors):
            err = td_errors.detach().to(device=self.storage_device, dtype=torch.float32, non_blocking=True).reshape(-1)
        else:
            err = torch.as_tensor(td_errors, dtype=torch.float32, device=self.storage_device).reshape(-1)

        n = min(int(idx.numel()), int(err.numel()))
        if n <= 0:
            return
        idx = idx[:n].clamp(0, int(self.filled_i) - 1)
        err = torch.nan_to_num(err[:n].abs(), nan=0.0, posinf=100.0, neginf=0.0)
        priority = err + float(eps)

        if success_flags is None:
            success = torch.zeros((n,), dtype=torch.bool, device=self.storage_device)
        elif torch.is_tensor(success_flags):
            success = success_flags.detach().to(device=self.storage_device, dtype=torch.bool, non_blocking=True).reshape(-1)
        else:
            success = torch.as_tensor(success_flags, dtype=torch.bool, device=self.storage_device).reshape(-1)
        if success.numel() < n:
            pad = torch.zeros((n - int(success.numel()),), dtype=torch.bool, device=self.storage_device)
            success = torch.cat([success, pad], dim=0)
        success = success[:n]
        priority = priority * torch.where(success, torch.full_like(priority, self.success_priority), torch.ones_like(priority))

        if tail_scores is None:
            tail = torch.zeros((n,), dtype=torch.float32, device=self.storage_device)
        elif torch.is_tensor(tail_scores):
            tail = tail_scores.detach().to(device=self.storage_device, dtype=torch.float32, non_blocking=True).reshape(-1)
        else:
            tail = torch.as_tensor(tail_scores, dtype=torch.float32, device=self.storage_device).reshape(-1)
        if tail.numel() < n:
            pad = torch.zeros((n - int(tail.numel()),), dtype=torch.float32, device=self.storage_device)
            tail = torch.cat([tail, pad], dim=0)
        tail = torch.nan_to_num(tail[:n], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if self.tail_priority_scale > 0.0:
            priority = priority * (1.0 + self.tail_priority_scale * tail)

        priority = torch.nan_to_num(priority, nan=float(eps), posinf=100.0, neginf=float(eps))
        self.priorities[idx] = priority.clamp(min=float(eps), max=100.0)

    def __len__(self):
        return int(self.filled_i)
