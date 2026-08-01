import numpy as np
import torch

from registry.rbe_disturbance import DISTURBANCE_KEYS, nominal_disturbance


class EpisodeMetricTracker:
    def __init__(self, stable_window=5, stable_nav_dist=2.0, stable_speed=0.35):
        self.stable_window = int(max(1, stable_window))
        self.stable_nav_dist = float(stable_nav_dist)
        self.stable_speed = float(stable_speed)
        self.reset(None, nominal_disturbance())

    def reset(self, env, xi):
        del env
        base = nominal_disturbance()
        base.update(dict(xi or {}))
        self.xi = {key: base[key] for key in DISTURBANCE_KEYS}
        self.reward_sums = []
        self.reward_means = []
        self.safety_costs = []
        self.recovery_costs = []
        self.actions = []
        self.prev_action = None
        self.smooth_costs = []
        self.stable_count = 0
        self.recovery_time = None

    @staticmethod
    def _as_tensor(value, dtype=torch.float32):
        if torch.is_tensor(value):
            return value.detach().to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    @staticmethod
    def _agent_pos(env):
        if hasattr(env, "_agent_pos"):
            return env._agent_pos.detach()
        return EpisodeMetricTracker._as_tensor(env.agent_pos)

    @staticmethod
    def _agent_vel(env):
        if hasattr(env, "_agent_vel"):
            return env._agent_vel.detach()
        return EpisodeMetricTracker._as_tensor(env.agent_vel)

    @staticmethod
    def _task_target(env):
        if hasattr(env, "_task_target"):
            return env._task_target.detach()
        return EpisodeMetricTracker._as_tensor(env.task_target)

    @staticmethod
    def _nav_targets(env):
        if hasattr(env, "_nav_targets"):
            return env._nav_targets.detach()
        return EpisodeMetricTracker._as_tensor(env.nav_targets)

    @staticmethod
    def _collision_flags(env):
        if hasattr(env, "_collision_flags"):
            return env._collision_flags.detach()
        return EpisodeMetricTracker._as_tensor(env.collision_flags, dtype=torch.bool)

    def _executor_nav_distance(self, env):
        pos = self._agent_pos(env)
        nav_targets = self._nav_targets(env)
        exec_i = int(getattr(env, "executor_idx", pos.shape[0] - 1))
        return float(torch.norm(pos[exec_i] - nav_targets[exec_i]).item())

    def _executor_task_distance(self, env):
        pos = self._agent_pos(env)
        target = self._task_target(env)
        exec_i = int(getattr(env, "executor_idx", pos.shape[0] - 1))
        return float(torch.norm(pos[exec_i] - target).item())

    def _executor_speed(self, env):
        vel = self._agent_vel(env)
        exec_i = int(getattr(env, "executor_idx", vel.shape[0] - 1))
        return float(torch.norm(vel[exec_i]).item())

    def _safety_cost(self, env):
        pos = self._agent_pos(env)
        safe_dist = float(getattr(env, "safe_dist", 0.0))
        cost = 0.0
        if pos.ndim == 2 and pos.shape[0] > 1:
            pairwise = torch.cdist(pos, pos)
            n = int(pos.shape[0])
            for i in range(n):
                for j in range(i + 1, n):
                    cost += float(torch.clamp(safe_dist - pairwise[i, j], min=0.0).item())
        flags = self._collision_flags(env)
        if flags.numel() > 0:
            cost += float(flags.to(dtype=torch.float32).sum().item())
        return float(cost)

    def step(self, env, actions, rewards=None, dones=None):
        del dones
        safety_cost_t = self._safety_cost(env)
        executor_nav_distance = self._executor_nav_distance(env)
        recovery_cost_t = executor_nav_distance / 10.0 + safety_cost_t

        action_t = self._as_tensor(actions).detach().cpu()
        if self.prev_action is None:
            smooth_cost_t = 0.0
        else:
            diff = action_t - self.prev_action
            smooth_cost_t = float(torch.sum(diff * diff, dim=-1).mean().item())
        self.prev_action = action_t.clone()
        self.actions.append(action_t)
        self.smooth_costs.append(smooth_cost_t)
        self.safety_costs.append(safety_cost_t)
        self.recovery_costs.append(recovery_cost_t)

        if rewards is not None:
            rewards_t = self._as_tensor(rewards).detach().cpu().reshape(-1)
            self.reward_sums.append(float(rewards_t.sum().item()))
            self.reward_means.append(float(rewards_t.mean().item()))

        if (
            executor_nav_distance < self.stable_nav_dist
            and self._executor_speed(env) < self.stable_speed
            and safety_cost_t == 0.0
        ):
            self.stable_count += 1
        else:
            self.stable_count = 0
        if self.recovery_time is None and self.stable_count >= self.stable_window:
            self.recovery_time = int(getattr(env, "step_count", len(self.safety_costs)))

        return {
            "recovery_cost_t": float(recovery_cost_t),
            "safety_cost_t": float(safety_cost_t),
            "smooth_cost_t": float(smooth_cost_t),
        }

    def finalize(self, env):
        metrics = {key: self.xi[key] for key in DISTURBANCE_KEYS}
        steps = int(getattr(env, "step_count", len(self.safety_costs)))
        recovery_time = self.recovery_time
        if recovery_time is None:
            recovery_time = int(getattr(env, "max_steps", steps))
        metrics.update(
            {
                "success_flag": bool(getattr(env, "mission_complete", False)),
                "found_flag": bool(getattr(env, "task_found", False)),
                "episode_reward_mean": float(np.mean(self.reward_means)) if self.reward_means else 0.0,
                "episode_reward_sum": float(np.sum(self.reward_sums)) if self.reward_sums else 0.0,
                "completion_steps": steps,
                "recovery_time": int(recovery_time),
                "safety_cost": float(np.sum(self.safety_costs)) if self.safety_costs else 0.0,
                "safety_cost_mean": float(np.mean(self.safety_costs)) if self.safety_costs else 0.0,
                "final_distance": float(self._executor_task_distance(env)),
                "final_nav_distance": float(self._executor_nav_distance(env)),
                "action_smoothness": float(np.mean(self.smooth_costs)) if self.smooth_costs else 0.0,
            }
        )
        return metrics
