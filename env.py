
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass

import numpy as np
import torch

from registry.rbe_disturbance import DISTURBANCE_KEYS, nominal_disturbance


def _missing_top_level_package(exc: ModuleNotFoundError, package_names) -> bool:
    """Return True only when a package-style import failed at its top level."""
    return getattr(exc, "name", None) in set(package_names)


try:
    from map.map_module import PheromoneWaypointPlanner, ProbabilisticTaskMapPlanner
except ModuleNotFoundError as exc:
    if not _missing_top_level_package(exc, {"map"}):
        raise
    from map.map_module import PheromoneWaypointPlanner, ProbabilisticTaskMapPlanner

@dataclass
class BasicCommConfig:
    n_agents: int
    space_z: float
    dt: float
    comm_reliable_range: float
    comm_max_range: float
    comm_base_loss: float
    comm_attenuation: float
    effective_sound_speed: float
    extra_delay_steps: int
    max_delay_steps: int
    payload_bits: int
    rate_bps: float
    msg_ttl_steps: int
    payload_loss_scale: float
    use_burst_comm: bool
    edge_delay_lambda: float = 1.20
    edge_age_lambda: float = 1.00
    edge_queue_lambda: float = 0.30
    device: object = None
    dtype: object = torch.float32


class BasicCommGraph:
    EDGE_FEATURE_NAMES = [
        "reachable", "success_prob", "loss_prob", "delay_norm",
        "msg_age_norm", "rate_norm", "bandwidth_norm", "queue_len_norm",
        "burst_state_norm", "distance_norm", "depth_gap_norm",
        "flow_diff_norm", "edge_weight",
    ]
    EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device) if config.device is not None else torch.device("cpu")
        self.dtype = config.dtype
        self.last_stats = {}

    def update_from_env(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def _payload_factor(self):
        cfg = self.config
        if cfg.payload_loss_scale <= 0.0:
            return 1.0
        bits = max(1.0, float(cfg.payload_bits))
        return float(torch.exp(torch.tensor(-cfg.payload_loss_scale * max(0.0, bits - 80.0) / 160.0)).item())

    def _queue_len_matrix(self, link_queues, shape, device):
        q = torch.zeros(shape, dtype=self.dtype, device=device)
        if link_queues is None:
            return q
        n = min(int(self.config.n_agents), len(link_queues))
        for sender in range(n):
            for receiver in range(min(int(self.config.n_agents), len(link_queues[sender]))):
                if sender != receiver:
                    q[receiver, sender] = float(len(link_queues[sender][receiver]))
        return q

    def build_lower_graph(
        self, *, agent_pos, step_count, delivered_steps, lower_link_state,
        burst_state_pdr, link_queues=None, flow=None, use_comm=True
    ):
        cfg = self.config
        pos = agent_pos.to(dtype=self.dtype)
        device = pos.device
        eps = 1e-6
        dists = torch.cdist(pos, pos)
        reachable = (dists <= float(cfg.comm_max_range)).to(self.dtype) if use_comm else torch.zeros_like(dists)
        reachable.fill_diagonal_(0.0)
        p_reliable = 1.0 - float(max(0.0, min(1.0, cfg.comm_base_loss)))
        success_prob = torch.zeros_like(dists, dtype=self.dtype, device=device)
        reliable = dists <= float(cfg.comm_reliable_range)
        fade = (dists > float(cfg.comm_reliable_range)) & (dists <= float(cfg.comm_max_range))
        success_prob = torch.where(reliable, torch.full_like(success_prob, p_reliable), success_prob)
        fade_prob = p_reliable * torch.exp(-float(cfg.comm_attenuation) * (dists - float(cfg.comm_reliable_range)))
        success_prob = torch.where(fade, fade_prob, success_prob)
        if cfg.use_burst_comm:
            burst_pdr = burst_state_pdr.to(device=device, dtype=self.dtype)[lower_link_state.to(device=device, dtype=torch.long)]
            success_prob = success_prob * burst_pdr
        success_prob = torch.clamp(success_prob * self._payload_factor() * reachable, 0.0, 1.0)
        success_prob.fill_diagonal_(0.0)
        loss_prob = 1.0 - success_prob
        bandwidth = torch.full_like(dists, float(cfg.rate_bps))
        prop_delay = dists / max(float(cfg.effective_sound_speed), eps)
        tx_delay = float(cfg.payload_bits) / torch.clamp(bandwidth, min=1.0)
        delay_steps = torch.ceil((prop_delay + tx_delay) / max(float(cfg.dt), eps)) + float(cfg.extra_delay_steps)
        delay_steps = torch.clamp(delay_steps, 0.0, float(cfg.max_delay_steps))
        delay_steps.fill_diagonal_(0.0)
        ages = (int(step_count) - delivered_steps.to(device=device, dtype=torch.int32)).to(dtype=self.dtype)
        age_norm = torch.clamp(ages / max(1.0, float(cfg.msg_ttl_steps)), 0.0, 1.0)
        age_norm.fill_diagonal_(0.0)
        queue_norm = torch.clamp(self._queue_len_matrix(link_queues, dists.shape, device) / 8.0, 0.0, 1.0)
        delay_norm = torch.clamp(delay_steps / max(1.0, float(cfg.max_delay_steps)), 0.0, 1.0)
        distance_norm = torch.clamp(dists / max(float(cfg.comm_max_range), eps), 0.0, 1.0)
        z = pos[:, 2]
        depth_gap = torch.abs(z.unsqueeze(0) - z.unsqueeze(1))
        depth_norm = torch.clamp(depth_gap / max(float(cfg.space_z), eps), 0.0, 1.0)
        if flow is None:
            flow_diff = torch.zeros_like(dists)
        else:
            flow_diff = torch.cdist(flow.to(device=device, dtype=self.dtype), flow.to(device=device, dtype=self.dtype))
        flow_norm = torch.clamp(flow_diff, 0.0, 1.0)
        rate_norm = torch.clamp(success_prob, 0.0, 1.0)
        bandwidth_norm = torch.ones_like(dists)
        burst_norm = lower_link_state.to(device=device, dtype=self.dtype) / 2.0
        edge_weight = success_prob * torch.exp(-float(cfg.edge_delay_lambda) * delay_norm) * torch.exp(-float(cfg.edge_age_lambda) * age_norm)
        edge_weight = edge_weight * torch.exp(-float(cfg.edge_queue_lambda) * queue_norm) * reachable
        edge_weight.fill_diagonal_(0.0)
        features = torch.stack([
            reachable, success_prob, loss_prob, delay_norm, age_norm,
            rate_norm, bandwidth_norm, queue_norm, burst_norm,
            distance_norm, depth_norm, flow_norm, edge_weight,
        ], dim=-1)
        non_diag = reachable > 0.5
        self.last_stats = {
            "edge_weight": float(edge_weight[non_diag].mean().item()) if torch.any(non_diag) else 0.0,
            "loss_prob": float(loss_prob[non_diag].mean().item()) if torch.any(non_diag) else 0.0,
            "snr_norm": float(rate_norm[non_diag].mean().item()) if torch.any(non_diag) else 0.0,
            "bandwidth": float(cfg.rate_bps),
            "comm_noise": 0.0,
            "depth_gap": float(depth_gap[non_diag].mean().item()) if torch.any(non_diag) else 0.0,
            "flow_diff": float(flow_diff[non_diag].mean().item()) if torch.any(non_diag) else 0.0,
        }
        return {
            "reachable": reachable,
            "success_prob": success_prob,
            "delay_steps": delay_steps,
            "edge_features": features,
            "edge_weights": edge_weight,
        }


def decode_msg_type_name(msg):
    return "HEARTBEAT"


class UAVEnv:
    def __init__(
        self,
        n_agents=4,
        space_size=(20, 20, 8),
        dt=0.2,
        max_steps=400,
        random_z_range=(0.5, 7.5),
        random_search_waypoint_count=(2, 4),
        random_executor_waypoint_count=(1, 1),
        use_obstacles=False,
        planner_grid_size=(10, 10, 8),
        planner_visit_radius=1,
        planner_pheromone_decay=0.990,
        planner_suppression=0.54,
        planner_min_waypoint_separation=4.8,
        planner_step_update_interval=2,
        planner_step_update_suppress_only=False,
        hold_steps=5,
        search_hold_steps=0,
        executor_hold_steps=None,
        hold_speed_thresh=0.20,
        device=None,
        return_numpy=True,
        use_comm=True,
        comm_range=8.0,
        comm_loss_prob=0.10,
        comm_delay_steps=1,
        comm_reliable_range=None,
        comm_max_range=None,
        comm_attenuation=0.18,
        lower_effective_sound_speed=12.0,
        lower_extra_delay_steps=None,
        lower_max_delay_steps=16,
        lower_payload_bits=160,
        lower_rate_bps=1000.0,
        lower_msg_ttl_steps=24,
        lower_initial_sync_mode="none",
        use_burst_comm=True,
        burst_transition_matrix=None,
        burst_state_pdr=(1.0, 0.45, 0.03),
        burst_initial_state=0,
        payload_loss_scale=0.025,
        use_comm_energy=True,
        comm_idle_power=0.02,
        comm_rx_power=0.26,
        comm_tx_power=5.0,
        lambda_comm_energy=0.004,
        use_upper_comm=True,
        upper_node_pos=None,
        upper_comm_reliable_range=7.0,
        upper_comm_max_range=18.0,
        upper_comm_base_loss=0.05,
        upper_comm_attenuation=0.18,
        upper_effective_sound_speed=12.0,
        upper_extra_delay_steps=0,
        upper_map_update_interval=None,
        upper_wait_timeout_steps=8,
        upper_map_payload_bits=640,
        upper_state_payload_bits=160,
        upper_waypoint_payload_bits=120,
        upper_task_payload_bits=160,
        upper_rate_bps=1000.0,
        use_upper_belief=True,
        upper_belief_pos_noise_std=0.05,
        diverse_fallback_prob=0.25,
        diverse_fallback_tries=64,
        search_spread_reward_gain=0.75,
        detect_proximity_reward_gain=0.8,
        detect_proximity_radius=4.5,
        planner_coverage_weight=1.10,
        planner_claim_weight=0.80,
        planner_stochastic_topk=14,
        planner_stochastic_eps=0.26,
        use_residual_prior=True,
        prior_kv_xy=1.10,
        prior_kv_z=1.00,
        prior_slow_radius_xy=2.40,
        prior_slow_radius_z=1.20,
        prior_strength_search=1.00,
        prior_strength_executor=1.00,
        residual_scale_search=0.55,
        residual_scale_executor=0.35,
        residual_action_mode="cartesian",
        residual_struct_base_gate=1.0,
        residual_struct_low_risk_gate=0.70,
        residual_struct_high_risk_gate=1.00,
        residual_struct_lateral_scale_search=0.35,
        residual_struct_lateral_scale_exec=0.50,
        residual_struct_antitarget_scale=0.20,
        residual_struct_near_target_gate=0.45,
        residual_struct_near_radius_search=0.9,
        residual_struct_near_radius_exec=1.0,
        residual_struct_unknown_target_gate_exec=0.35,
        residual_hybrid_exec_cartesian_blend_known=0.85,
        residual_hybrid_exec_cartesian_blend_found=0.45,
        residual_hybrid_exec_cartesian_blend_unknown=0.00,
        residual_hybrid_exec_struct_blend_min=0.15,
        residual_hybrid_target_known_soft_gate=1.00,
        residual_hybrid_target_found_soft_gate=0.35,
        residual_hybrid_target_unknown_soft_gate=0.10,
        residual_hybrid_use_soft_gate=True,
        # Robust disturbance settings. Disabled by default so the standard
        # environment remains unchanged unless train/evaluate enables them.
        use_robust_disturbance=False,
        flow_gain_range=(0.18, 0.18),
        flow_z_gain_range=(0.0, 0.0),
        flow_phase_random=True,
        a_max_scale_range=(1.0, 1.0),
        v_max_scale_range=(1.0, 1.0),
        drag_scale_range=(1.0, 1.0),
        buoyancy_bias_delta_range=(0.0, 0.0),
        actuator_lag=0.0,
        robust_action_delay_steps=0,
        action_noise_std=0.0,
        action_noise_std_range=None,
        residual_penalty=0.0,
        reward_profile="original",
        exec_task_progress_gain_v2=18.0,
        near_target_speed_penalty_gain_v2=1.0,
        near_target_speed_radius_v2=2.0,
        hold_dense_reward_gain_v2=4.0,
        target_assigned_bonus_exec_v2=35.0,
        target_assigned_bonus_search_v2=10.0,
        handoff_delay_penalty_gain_v2=0.05,
        belief_age_penalty_gain_v2=0.02,
        belief_age_penalty_norm_v2=100.0,
        low_risk_residual_penalty_v2=0.03,
        high_risk_residual_penalty_v2=0.005,
        anti_prior_penalty_gain_v2=0.01,
        near_target_residual_penalty_gain_v2=0.02,
        near_target_residual_penalty_radius_v2=2.0,
        point_vel_align_gain_search_v3=0.20,
        point_vel_align_gain_exec_v3=0.45,
        point_lateral_vel_penalty_search_v3=0.08,
        point_lateral_vel_penalty_exec_v3=0.08,
        point_speed_profile_penalty_search_v3=0.06,
        point_speed_profile_penalty_exec_v3=0.05,
        point_residual_align_gain_search_v3=0.03,
        point_residual_align_gain_exec_v3=0.10,
        point_residual_antitarget_penalty_search_v3=0.02,
        point_residual_antitarget_penalty_exec_v3=0.04,
        point_slow_radius_search_v3=2.5,
        point_slow_radius_exec_v3=2.0,
        point_near_radius_search_v3=0.9,
        point_near_radius_exec_v3=1.0,
        point_shaping_total_clip_v3=6.0,
        # Basic distance/probability communication graph for Chapter 3/4
        use_dynamic_comm_graph=False,
        use_depth_comm_loss=False,
        use_flow_comm_loss=False,
        use_noise_comm_loss=False,
        use_snr_comm_model=False,
        comm_snr_ref_db=25.0,
        comm_snr_threshold_db=8.0,
        comm_snr_temperature=4.0,
        comm_depth_loss_coeff=0.35,
        comm_flow_loss_coeff=1.20,
        comm_noise_loss_coeff=1.00,
        comm_noise_base=0.10,
        comm_noise_spike_prob=0.02,
        comm_noise_spike_scale=0.50,
        comm_min_rate_bps=200.0,
        comm_max_rate_bps=5000.0,
        edge_delay_lambda=1.20,
        edge_age_lambda=1.00,
        edge_queue_lambda=0.30,
        # Removed robust-communication switches kept only for config compatibility
        use_semantic_messages=False,
        use_voi_selector=False,
        use_critical_priority=True,
        force_heartbeat_only=False,
        comm_budget_bits_per_step=640,
        lower_message_topk=1,
        critical_reserve_slots=1,
        use_adaptive_voi=True,
        voi_stress_loss_gain=0.65,
        voi_stress_delay_gain=0.35,
        voi_max_penalty_suppression=0.75,
        voi_priority_fallback_gain=1.00,
        voi_critical_fallback_gain=2.00,
        voi_receiver_need_fallback_gain=0.80,
        voi_type_diversity=True,
        voi_diversity_stress_threshold=0.65,
        voi_max_same_type_per_topk=1,
        voi_stress_critical_bonus_gain=1.25,
        voi_critical_bypass_diversity=True,
        use_stage_aware_voi=False,
        disable_handover_diversity_gate=True,
        disable_search_diversity_gate=False,
        voi_handover_window_steps=32,
        voi_search_diversity_pressure_threshold=0.70,
        voi_upper_success_bad_threshold=0.20,
        voi_island_diversity_threshold=0.20,
        voi_execute_diversity_pressure_threshold=0.78,
        use_multi_packet_semantic_comm=False,
        multi_packet_aggregate_mode="voi_reliability",
        multi_packet_max_per_edge=None,
        # Removed role-topology/direct-handover switches kept only for config compatibility
        use_role_topology=False,
        role_topology_mode="mission_chain",
        use_direct_handover_lane=False,
        role_topology_allow_executor_status=False,
        role_topology_allow_searcher_executor_pre_found=False,
        role_topology_strict_message_filter=True,
        direct_handover_bypass_message_filter=False,
        use_adaptive_handover_bypass=False,
        adaptive_handover_bypass_min_handoff_age_steps=6,
        adaptive_handover_bypass_max_handoff_age_steps=10**9,
        adaptive_handover_bypass_comm_pressure_threshold=0.50,
        adaptive_handover_bypass_island_threshold=0.20,
        adaptive_handover_bypass_upper_success_bad_threshold=0.35,
        adaptive_handover_bypass_require_executor_unknown=True,
        use_adaptive_critical_semantic_filter=False,
        use_guarded_adaptive_critical_filter=False,
        adaptive_critical_filter_default_soft=False,
        adaptive_critical_filter_preserve_ordinary_strict=True,
        adaptive_critical_filter_msg_types=("TARGET_FOUND", "TARGET_RELOCATE", "HANDOVER", "RISK_ALERT"),
        adaptive_critical_filter_comm_pressure_threshold=0.50,
        adaptive_critical_filter_island_threshold=0.20,
        adaptive_critical_filter_upper_success_bad_threshold=0.35,
        adaptive_critical_filter_min_handoff_age_steps=6,
        adaptive_critical_filter_require_executor_unknown=True,
        adaptive_ordinary_message_penalty=0.45,
        adaptive_noncritical_cross_role_penalty=0.35,
        adaptive_critical_message_bonus=1.35,
        adaptive_reconnect_voi_boost=False,
        reconnect_voi_boost_gain=1.25,
        reconnect_voi_boost_island_threshold=0.20,
        guarded_adaptive_comm_pressure_min=0.35,
        guarded_adaptive_comm_pressure_max=0.70,
        guarded_adaptive_island_min=0.12,
        guarded_adaptive_island_max=0.50,
        guarded_adaptive_upper_success_min=0.15,
        guarded_adaptive_critical_drop_max=0.68,
        guarded_extreme_comm_pressure_threshold=0.78,
        guarded_extreme_island_threshold=0.55,
        guarded_extreme_upper_success_threshold=0.12,
        guarded_extreme_critical_drop_threshold=0.70,
        guarded_handover_bypass_cooldown_steps=12,
        guarded_handover_bypass_max_per_window=1,
        guarded_preserve_context_slot=False,
        guarded_max_adaptive_critical_per_topk=1,
        guarded_allow_double_critical_when_handoff_pending=True,
        direct_handover_max_age_steps=24,
        use_reconnect_lane=False,
        reconnect_island_timeout_steps=None,
        # Removed teammate-prediction switches kept only for config compatibility
        use_teammate_prediction=False,
        use_prediction_guided_fallback=False,
        teammate_pred_max_age_steps=48,
        teammate_pred_uncertainty_growth=0.030,
        teammate_pred_waypoint_pull=0.35,
        fallback_uncertainty_threshold=0.75,
        prediction_guided_fallback_tries=64,
        fallback_map_weight=0.70,
        fallback_teammate_avoid_weight=0.70,
        fallback_reconnect_weight=0.35,
        fallback_distance_weight=0.40,
        use_target_belief_memory=False,
        target_belief_max_age_steps=160,
        target_belief_confidence_decay=0.006,
        target_belief_min_confidence=0.25,
        target_belief_executor_fallback=True,
        target_belief_mark_executor_soft_known=True,
        target_belief_count_as_direct_handover=False,
        # Chapter 3 PSE task-map planner. Disabled by default to keep all
        # existing Chapter-5 baselines on the original planner path.
        use_pse_planner=False,
        pse_use_belief=True,
        pse_use_exec_cost=True,
        pse_use_standby=True,
        pse_belief_detect_prob=0.75,
        pse_belief_miss_decay=0.20,
        pse_detect_sigma=1.20,
        pse_belief_topk=48,
        pse_belief_weight=1.20,
        pse_exec_cost_weight=0.18,
        pse_use_exec_cost_schedule=False,
        pse_exec_cost_weight_min=0.10,
        pse_exec_cost_weight_max=0.24,
        pse_exec_cost_schedule_warmup_steps=120,
        pse_exec_cost_entropy_low=0.45,
        pse_exec_cost_entropy_high=0.95,
        pse_search_cost_weight=0.08,
        pse_base_score_weight=0.25,
        pse_standby_topk=48,
        pse_standby_candidates=64,
        pse_standby_move_weight=0.25,
        pse_standby_hysteresis_weight=0.10,
        pse_standby_safe_weight=0.05,
        pse_lazy_standby=False,
        pse_standby_entropy_gate=0.75,
        pse_standby_min_step=80,
        pse_standby_update_interval_lazy=4,
        pse_standby_move_weight_lazy=0.40,
        pse_standby_hysteresis_weight_lazy=0.25,
        # Removed reliability-fusion switches kept only for config compatibility
        use_reliability_fusion=False,
        use_quarantine_buffer=False,
        reliability_age_lambda=1.20,
        reliability_conflict_lambda=1.50,
        # Removed channel-aware attention switches kept only for config compatibility
        use_channel_attention=False,
        use_channel_edge_features=True,
        use_channel_critical_attention=True,
        use_channel_voi_attention=True,
        use_channel_reliability_attention=True,
        use_channel_uncertainty_penalty=True,
        use_channel_role_attention=True,
        use_learned_role_edge_scorer=False,
        learned_role_edge_hidden=64,
        learned_role_edge_gain=0.50,
    ):
        if n_agents != 4:
            raise ValueError("当前环境固定为 4 个智能体：3 个搜索 + 1 个执行")

        self.device = self._resolve_device(device)
        self.return_numpy = bool(return_numpy)
        self.dtype = torch.float32
        self.eps = 1e-6

        self.n_agents = int(n_agents)
        self.num_agents = int(n_agents)
        self.n_search = 3
        self.executor_idx = 3
        self.space_size = self._vec(space_size)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.task_mode = "mission"
        self.use_obstacles = bool(use_obstacles)
        self.random_z_range = tuple(float(x) for x in random_z_range)
        self.random_search_waypoint_count = tuple(int(x) for x in random_search_waypoint_count)
        self.random_executor_waypoint_count = tuple(int(x) for x in random_executor_waypoint_count)

        self.hold_steps = int(hold_steps)
        # 搜索体连续巡航：0 表示到达 waypoint 半径就立即切换，不需要低速 hold。
        self.search_hold_steps = max(0, int(search_hold_steps))
        self.executor_hold_steps = int(hold_steps if executor_hold_steps is None else executor_hold_steps)
        self.hold_speed_thresh = float(hold_speed_thresh)
        self.planner_min_waypoint_separation = float(planner_min_waypoint_separation)

        self.role_names = ["search_fast", "search_balanced", "search_precise", "executor"]
        self.role_onehots = torch.eye(self.n_agents, dtype=self.dtype, device=self.device)
        self._eye_bool = torch.eye(self.n_agents, dtype=torch.bool, device=self.device)
        self._non_diag_bool = ~self._eye_bool
        self._zero3 = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._zero1 = torch.zeros(1, dtype=self.dtype, device=self.device)
        self._one1 = torch.ones(1, dtype=self.dtype, device=self.device)

        self.use_comm = bool(use_comm)
        self.comm_range = float(comm_range)
        self.comm_loss_prob = float(comm_loss_prob)
        self.comm_delay_steps = max(0, int(comm_delay_steps))
        self.lower_initial_sync_mode = str(lower_initial_sync_mode)
        if self.lower_initial_sync_mode not in ("none", "full", "probabilistic"):
            raise ValueError(f"Unknown lower_initial_sync_mode={self.lower_initial_sync_mode}")
        self.last_lower_initial_sync_edges = 0.0
        self.last_lower_initial_sync_mode = self.lower_initial_sync_mode
        self.base_obs_dim = 28
        self.comm_msg_dim = 20
        # Structured neighbor observation kept at the legacy width for model compatibility.
        # 37 dims = basic status fields + delivered link state + edge state +
        # 5.4 prediction fields + 5.5 reliability/conflict fields + 5.6 role score.
        self.local_obs_dim = self.base_obs_dim
        self.n_neighbors = self.n_agents - 1
        self.comm_obs_per_neighbor_dim = 37
        self.neighbor_obs_dim = self.comm_obs_per_neighbor_dim
        self.edge_feature_dim = BasicCommGraph.EDGE_FEATURE_DIM
        self.edge_feature_names = list(BasicCommGraph.EDGE_FEATURE_NAMES)
        self._edge_features = torch.zeros((self.n_agents, self.n_agents, self.edge_feature_dim), dtype=self.dtype, device=self.device)
        self._edge_weights = torch.zeros((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        self.comm_graph = {}
        self._comm_matrix = torch.zeros((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        self._comm_range_matrix = torch.zeros_like(self._comm_matrix)
        self._comm_messages = torch.zeros((self.n_agents, self.comm_msg_dim), dtype=self.dtype, device=self.device)
        self._delayed_comm_messages = self._comm_messages.clone()
        self._comm_message_history = []  # kept only for backward compatibility; new lower-layer comm uses per-link queues.
        self._message_for_receiver_cache = {}
        self._message_type_cache = {}
        self.last_comm_density = 0.0
        self.last_comm_success_rate = 0.0
        self.last_avg_neighbor_num = 0.0
        self.last_lower_avg_delay_steps = 0.0
        self.last_lower_msg_age = 0.0
        self.last_lower_comm_energy = 0.0
        self.last_lower_good_state_ratio = 1.0
        self.last_comm_energy_total = 0.0
        self.last_edge_weight = 0.0
        self.last_link_loss_prob = 0.0
        self.last_link_snr_norm = 0.0
        self.last_link_bandwidth_bps = 0.0
        self.last_comm_noise_mean = 0.0
        self.last_depth_gap_mean = 0.0
        self.last_flow_diff_mean = 0.0
        self.last_sem_target_msg_count = 0.0
        self.last_sem_map_msg_count = 0.0
        self.last_sem_waypoint_req_count = 0.0
        self.last_sem_handover_msg_count = 0.0
        self.last_sem_heartbeat_count = 0.0
        self.last_sem_risk_alert_count = 0.0
        self.last_sem_critical_count = 0.0
        self.last_sem_payload_bits = 0.0
        self.last_sem_selected_count = 0.0
        self.last_sem_critical_selected_count = 0.0
        self.last_sem_voi_score = 0.0
        self.last_sem_voi_stress = 0.0
        self.last_sem_voi_penalty_scale = 1.0
        self.last_sem_voi_adaptive_mix = 0.0
        self.last_sem_voi_type_diversity_used = 0.0
        self.last_sem_voi_comm_pressure = 0.0
        self.last_sem_voi_task_stage_id = 0.0
        self.last_sem_voi_stage_diversity_used = 0.0
        self.last_sem_voi_upper_success_context = 1.0
        self.last_sem_voi_island_context = 0.0
        self.last_sem_voi_critical_drop_context = 0.0
        self.last_sem_voi_search_diversity_disabled = 0.0
        self.last_sem_dropped_by_budget = 0.0
        self.last_sem_critical_dropped_by_budget = 0.0
        self.last_sem_critical_delivery_rate = 0.0
        self.last_sem_critical_drop_rate = 0.0
        self.last_sem_sent_packet_count = 0.0
        self.last_sem_delivered_packet_count = 0.0
        self.last_sem_arrived_packet_count = 0.0
        self.last_sem_aggregated_packet_count = 0.0
        self.last_sem_multi_packet_edge_count = 0.0
        self.last_sem_avg_packets_per_active_edge = 0.0
        self.last_sem_packet_delivery_rate = 0.0
        self.last_sem_packet_drop_rate = 0.0
        self.last_sem_critical_packet_delivery_rate = 0.0
        self.last_sem_critical_packet_drop_rate = 0.0
        self.last_role_topology_density = 1.0
        self.last_role_active_density = 1.0
        self.last_role_physical_density = 1.0
        self.last_role_pruned_ratio = 0.0
        self.last_role_critical_link_density = 0.0
        self.last_direct_handover_ready = 0.0
        self.last_direct_handover_bypass_enabled = 0.0
        self.last_direct_handover_bypass_ready = 0.0
        self.last_direct_handover_bypass_gate_pending = 0.0
        self.last_direct_handover_bypass_gate_pressure = 0.0
        self.last_adaptive_critical_filter_active = 0.0
        self.last_adaptive_critical_filter_score = 1.0
        self.last_adaptive_critical_filter_critical_boost = 0.0
        self.last_adaptive_critical_filter_ordinary_penalty = 0.0
        self.last_reconnect_voi_boost_active = 0.0
        self.last_guarded_adaptive_strength = 0.0
        self.last_guarded_medium_risk_active = 0.0
        self.last_guarded_extreme_guard_active = 0.0
        self.last_guarded_context_preserve_active = 0.0
        self.last_guarded_bypass_cooldown_active = 0.0
        self.last_direct_handover_attempts = 0.0
        self.last_direct_handover_success = 0.0
        self.last_direct_handover_episode_attempts = 0.0
        self.last_direct_handover_episode_success = 0.0
        self.last_direct_handover_count = 0.0
        self.last_handover_success_rate = 0.0
        self.last_handover_episode_success_rate = 0.0
        self._reset_residual_prior_diagnostics()
        self.last_pred_usage_ratio = 0.0  # backward-compatible alias of pred_obs_ratio
        self.last_pred_generated_ratio = 0.0
        self.last_pred_obs_ratio = 0.0
        self.last_pred_fallback_ratio = 0.0
        self.last_pred_uncertainty = 0.0
        self.last_pred_sigma = 0.0
        self.last_pred_error = 0.0
        self.last_repair_gain = 0.0
        self.last_avg_reliability = 1.0
        self.last_conflict_score = 0.0
        self.last_belief_disagreement = 0.0
        self.last_quarantine_count = 0.0
        self.last_quarantine_active_count = 0.0
        self.last_quarantine_release_count = 0.0
        self.last_quarantine_expire_count = 0.0
        self.last_map_update_reliability = 1.0
        self.last_prediction_reliability = 0.0
        self.last_target_fusion_reliability = 0.0
        self.comm_scenario_id = 0
        # 距离概率通信模型：可靠区内仅有 base loss，衰减区内成功率随距离指数下降，超过 max range 基本不可用。
        self.comm_reliable_range = float(0.70 * self.comm_range if comm_reliable_range is None else comm_reliable_range)
        self.comm_max_range = float(self.comm_range if comm_max_range is None else comm_max_range)
        self.comm_attenuation = float(max(0.0, comm_attenuation))
        self.lower_effective_sound_speed = float(max(1e-3, lower_effective_sound_speed))
        self.lower_extra_delay_steps = max(0, int(self.comm_delay_steps if lower_extra_delay_steps is None else lower_extra_delay_steps))
        self.lower_max_delay_steps = max(0, int(lower_max_delay_steps))
        self.lower_payload_bits = int(max(1, lower_payload_bits))
        self.lower_rate_bps = float(max(1.0, lower_rate_bps))
        self.lower_msg_ttl_steps = max(1, int(lower_msg_ttl_steps))
        self.use_burst_comm = bool(use_burst_comm)
        if burst_transition_matrix is None:
            burst_transition_matrix = ((0.72, 0.23, 0.05), (0.18, 0.67, 0.15), (0.10, 0.30, 0.60))
        self.burst_transition_matrix = torch.as_tensor(burst_transition_matrix, dtype=self.dtype, device=self.device)
        self.burst_state_pdr = torch.as_tensor(burst_state_pdr, dtype=self.dtype, device=self.device)
        if self.burst_transition_matrix.shape != (3, 3) or self.burst_state_pdr.numel() != 3:
            raise ValueError("burst_transition_matrix must be 3x3 and burst_state_pdr must have length 3")
        self.burst_initial_state = int(np.clip(burst_initial_state, 0, 2))
        self.payload_loss_scale = float(max(0.0, payload_loss_scale))
        self.use_comm_energy = bool(use_comm_energy)
        self.comm_idle_power = float(max(0.0, comm_idle_power))
        self.comm_rx_power = float(max(0.0, comm_rx_power))
        self.comm_tx_power = float(max(0.0, comm_tx_power))
        self.lambda_comm_energy = float(max(0.0, lambda_comm_energy))
        self.last_comm_energy = torch.zeros(self.n_agents, dtype=self.dtype, device=self.device)

        # 5.2 dynamic random communication graph configuration.
        self.use_dynamic_comm_graph = bool(use_dynamic_comm_graph)
        self.use_depth_comm_loss = bool(use_depth_comm_loss)
        self.use_flow_comm_loss = bool(use_flow_comm_loss)
        self.use_noise_comm_loss = bool(use_noise_comm_loss)
        self.use_snr_comm_model = bool(use_snr_comm_model)
        self.comm_snr_ref_db = float(comm_snr_ref_db)
        self.comm_snr_threshold_db = float(comm_snr_threshold_db)
        self.comm_snr_temperature = float(max(1e-3, comm_snr_temperature))
        self.comm_depth_loss_coeff = float(max(0.0, comm_depth_loss_coeff))
        self.comm_flow_loss_coeff = float(max(0.0, comm_flow_loss_coeff))
        self.comm_noise_loss_coeff = float(max(0.0, comm_noise_loss_coeff))
        self.comm_noise_base = float(np.clip(comm_noise_base, 0.0, 1.0))
        self.comm_noise_spike_prob = float(np.clip(comm_noise_spike_prob, 0.0, 1.0))
        self.comm_noise_spike_scale = float(max(0.0, comm_noise_spike_scale))
        self.comm_min_rate_bps = float(max(1.0, comm_min_rate_bps))
        self.comm_max_rate_bps = float(max(self.comm_min_rate_bps + 1.0, comm_max_rate_bps))
        self.edge_delay_lambda = float(max(0.0, edge_delay_lambda))
        self.edge_age_lambda = float(max(0.0, edge_age_lambda))
        self.edge_queue_lambda = float(max(0.0, edge_queue_lambda))

        # 5.3 semantic communication and VOI selector.
        self.use_semantic_messages = bool(use_semantic_messages)
        self.use_voi_selector = bool(use_voi_selector)
        self.use_critical_priority = bool(use_critical_priority)
        self.force_heartbeat_only = bool(force_heartbeat_only)
        self.comm_budget_bits_per_step = int(max(1, comm_budget_bits_per_step))
        self.lower_message_topk = int(max(1, lower_message_topk))
        self.critical_reserve_slots = int(max(0, critical_reserve_slots))
        self.use_adaptive_voi = bool(use_adaptive_voi)
        self.voi_stress_loss_gain = float(voi_stress_loss_gain)
        self.voi_stress_delay_gain = float(voi_stress_delay_gain)
        self.voi_max_penalty_suppression = float(max(0.0, min(1.0, voi_max_penalty_suppression)))
        self.voi_priority_fallback_gain = float(voi_priority_fallback_gain)
        self.voi_critical_fallback_gain = float(voi_critical_fallback_gain)
        self.voi_receiver_need_fallback_gain = float(voi_receiver_need_fallback_gain)
        self.voi_type_diversity = bool(voi_type_diversity)
        self.voi_diversity_stress_threshold = float(max(0.0, min(1.0, voi_diversity_stress_threshold)))
        self.voi_max_same_type_per_topk = int(max(1, voi_max_same_type_per_topk))
        self.voi_stress_critical_bonus_gain = float(voi_stress_critical_bonus_gain)
        self.voi_critical_bypass_diversity = bool(voi_critical_bypass_diversity)
        self.use_stage_aware_voi = bool(use_stage_aware_voi)
        self.disable_handover_diversity_gate = bool(disable_handover_diversity_gate)
        self.disable_search_diversity_gate = bool(disable_search_diversity_gate)
        self.voi_handover_window_steps = int(max(1, voi_handover_window_steps))
        self.voi_search_diversity_pressure_threshold = float(max(0.0, min(1.0, voi_search_diversity_pressure_threshold)))
        self.voi_upper_success_bad_threshold = float(max(0.0, min(1.0, voi_upper_success_bad_threshold)))
        self.voi_island_diversity_threshold = float(max(0.0, min(1.0, voi_island_diversity_threshold)))
        self.voi_execute_diversity_pressure_threshold = float(max(0.0, min(1.0, voi_execute_diversity_pressure_threshold)))
        self.use_multi_packet_semantic_comm = bool(use_multi_packet_semantic_comm and self.use_semantic_messages)
        self.multi_packet_aggregate_mode = str(multi_packet_aggregate_mode)
        if self.multi_packet_aggregate_mode not in ("voi_reliability", "primary"):
            raise ValueError(f"Unknown multi_packet_aggregate_mode={self.multi_packet_aggregate_mode}")
        self.multi_packet_max_per_edge = None if multi_packet_max_per_edge is None else int(max(1, multi_packet_max_per_edge))
        if self.use_semantic_messages or self.use_voi_selector:
            raise RuntimeError("Semantic communication and VOI selection were removed from this Chapter-4 workspace.")

        # 5.3b role-aware mission-chain topology.  This layer masks the
        # dynamic physical graph by role/message semantics, but keeps the
        # neural-network observation dimension unchanged.
        self.use_role_topology = bool(use_role_topology)
        self.role_topology_mode = str(role_topology_mode)
        self.use_direct_handover_lane = bool(use_direct_handover_lane)
        self.role_topology_allow_executor_status = bool(role_topology_allow_executor_status)
        self.role_topology_allow_searcher_executor_pre_found = bool(role_topology_allow_searcher_executor_pre_found)
        self.role_topology_strict_message_filter = bool(role_topology_strict_message_filter)
        self.direct_handover_bypass_message_filter = bool(direct_handover_bypass_message_filter)
        self.use_adaptive_handover_bypass = bool(use_adaptive_handover_bypass)
        self.adaptive_handover_bypass_min_handoff_age_steps = int(max(0, adaptive_handover_bypass_min_handoff_age_steps))
        self.adaptive_handover_bypass_max_handoff_age_steps = int(max(self.adaptive_handover_bypass_min_handoff_age_steps, adaptive_handover_bypass_max_handoff_age_steps))
        self.adaptive_handover_bypass_comm_pressure_threshold = float(np.clip(adaptive_handover_bypass_comm_pressure_threshold, 0.0, 1.0))
        self.adaptive_handover_bypass_island_threshold = float(np.clip(adaptive_handover_bypass_island_threshold, 0.0, 1.0))
        self.adaptive_handover_bypass_upper_success_bad_threshold = float(np.clip(adaptive_handover_bypass_upper_success_bad_threshold, 0.0, 1.0))
        self.adaptive_handover_bypass_require_executor_unknown = bool(adaptive_handover_bypass_require_executor_unknown)
        self.use_adaptive_critical_semantic_filter = bool(use_adaptive_critical_semantic_filter)
        self.use_guarded_adaptive_critical_filter = bool(use_guarded_adaptive_critical_filter)
        self.adaptive_critical_filter_default_soft = bool(adaptive_critical_filter_default_soft)
        self.adaptive_critical_filter_preserve_ordinary_strict = bool(adaptive_critical_filter_preserve_ordinary_strict)
        self.adaptive_critical_filter_msg_types = tuple(str(x) for x in adaptive_critical_filter_msg_types)
        self.adaptive_critical_filter_comm_pressure_threshold = float(np.clip(adaptive_critical_filter_comm_pressure_threshold, 0.0, 1.0))
        self.adaptive_critical_filter_island_threshold = float(np.clip(adaptive_critical_filter_island_threshold, 0.0, 1.0))
        self.adaptive_critical_filter_upper_success_bad_threshold = float(np.clip(adaptive_critical_filter_upper_success_bad_threshold, 0.0, 1.0))
        self.adaptive_critical_filter_min_handoff_age_steps = int(max(0, adaptive_critical_filter_min_handoff_age_steps))
        self.adaptive_critical_filter_require_executor_unknown = bool(adaptive_critical_filter_require_executor_unknown)
        self.adaptive_ordinary_message_penalty = float(max(0.0, adaptive_ordinary_message_penalty))
        self.adaptive_noncritical_cross_role_penalty = float(max(0.0, adaptive_noncritical_cross_role_penalty))
        self.adaptive_critical_message_bonus = float(max(0.0, adaptive_critical_message_bonus))
        self.adaptive_reconnect_voi_boost = bool(adaptive_reconnect_voi_boost)
        self.reconnect_voi_boost_gain = float(max(0.0, reconnect_voi_boost_gain))
        self.reconnect_voi_boost_island_threshold = float(np.clip(reconnect_voi_boost_island_threshold, 0.0, 1.0))
        self.guarded_adaptive_comm_pressure_min = float(np.clip(guarded_adaptive_comm_pressure_min, 0.0, 1.0))
        self.guarded_adaptive_comm_pressure_max = float(np.clip(guarded_adaptive_comm_pressure_max, 0.0, 1.0))
        self.guarded_adaptive_island_min = float(np.clip(guarded_adaptive_island_min, 0.0, 1.0))
        self.guarded_adaptive_island_max = float(np.clip(guarded_adaptive_island_max, 0.0, 1.0))
        self.guarded_adaptive_upper_success_min = float(np.clip(guarded_adaptive_upper_success_min, 0.0, 1.0))
        self.guarded_adaptive_critical_drop_max = float(np.clip(guarded_adaptive_critical_drop_max, 0.0, 1.0))
        self.guarded_extreme_comm_pressure_threshold = float(np.clip(guarded_extreme_comm_pressure_threshold, 0.0, 1.0))
        self.guarded_extreme_island_threshold = float(np.clip(guarded_extreme_island_threshold, 0.0, 1.0))
        self.guarded_extreme_upper_success_threshold = float(np.clip(guarded_extreme_upper_success_threshold, 0.0, 1.0))
        self.guarded_extreme_critical_drop_threshold = float(np.clip(guarded_extreme_critical_drop_threshold, 0.0, 1.0))
        self.guarded_handover_bypass_cooldown_steps = int(max(0, guarded_handover_bypass_cooldown_steps))
        self.guarded_handover_bypass_max_per_window = int(max(1, guarded_handover_bypass_max_per_window))
        self.guarded_preserve_context_slot = bool(guarded_preserve_context_slot)
        self.guarded_max_adaptive_critical_per_topk = int(max(1, guarded_max_adaptive_critical_per_topk))
        self.guarded_allow_double_critical_when_handoff_pending = bool(guarded_allow_double_critical_when_handoff_pending)
        self.direct_handover_max_age_steps = int(max(1, direct_handover_max_age_steps))
        if self.use_role_topology or self.use_direct_handover_lane:
            raise RuntimeError("Role-topology and direct-handover communication code was removed from this Chapter-4 workspace.")
        self._role_topology_mask = torch.ones((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        self._role_topology_mask.fill_diagonal_(0.0)
        self._role_score_matrix = self._role_topology_mask.clone()
        self._direct_handover_bypass_mask = torch.zeros((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        self._direct_handover_bypass_probs = torch.zeros((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        self._guarded_bypass_cooldown = torch.zeros(self.n_agents, dtype=torch.int64, device=self.device)
        self._guarded_bypass_window_count = torch.zeros(self.n_agents, dtype=torch.int64, device=self.device)
        self._guarded_bypass_window_start_step = torch.zeros(self.n_agents, dtype=torch.int64, device=self.device)
        self._guarded_bypass_last_trigger_step = torch.full((self.n_agents,), -1, dtype=torch.int64, device=self.device)

        # 5.4 teammate prediction and prediction-guided fallback.
        self.use_teammate_prediction = bool(use_teammate_prediction)
        self.use_prediction_guided_fallback = bool(use_prediction_guided_fallback)
        self.fallback_uncertainty_threshold = float(np.clip(fallback_uncertainty_threshold, 0.0, 1.0))
        self.prediction_guided_fallback_tries = int(max(4, prediction_guided_fallback_tries))
        self.fallback_map_weight = float(fallback_map_weight)
        self.fallback_teammate_avoid_weight = float(fallback_teammate_avoid_weight)
        self.fallback_reconnect_weight = float(fallback_reconnect_weight)
        self.fallback_distance_weight = float(fallback_distance_weight)
        self.use_target_belief_memory = bool(use_target_belief_memory)
        self.target_belief_max_age_steps = int(max(1, target_belief_max_age_steps))
        self.target_belief_confidence_decay = float(max(0.0, target_belief_confidence_decay))
        self.target_belief_min_confidence = float(np.clip(target_belief_min_confidence, 0.0, 1.0))
        self.target_belief_executor_fallback = bool(target_belief_executor_fallback)
        self.target_belief_mark_executor_soft_known = bool(target_belief_mark_executor_soft_known)
        self.target_belief_count_as_direct_handover = bool(target_belief_count_as_direct_handover)

        self.use_pse_planner = bool(use_pse_planner)
        self.pse_use_belief = bool(pse_use_belief)
        self.pse_use_exec_cost = bool(pse_use_exec_cost)
        self.pse_use_standby = bool(pse_use_standby)
        self.pse_belief_detect_prob = float(pse_belief_detect_prob)
        self.pse_belief_miss_decay = float(pse_belief_miss_decay)
        self.pse_detect_sigma = float(pse_detect_sigma)
        self.pse_belief_topk = int(max(1, pse_belief_topk))
        self.pse_belief_weight = float(pse_belief_weight)
        self.pse_exec_cost_weight = float(pse_exec_cost_weight)
        self.pse_use_exec_cost_schedule = bool(pse_use_exec_cost_schedule)
        self.pse_exec_cost_weight_min = float(pse_exec_cost_weight_min)
        self.pse_exec_cost_weight_max = float(pse_exec_cost_weight_max)
        self.pse_exec_cost_schedule_warmup_steps = int(max(1, pse_exec_cost_schedule_warmup_steps))
        self.pse_exec_cost_entropy_low = float(pse_exec_cost_entropy_low)
        self.pse_exec_cost_entropy_high = float(pse_exec_cost_entropy_high)
        self.pse_search_cost_weight = float(pse_search_cost_weight)
        self.pse_base_score_weight = float(pse_base_score_weight)
        self.pse_standby_topk = int(max(1, pse_standby_topk))
        self.pse_standby_candidates = int(max(1, pse_standby_candidates))
        self.pse_standby_move_weight = float(pse_standby_move_weight)
        self.pse_standby_hysteresis_weight = float(pse_standby_hysteresis_weight)
        self.pse_standby_safe_weight = float(pse_standby_safe_weight)
        self.pse_lazy_standby = bool(pse_lazy_standby)
        self.pse_standby_entropy_gate = float(pse_standby_entropy_gate)
        self.pse_standby_min_step = int(max(0, pse_standby_min_step))
        self.pse_standby_update_interval_lazy = int(max(1, pse_standby_update_interval_lazy))
        self.pse_standby_move_weight_lazy = float(pse_standby_move_weight_lazy)
        self.pse_standby_hysteresis_weight_lazy = float(pse_standby_hysteresis_weight_lazy)
        self.last_belief_entropy = 0.0
        self.last_standby_to_target_dist = 0.0
        self.last_exec_response_cost = 0.0
        self.last_search_score_mean = 0.0
        self.last_pse_claim_overlap = 0.0
        self.last_pse_fallback_fast_path_used = 0.0
        self.last_pse_fallback_candidate_count = 0.0
        self.last_pse_exec_cost_weight_effective = float(self.pse_exec_cost_weight)
        self.last_pse_exec_cost_schedule_factor = 0.0
        self.last_pse_lazy_standby_active = 0.0
        self.last_pse_standby_update_allowed = 1.0
        self.last_pse_standby_update_skipped_by_lazy_gate = 0.0
        self.last_pse_standby_update_used = 0.0
        self.last_pse_standby_update_interval = float(max(1, int(planner_step_update_interval)))
        self._last_pse_standby_update_step = -1
        self.last_success_step_minus_found_step = float("nan")
        self.use_reliability_fusion = bool(use_reliability_fusion)
        self.use_quarantine_buffer = bool(use_quarantine_buffer)
        if self.use_teammate_prediction or self.use_prediction_guided_fallback:
            raise RuntimeError("Teammate prediction communication code was removed from this Chapter-4 workspace.")
        if self.use_reliability_fusion or self.use_quarantine_buffer:
            raise RuntimeError("Reliability fusion and quarantine communication code was removed from this Chapter-4 workspace.")

        # 5.6 lightweight channel-aware attention actor switches.
        self.use_channel_attention = bool(use_channel_attention)
        self.use_channel_edge_features = bool(use_channel_edge_features)
        self.use_channel_critical_attention = bool(use_channel_critical_attention)
        self.use_channel_voi_attention = bool(use_channel_voi_attention)
        self.use_channel_reliability_attention = bool(use_channel_reliability_attention)
        self.use_channel_uncertainty_penalty = bool(use_channel_uncertainty_penalty)
        self.use_channel_role_attention = bool(use_channel_role_attention)
        self.use_learned_role_edge_scorer = bool(use_learned_role_edge_scorer)
        self.learned_role_edge_hidden = int(max(1, learned_role_edge_hidden))
        self.learned_role_edge_gain = float(learned_role_edge_gain)
        removed_comm_switches = [
            name for name in (
                "use_dynamic_comm_graph",
                "use_depth_comm_loss",
                "use_flow_comm_loss",
                "use_noise_comm_loss",
                "use_snr_comm_model",
                "use_semantic_messages",
                "use_voi_selector",
                "use_role_topology",
                "use_direct_handover_lane",
                "use_teammate_prediction",
                "use_prediction_guided_fallback",
                "use_reliability_fusion",
                "use_quarantine_buffer",
                "use_channel_attention",
                "use_learned_role_edge_scorer",
            )
            if bool(getattr(self, name, False))
        ]
        if removed_comm_switches:
            raise RuntimeError(f"Removed Chapter-5 communication switches are enabled: {removed_comm_switches}")
        if self.use_channel_attention or self.use_learned_role_edge_scorer:
            raise RuntimeError("Channel-aware communication attention code was removed from this Chapter-4 workspace.")

        self.lower_comm_graph = BasicCommGraph(self._make_lower_comm_graph_config())
        self.use_upper_comm = bool(use_upper_comm)

        # 上层决策模块通信。upper_node_pos 默认放在水面中心，可理解为浮标/母船/中继节点。
        self.use_upper_comm = bool(use_upper_comm)
        self.upper_node_pos = self._vec(
            upper_node_pos if upper_node_pos is not None else (
                float(space_size[0]) * 0.5,
                float(space_size[1]) * 0.5,
                float(space_size[2]),
            )
        )
        self.upper_comm_reliable_range = float(upper_comm_reliable_range)
        self.upper_comm_max_range = float(upper_comm_max_range)
        self.upper_comm_base_loss = float(np.clip(upper_comm_base_loss, 0.0, 1.0))
        self.upper_comm_attenuation = float(max(0.0, upper_comm_attenuation))
        self.upper_effective_sound_speed = float(max(1e-3, upper_effective_sound_speed))
        self.upper_extra_delay_steps = max(0, int(upper_extra_delay_steps))
        self.upper_map_update_interval = max(1, int(planner_step_update_interval if upper_map_update_interval is None else upper_map_update_interval))
        self.upper_wait_timeout_steps = max(1, int(upper_wait_timeout_steps))
        self.upper_map_payload_bits = int(max(1, upper_map_payload_bits))
        self.upper_state_payload_bits = int(max(1, upper_state_payload_bits))
        self.upper_waypoint_payload_bits = int(max(1, upper_waypoint_payload_bits))
        self.upper_task_payload_bits = int(max(1, upper_task_payload_bits))
        self.upper_rate_bps = float(max(1.0, upper_rate_bps))
        self.use_upper_belief = bool(use_upper_belief)
        self.upper_belief_pos_noise_std = float(max(0.0, upper_belief_pos_noise_std))
        self.last_upper_comm_density = 0.0
        self.last_upper_uplink_success_rate = 0.0
        self.last_upper_downlink_success_rate = 0.0
        self.last_upper_uplink_success_rate_valid = float("nan")
        self.last_upper_downlink_success_rate_valid = float("nan")
        self.last_upper_uplink_has_attempt = 0.0
        self.last_upper_downlink_has_attempt = 0.0
        self.last_upper_up_attempts = 0.0
        self.last_upper_down_attempts = 0.0
        self.last_upper_up_successes = 0.0
        self.last_upper_down_successes = 0.0
        self.last_upper_total_attempts = 0.0
        self.last_upper_total_successes = 0.0
        self.last_upper_has_attempt = 0.0
        self.last_upper_avg_delay_steps = 0.0
        self.last_upper_map_uploads = 0
        self.last_upper_waypoint_downlinks = 0
        self.last_upper_target_downlinks = 0
        self.last_upper_comm_energy = 0.0
        self.last_upper_msg_age = 0.0
        self.last_upper_good_state_ratio = 1.0
        self.last_upper_belief_age = 0.0
        self.last_local_fallback_ratio = 0.0
        self.last_handoff_delay = float("nan")
        self.use_reconnect_lane = bool(use_reconnect_lane)
        if self.use_reconnect_lane:
            raise RuntimeError("Reconnect-lane communication code was removed from this Chapter-4 workspace.")
        island_timeout = upper_wait_timeout_steps if reconnect_island_timeout_steps is None else reconnect_island_timeout_steps
        self.reconnect_island_timeout_steps = max(1, int(island_timeout))
        self.last_island_duration = 0.0
        self.last_island_agent_ratio = 0.0
        self.last_island_count = 0.0
        self.last_reconnect_count = 0.0
        self.last_reconnect_success_count = 0.0
        self.last_avg_reconnect_time = 0.0
        self.last_reconnect_lane_score = 0.0
        self.last_upper_role_topology_density = 0.0
        self.last_upper_role_active_density = 0.0
        self.last_upper_role_physical_density = 0.0
        self._reconnect_time_sum_current = 0.0
        self._reconnect_time_count_current = 0

        self.diverse_fallback_prob = float(np.clip(diverse_fallback_prob, 0.0, 1.0))
        self.diverse_fallback_tries = max(4, int(diverse_fallback_tries))
        self.search_spread_reward_gain = float(search_spread_reward_gain)
        self.detect_proximity_reward_gain = float(detect_proximity_reward_gain)
        self.detect_proximity_radius = float(max(1.0, detect_proximity_radius))
        self.planner_coverage_weight = float(planner_coverage_weight)
        self.planner_claim_weight = float(planner_claim_weight)
        self.planner_stochastic_topk = max(1, int(planner_stochastic_topk))
        self.planner_stochastic_eps = float(np.clip(planner_stochastic_eps, 0.0, 1.0))

        self.use_residual_prior = bool(use_residual_prior)
        self.prior_kv_xy = float(prior_kv_xy)
        self.prior_kv_z = float(prior_kv_z)
        self.prior_slow_radius_xy = float(max(prior_slow_radius_xy, 1e-3))
        self.prior_slow_radius_z = float(max(prior_slow_radius_z, 1e-3))
        self.prior_strength_search = float(prior_strength_search)
        self.prior_strength_executor = float(prior_strength_executor)
        self.residual_scale_search = float(residual_scale_search)
        self.residual_scale_executor = float(residual_scale_executor)
        self.residual_action_mode = str(residual_action_mode)
        if self.residual_action_mode not in ("cartesian", "target_decomposed_v1", "hybrid_v4"):
            raise ValueError(f"Unknown residual_action_mode={self.residual_action_mode}")
        self.residual_struct_base_gate = float(residual_struct_base_gate)
        self.residual_struct_low_risk_gate = float(residual_struct_low_risk_gate)
        self.residual_struct_high_risk_gate = float(residual_struct_high_risk_gate)
        self.residual_struct_lateral_scale_search = float(residual_struct_lateral_scale_search)
        self.residual_struct_lateral_scale_exec = float(residual_struct_lateral_scale_exec)
        self.residual_struct_antitarget_scale = float(residual_struct_antitarget_scale)
        self.residual_struct_near_target_gate = float(residual_struct_near_target_gate)
        self.residual_struct_near_radius_search = float(max(0.0, residual_struct_near_radius_search))
        self.residual_struct_near_radius_exec = float(max(0.0, residual_struct_near_radius_exec))
        self.residual_struct_unknown_target_gate_exec = float(residual_struct_unknown_target_gate_exec)
        self.residual_hybrid_exec_cartesian_blend_known = float(np.clip(residual_hybrid_exec_cartesian_blend_known, 0.0, 1.0))
        self.residual_hybrid_exec_cartesian_blend_found = float(np.clip(residual_hybrid_exec_cartesian_blend_found, 0.0, 1.0))
        self.residual_hybrid_exec_cartesian_blend_unknown = float(np.clip(residual_hybrid_exec_cartesian_blend_unknown, 0.0, 1.0))
        self.residual_hybrid_exec_struct_blend_min = float(np.clip(residual_hybrid_exec_struct_blend_min, 0.0, 1.0))
        self.residual_hybrid_target_known_soft_gate = float(np.clip(residual_hybrid_target_known_soft_gate, 0.0, 1.0))
        self.residual_hybrid_target_found_soft_gate = float(np.clip(residual_hybrid_target_found_soft_gate, 0.0, 1.0))
        self.residual_hybrid_target_unknown_soft_gate = float(np.clip(residual_hybrid_target_unknown_soft_gate, 0.0, 1.0))
        self.residual_hybrid_use_soft_gate = bool(residual_hybrid_use_soft_gate)

        # Robust disturbance configuration. The prior controller still uses the
        # nominal model, while the simulator can use disturbed dynamics and
        # actuation. This makes Prior-only imperfect under robust profiles and
        # gives the learned residual a clear compensation role.
        self.use_robust_disturbance = bool(use_robust_disturbance)
        self.flow_gain_range = self._range_tuple(flow_gain_range, 2)
        self.flow_z_gain_range = self._range_tuple(flow_z_gain_range, 2)
        self.flow_phase_random = bool(flow_phase_random)
        self.a_max_scale_range = self._range_tuple(a_max_scale_range, 2)
        self.v_max_scale_range = self._range_tuple(v_max_scale_range, 2)
        self.drag_scale_range = self._range_tuple(drag_scale_range, 2)
        self.buoyancy_bias_delta_range = self._range_tuple(buoyancy_bias_delta_range, 2)
        self.actuator_lag = float(np.clip(actuator_lag, 0.0, 0.95))
        self.robust_action_delay_steps = max(0, int(robust_action_delay_steps))
        self.action_noise_std = float(max(0.0, action_noise_std))
        self.action_noise_std_range = None if action_noise_std_range is None else self._range_tuple(action_noise_std_range, 2)
        self.residual_penalty = float(max(0.0, residual_penalty))
        self.last_residual_norm = 0.0
        self.reward_profile = str(reward_profile)
        if self.reward_profile not in ("original", "robust_residual_v2", "residual_point_v3"):
            raise ValueError(f"Unknown reward_profile={self.reward_profile}")
        self.exec_task_progress_gain_v2 = float(exec_task_progress_gain_v2)
        self.near_target_speed_penalty_gain_v2 = float(near_target_speed_penalty_gain_v2)
        self.near_target_speed_radius_v2 = float(max(0.0, near_target_speed_radius_v2))
        self.hold_dense_reward_gain_v2 = float(hold_dense_reward_gain_v2)
        self.target_assigned_bonus_exec_v2 = float(target_assigned_bonus_exec_v2)
        self.target_assigned_bonus_search_v2 = float(target_assigned_bonus_search_v2)
        self.handoff_delay_penalty_gain_v2 = float(handoff_delay_penalty_gain_v2)
        self.belief_age_penalty_gain_v2 = float(belief_age_penalty_gain_v2)
        self.belief_age_penalty_norm_v2 = float(max(1e-6, belief_age_penalty_norm_v2))
        self.low_risk_residual_penalty_v2 = float(max(0.0, low_risk_residual_penalty_v2))
        self.high_risk_residual_penalty_v2 = float(max(0.0, high_risk_residual_penalty_v2))
        self.anti_prior_penalty_gain_v2 = float(max(0.0, anti_prior_penalty_gain_v2))
        self.near_target_residual_penalty_gain_v2 = float(max(0.0, near_target_residual_penalty_gain_v2))
        self.near_target_residual_penalty_radius_v2 = float(max(0.0, near_target_residual_penalty_radius_v2))
        self.point_vel_align_gain_search_v3 = float(point_vel_align_gain_search_v3)
        self.point_vel_align_gain_exec_v3 = float(point_vel_align_gain_exec_v3)
        self.point_lateral_vel_penalty_search_v3 = float(max(0.0, point_lateral_vel_penalty_search_v3))
        self.point_lateral_vel_penalty_exec_v3 = float(max(0.0, point_lateral_vel_penalty_exec_v3))
        self.point_speed_profile_penalty_search_v3 = float(max(0.0, point_speed_profile_penalty_search_v3))
        self.point_speed_profile_penalty_exec_v3 = float(max(0.0, point_speed_profile_penalty_exec_v3))
        self.point_residual_align_gain_search_v3 = float(point_residual_align_gain_search_v3)
        self.point_residual_align_gain_exec_v3 = float(point_residual_align_gain_exec_v3)
        self.point_residual_antitarget_penalty_search_v3 = float(max(0.0, point_residual_antitarget_penalty_search_v3))
        self.point_residual_antitarget_penalty_exec_v3 = float(max(0.0, point_residual_antitarget_penalty_exec_v3))
        self.point_slow_radius_search_v3 = float(max(1e-6, point_slow_radius_search_v3))
        self.point_slow_radius_exec_v3 = float(max(1e-6, point_slow_radius_exec_v3))
        self.point_near_radius_search_v3 = float(max(0.0, point_near_radius_search_v3))
        self.point_near_radius_exec_v3 = float(max(0.0, point_near_radius_exec_v3))
        self.point_shaping_total_clip_v3 = float(max(0.0, point_shaping_total_clip_v3))
        self._reset_robust_residual_reward_diagnostics_v2()
        self._reset_point_control_reward_diagnostics_v3()

        self.agent_specs = [
            dict(name="search_fast", type="search", a_xy_max=1.30, a_z_max=0.75, v_xy_max=2.80, v_z_max=1.20,
                 drag_xy=0.10, drag_z=0.16, buoyancy_bias=0.00, sensor_range=2.00, energy_coeff=1.15,
                 progress_gain=20.0, waypoint_bonus=12.0, detect_bonus=120.0),
            dict(name="search_balanced", type="search", a_xy_max=1.00, a_z_max=0.70, v_xy_max=2.20, v_z_max=1.00,
                 drag_xy=0.12, drag_z=0.18, buoyancy_bias=0.00, sensor_range=2.35, energy_coeff=1.00,
                 progress_gain=18.0, waypoint_bonus=10.0, detect_bonus=140.0),
            dict(name="search_precise", type="search", a_xy_max=0.90, a_z_max=0.65, v_xy_max=1.80, v_z_max=0.90,
                 drag_xy=0.14, drag_z=0.20, buoyancy_bias=0.00, sensor_range=2.75, energy_coeff=0.90,
                 progress_gain=16.0, waypoint_bonus=9.0, detect_bonus=160.0),
            dict(name="executor", type="execute", a_xy_max=0.80, a_z_max=0.55, v_xy_max=1.50, v_z_max=0.75,
                 drag_xy=0.16, drag_z=0.24, buoyancy_bias=0.02, sensor_range=0.0, energy_coeff=1.05,
                 progress_gain=28.0, waypoint_bonus=0.0, detect_bonus=0.0),
        ]
        self._build_agent_spec_tensors()
        self._init_robust_state()

        self.safe_dist = 1.6
        self.collision_penalty = -60.0
        self.sep_penalty_k = 1.2
        self.time_penalty = 0.03
        self.lambda_a = 0.010
        self.lambda_da = 0.040
        self.reward_scale = 100.0

        self.search_arrive_eps = 0.9
        self.detect_eps_bias = 0.10
        self.executor_arrive_eps = 1.0
        self.executor_hold_radius = 0.8
        self.executor_hold_bonus = 1.2
        self.mission_complete_bonus = 300.0
        self.team_find_bonus = 20.0
        self.finder_extra_bonus = 40.0
        self.search_task_progress_bonus = 6.0
        self.coverage_reward_gain = 100.0
        self.flow_gain = 0.18
        self.flow_z_gain = 0.0

        self.default_obstacles = [
            {"center": np.array([5.0, 5.0, 2.0], dtype=np.float32), "size": np.array([2.5, 2.5, 2.0], dtype=np.float32)},
            {"center": np.array([11.0, 10.0, 4.0], dtype=np.float32), "size": np.array([3.0, 3.0, 2.5], dtype=np.float32)},
            {"center": np.array([15.5, 6.0, 5.5], dtype=np.float32), "size": np.array([2.0, 3.0, 2.0], dtype=np.float32)},
        ]
        self.obstacles = self.default_obstacles if self.use_obstacles else []
        self._build_obstacle_tensors()

        self._lower_bound = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._sampling_margin_min = self._vec([0.0, 0.0, 0.5])

        self.step_count = 0
        self._agent_pos = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._agent_vel = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._agent_acc = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._prev_acc = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._nav_targets = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._targets = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._prev_nav_distances = torch.zeros(self.n_agents, dtype=self.dtype, device=self.device)
        self._collision_flags = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._task_target = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._search_waypoints = torch.zeros((self.n_search, 3), dtype=self.dtype, device=self.device)
        self._executor_wait_point = torch.zeros(3, dtype=self.dtype, device=self.device)

        self.task_found = False
        self.finder_idx = -1
        self.mission_complete = False
        self.search_stage_complete = False
        self.executor_target_assigned = False
        self.executor_wait_held = False
        self.last_direct_task_assignment = 0.0

        self.waypoint_reached_counts = torch.zeros(self.n_agents, dtype=torch.int32, device=self.device)
        self.hold_success_counts = torch.zeros(self.n_agents, dtype=torch.int32, device=self.device)
        self.total_waypoints_per_agent = torch.zeros(self.n_agents, dtype=torch.int32, device=self.device)
        self.agent_finished = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self.just_reached_waypoint = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self.just_held_target = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self.hold_counters = torch.zeros(self.n_agents, dtype=torch.int32, device=self.device)
        self.current_target_arrived = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)

        self._found_event = False
        self._mission_complete_event = False
        self._executor_wait_hold_event = False
        self._prev_coverage_ratio = 0.0
        self._prev_search_task_min_dist = 0.0

        self._agent_task_known = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._agent_task_est = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._upper_task_known = False
        self._upper_task_est = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._pending_upper_upload = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._waiting_upper_waypoint = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._upper_wait_steps = torch.zeros(self.n_agents, dtype=torch.int32, device=self.device)
        self._using_local_fallback = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._upper_uplink_queue = []
        self._upper_downlink_queue = []
        self._found_step = None
        self._executor_received_target_step = None
        self._upper_comm_attempts = {"up": 0, "down": 0, "up_success": 0, "down_success": 0, "reachable": 0}
        self._upper_delay_sum = 0.0
        self._upper_delay_count = 0
        self._upper_last_contact_step = torch.zeros(self.n_agents, dtype=torch.int32, device=self.device)
        self._upper_island_active = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._upper_reconnect_active = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._upper_reconnect_start_step = torch.full((self.n_agents,), -1, dtype=torch.int32, device=self.device)
        self._lower_link_queues = [[[] for _ in range(self.n_agents)] for _ in range(self.n_agents)]
        self._lower_delivered_messages = torch.zeros((self.n_agents, self.n_agents, self.comm_msg_dim), dtype=self.dtype, device=self.device)
        self._lower_delivered_steps = torch.full((self.n_agents, self.n_agents), -10_000, dtype=torch.int32, device=self.device)
        self._lower_delivered_quality = torch.zeros((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        self._lower_delivered_state = torch.full((self.n_agents, self.n_agents), 2, dtype=torch.long, device=self.device)
        self._lower_delivered_mask = torch.zeros((self.n_agents, self.n_agents), dtype=torch.bool, device=self.device)
        self._lower_link_state = torch.full((self.n_agents, self.n_agents), self.burst_initial_state, dtype=torch.long, device=self.device)
        self._upper_link_state = torch.full((self.n_agents, 2), self.burst_initial_state, dtype=torch.long, device=self.device)
        self._upper_belief_pos = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._upper_belief_vel = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._upper_belief_step = torch.full((self.n_agents,), -10_000, dtype=torch.int32, device=self.device)
        self._upper_belief_valid = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)

        planner_cls = ProbabilisticTaskMapPlanner if self.use_pse_planner else PheromoneWaypointPlanner
        planner_kwargs = dict(
            space_size=self.space_size,
            n_agents=self.n_agents,
            n_search=self.n_search,
            executor_idx=self.executor_idx,
            grid_size=planner_grid_size,
            z_range=self.random_z_range,
            search_count_range=self.random_search_waypoint_count,
            executor_count_range=self.random_executor_waypoint_count,
            visit_radius=planner_visit_radius,
            pheromone_decay=planner_pheromone_decay,
            suppression=planner_suppression,
            min_waypoint_separation=planner_min_waypoint_separation,
            coverage_weight=self.planner_coverage_weight,
            claim_weight=self.planner_claim_weight,
            stochastic_topk=self.planner_stochastic_topk,
            stochastic_eps=self.planner_stochastic_eps,
            device=self.device,
            dtype=self.dtype,
        )
        if self.use_pse_planner:
            planner_kwargs.update(
                pse_belief_detect_prob=self.pse_belief_detect_prob,
                pse_belief_miss_decay=self.pse_belief_miss_decay,
                pse_detect_sigma=self.pse_detect_sigma,
                pse_belief_topk=self.pse_belief_topk,
                pse_belief_weight=self.pse_belief_weight,
                pse_exec_cost_weight=self.pse_exec_cost_weight,
                pse_search_cost_weight=self.pse_search_cost_weight,
                pse_base_score_weight=self.pse_base_score_weight,
                pse_standby_topk=self.pse_standby_topk,
                pse_standby_candidates=self.pse_standby_candidates,
                pse_standby_move_weight=self.pse_standby_move_weight,
                pse_standby_hysteresis_weight=self.pse_standby_hysteresis_weight,
                pse_standby_safe_weight=self.pse_standby_safe_weight,
            )
        self.map_module = planner_cls(**planner_kwargs)
        self.planner_step_update_interval = max(1, int(planner_step_update_interval))
        self.planner_step_update_suppress_only = bool(planner_step_update_suppress_only)
        self._reset_target_belief_memory()
        self.reset()

    @property
    def agent_pos(self): return self._to_public(self._agent_pos)
    @property
    def agent_vel(self): return self._to_public(self._agent_vel)
    @property
    def agent_acc(self): return self._to_public(self._agent_acc)
    @property
    def prev_acc(self): return self._to_public(self._prev_acc)
    @property
    def nav_targets(self): return self._to_public(self._nav_targets)
    @property
    def targets(self): return self._to_public(self._targets)
    @property
    def collision_flags(self): return self._to_public(self._collision_flags)
    @property
    def task_target(self): return self._to_public(self._task_target)
    @property
    def edge_features(self): return self._to_public(self._edge_features)
    @property
    def edge_weights(self): return self._to_public(self._edge_weights)

    def _resolve_device(self, device):
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _range_tuple(self, x, n=2):
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                return (float(x[0]), float(x[0]))
            return tuple(float(v) for v in x[:n])
        return (float(x), float(x))

    def _vec(self, x): return torch.as_tensor(x, dtype=self.dtype, device=self.device)
    def _scalar(self, x): return torch.as_tensor(x, dtype=self.dtype, device=self.device)
    def _to_public(self, x): return x.detach().cpu().numpy().copy() if self.return_numpy else x.detach().clone()
    def _obs_to_public(self, obs_list):
        return [o.detach().cpu().numpy().astype(np.float32) for o in obs_list] if self.return_numpy else [o.detach().clone() for o in obs_list]
    def _rewards_to_public(self, rewards):
        return rewards.detach().cpu().numpy().astype(np.float32) if self.return_numpy else rewards.detach().clone()

    def _actions_to_tensor(self, actions):
        if torch.is_tensor(actions):
            act = actions.to(device=self.device, dtype=self.dtype)
        elif isinstance(actions, (list, tuple)):
            act = torch.stack([self._vec(a).reshape(-1) for a in actions], dim=0)
        else:
            act = self._vec(actions)
        if act.ndim == 1: act = act.unsqueeze(0)
        if act.shape != (self.n_agents, 3):
            raise ValueError(f"actions 形状应为 ({self.n_agents}, 3)，当前为 {tuple(act.shape)}")
        return act

    def _build_agent_spec_tensors(self):
        self._a_xy_max = self._vec([s["a_xy_max"] for s in self.agent_specs])
        self._a_z_max = self._vec([s["a_z_max"] for s in self.agent_specs])
        self._v_xy_max = self._vec([s["v_xy_max"] for s in self.agent_specs])
        self._v_z_max = self._vec([s["v_z_max"] for s in self.agent_specs])
        self._drag_xy = self._vec([s["drag_xy"] for s in self.agent_specs])
        self._drag_z = self._vec([s["drag_z"] for s in self.agent_specs])
        self._buoyancy_bias = self._vec([s["buoyancy_bias"] for s in self.agent_specs])
        self._sensor_range = self._vec([s["sensor_range"] for s in self.agent_specs])
        self._energy_coeff = self._vec([s["energy_coeff"] for s in self.agent_specs])
        self._progress_gain = self._vec([s["progress_gain"] for s in self.agent_specs])
        self._waypoint_bonus = self._vec([s["waypoint_bonus"] for s in self.agent_specs])
        self._detect_bonus = self._vec([s["detect_bonus"] for s in self.agent_specs])
        self._prior_strength = self._vec([self.prior_strength_search if i < self.n_search else self.prior_strength_executor for i in range(self.n_agents)])
        self._residual_scale = self._vec([self.residual_scale_search if i < self.n_search else self.residual_scale_executor for i in range(self.n_agents)])

        # Nominal copies are used by the prior controller and observation scaling.
        # Disturbed dynamics use the effective tensors sampled per episode.
        self._a_xy_max_nominal = self._a_xy_max.clone()
        self._a_z_max_nominal = self._a_z_max.clone()
        self._v_xy_max_nominal = self._v_xy_max.clone()
        self._v_z_max_nominal = self._v_z_max.clone()
        self._drag_xy_nominal = self._drag_xy.clone()
        self._drag_z_nominal = self._drag_z.clone()
        self._buoyancy_bias_nominal = self._buoyancy_bias.clone()

    def _init_robust_state(self):
        self._dyn_a_xy_max = self._a_xy_max_nominal.clone()
        self._dyn_a_z_max = self._a_z_max_nominal.clone()
        self._dyn_v_xy_max = self._v_xy_max_nominal.clone()
        self._dyn_v_z_max = self._v_z_max_nominal.clone()
        self._dyn_drag_xy = self._drag_xy_nominal.clone()
        self._dyn_drag_z = self._drag_z_nominal.clone()
        self._dyn_buoyancy_bias = self._buoyancy_bias_nominal.clone()
        self._actuator_scale_xy = torch.ones(self.n_agents, dtype=self.dtype, device=self.device)
        self._actuator_scale_z = torch.ones(self.n_agents, dtype=self.dtype, device=self.device)
        self._desired_acc_history = []
        self._prev_desired_acc_cmd = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._last_residual_acc = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._last_prior_acc_v2 = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._last_prior_term_v3 = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._last_residual_term_v3 = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._last_final_acc_cmd_v3 = torch.zeros((self.n_agents, 3), dtype=self.dtype, device=self.device)
        self._current_action_noise_std = float(self.action_noise_std)
        self._flow_phase_x = 0.0
        self._flow_phase_y = 0.0
        self._next_disturbance = None
        self.current_disturbance = nominal_disturbance()

    def _sample_uniform_float(self, lo_hi):
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-12:
            return lo
        return float(lo + (hi - lo) * torch.rand((), dtype=self.dtype, device=self.device).item())

    def set_next_disturbance(self, xi):
        """Set episode-level physical disturbance for the next reset()."""
        self._next_disturbance = None if xi is None else dict(xi)

    def get_current_disturbance(self):
        """Return a copy of the disturbance used by the current episode."""
        if hasattr(self, "current_disturbance"):
            return dict(self.current_disturbance)
        return nominal_disturbance()

    def _nominal_disturbance_dict(self):
        return nominal_disturbance()

    def _sanitize_disturbance(self, xi):
        source = self._nominal_disturbance_dict()
        if xi is not None:
            source.update(dict(xi))
        sanitized = {}
        sanitized["flow_gain"] = max(0.0, float(source.get("flow_gain", 0.18)))
        sanitized["flow_z_gain"] = float(source.get("flow_z_gain", 0.0))
        sanitized["drag_scale"] = max(1e-6, float(source.get("drag_scale", 1.0)))
        sanitized["buoyancy_bias_delta"] = float(source.get("buoyancy_bias_delta", 0.0))
        sanitized["a_max_scale"] = max(1e-6, float(source.get("a_max_scale", 1.0)))
        sanitized["v_max_scale"] = max(1e-6, float(source.get("v_max_scale", 1.0)))
        sanitized["actuator_lag"] = float(np.clip(float(source.get("actuator_lag", 0.0)), 0.0, 0.95))
        sanitized["action_delay_steps"] = max(0, int(round(float(source.get("action_delay_steps", 0)))))
        sanitized["action_noise_std"] = max(0.0, float(source.get("action_noise_std", 0.0)))
        if xi is not None:
            if "flow_phase_x" in xi:
                sanitized["flow_phase_x"] = float(xi["flow_phase_x"])
            if "flow_phase_y" in xi:
                sanitized["flow_phase_y"] = float(xi["flow_phase_y"])
        return sanitized

    def _reset_disturbance_state(self):
        self._dyn_a_xy_max = self._a_xy_max_nominal.clone()
        self._dyn_a_z_max = self._a_z_max_nominal.clone()
        self._dyn_v_xy_max = self._v_xy_max_nominal.clone()
        self._dyn_v_z_max = self._v_z_max_nominal.clone()
        self._dyn_drag_xy = self._drag_xy_nominal.clone()
        self._dyn_drag_z = self._drag_z_nominal.clone()
        self._dyn_buoyancy_bias = self._buoyancy_bias_nominal.clone()
        self._actuator_scale_xy = torch.ones(self.n_agents, dtype=self.dtype, device=self.device)
        self._actuator_scale_z = torch.ones(self.n_agents, dtype=self.dtype, device=self.device)
        self.flow_gain = 0.18
        self.flow_z_gain = 0.0
        self._flow_phase_x = 0.0
        self._flow_phase_y = 0.0
        self._current_action_noise_std = float(self.action_noise_std)
        self._desired_acc_history = []
        self._prev_desired_acc_cmd.zero_()
        self._last_residual_acc.zero_()
        self._last_prior_acc_v2.zero_()
        self._last_prior_term_v3.zero_()
        self._last_residual_term_v3.zero_()
        self._last_final_acc_cmd_v3.zero_()
        self.last_residual_norm = 0.0

    def _apply_disturbance_vector(self, xi):
        xi = self._sanitize_disturbance(xi)
        a_scale = xi["a_max_scale"]
        v_scale = xi["v_max_scale"]
        drag_scale = xi["drag_scale"]
        buoy_delta = xi["buoyancy_bias_delta"]

        self.flow_gain = xi["flow_gain"]
        self.flow_z_gain = xi["flow_z_gain"]
        self._flow_phase_x = float(xi.get("flow_phase_x", 0.0))
        self._flow_phase_y = float(xi.get("flow_phase_y", 0.0))
        self._dyn_a_xy_max = self._a_xy_max_nominal * a_scale
        self._dyn_a_z_max = self._a_z_max_nominal * a_scale
        self._dyn_v_xy_max = self._v_xy_max_nominal * v_scale
        self._dyn_v_z_max = self._v_z_max_nominal * v_scale
        self._dyn_drag_xy = self._drag_xy_nominal * drag_scale
        self._dyn_drag_z = self._drag_z_nominal * drag_scale
        self._dyn_buoyancy_bias = self._buoyancy_bias_nominal + buoy_delta
        self._actuator_scale_xy = torch.full((self.n_agents,), a_scale, dtype=self.dtype, device=self.device)
        self._actuator_scale_z = torch.full((self.n_agents,), a_scale, dtype=self.dtype, device=self.device)
        self.actuator_lag = xi["actuator_lag"]
        self.robust_action_delay_steps = xi["action_delay_steps"]
        self._current_action_noise_std = xi["action_noise_std"]
        return xi

    def _record_current_disturbance(self, xi):
        sanitized = self._sanitize_disturbance(xi)
        self.current_disturbance = {key: sanitized[key] for key in DISTURBANCE_KEYS}

    def _sample_robust_disturbance_params(self):
        self._reset_disturbance_state()

        if not self.use_robust_disturbance:
            self._record_current_disturbance(self._nominal_disturbance_dict())
            self._next_disturbance = None
            return

        if getattr(self, "_next_disturbance", None) is not None:
            xi = self._apply_disturbance_vector(self._next_disturbance)
            self._next_disturbance = None
            self._record_current_disturbance(xi)
            return

        self.flow_gain = self._sample_uniform_float(self.flow_gain_range)
        self.flow_z_gain = self._sample_uniform_float(self.flow_z_gain_range)
        if self.flow_phase_random:
            self._flow_phase_x = self._sample_uniform_float((0.0, 2.0 * np.pi))
            self._flow_phase_y = self._sample_uniform_float((0.0, 2.0 * np.pi))

        a_scale = self._sample_uniform_float(self.a_max_scale_range)
        v_scale = self._sample_uniform_float(self.v_max_scale_range)
        drag_scale = self._sample_uniform_float(self.drag_scale_range)
        buoy_delta = self._sample_uniform_float(self.buoyancy_bias_delta_range)

        # Actual plant limits/damping are disturbed, while the prior still uses nominal values.
        self._dyn_a_xy_max = self._a_xy_max_nominal * a_scale
        self._dyn_a_z_max = self._a_z_max_nominal * a_scale
        self._dyn_v_xy_max = self._v_xy_max_nominal * v_scale
        self._dyn_v_z_max = self._v_z_max_nominal * v_scale
        self._dyn_drag_xy = self._drag_xy_nominal * drag_scale
        self._dyn_drag_z = self._drag_z_nominal * drag_scale
        self._dyn_buoyancy_bias = self._buoyancy_bias_nominal + buoy_delta
        self._actuator_scale_xy = torch.full((self.n_agents,), a_scale, dtype=self.dtype, device=self.device)
        self._actuator_scale_z = torch.full((self.n_agents,), a_scale, dtype=self.dtype, device=self.device)
        if self.action_noise_std_range is not None:
            self._current_action_noise_std = self._sample_uniform_float(self.action_noise_std_range)
        self._record_current_disturbance(
            {
                "flow_gain": self.flow_gain,
                "flow_z_gain": self.flow_z_gain,
                "drag_scale": drag_scale,
                "buoyancy_bias_delta": buoy_delta,
                "a_max_scale": a_scale,
                "v_max_scale": v_scale,
                "actuator_lag": self.actuator_lag,
                "action_delay_steps": self.robust_action_delay_steps,
                "action_noise_std": self._current_action_noise_std,
            }
        )

    def _build_obstacle_tensors(self):
        self._obstacles_t = [{"center": self._vec(obs["center"]), "size": self._vec(obs["size"])} for obs in self.obstacles]
        if self._obstacles_t:
            self._obstacle_centers = torch.stack([ob["center"] for ob in self._obstacles_t], dim=0)
            self._obstacle_sizes = torch.stack([ob["size"] for ob in self._obstacles_t], dim=0)
            half = self._obstacle_sizes / 2.0
            self._obstacle_lower = self._obstacle_centers - half
            self._obstacle_upper = self._obstacle_centers + half
        else:
            self._obstacle_lower = None
            self._obstacle_upper = None

    def _sample_free_point(self, margin=0.8):
        low = self._vec([margin, margin, max(margin, self.random_z_range[0])])
        high = self._vec([
            float(self.space_size[0].item()) - margin,
            float(self.space_size[1].item()) - margin,
            min(float(self.space_size[2].item()) - margin, self.random_z_range[1]),
        ])
        while True:
            p = low + (high - low) * torch.rand(3, dtype=self.dtype, device=self.device)
            if not self.is_inside_obstacle(p):
                return p

    def _sample_initial_positions(self, min_dist=2.8):
        positions = []
        while len(positions) < self.n_agents:
            pos = self._sample_free_point(margin=1.0)
            if not positions:
                positions.append(pos); continue
            prev = torch.stack(positions, dim=0)
            if torch.all(torch.norm(prev - pos, dim=1) >= min_dist):
                positions.append(pos)
        return torch.stack(positions, dim=0)

    def _flow_at(self, pos):
        x, y = pos[..., 0], pos[..., 1]
        flow_x = self.flow_gain * torch.sin(0.25 * y + self._flow_phase_x)
        flow_y = self.flow_gain * torch.cos(0.22 * x + self._flow_phase_y)
        flow_z = torch.full_like(x, self.flow_z_gain)
        return torch.stack([flow_x, flow_y, flow_z], dim=-1)

    def _current_coverage_ratio_internal(self): return float((self.map_module.coverage > 1e-6).float().mean().item())
    def _current_search_task_min_dist(self): return float(torch.min(torch.norm(self._agent_pos[: self.n_search] - self._task_target.unsqueeze(0), dim=1)).item())

    def _set_pse_planner_context(self):
        if not bool(getattr(self, "use_pse_planner", False)):
            return
        effective_weight, schedule_factor = self._compute_pse_exec_cost_weight()
        if hasattr(self.map_module, "set_runtime_context"):
            self.map_module.set_runtime_context(
                executor_pos=self._agent_pos[self.executor_idx],
                executor_wait_point=getattr(self, "_executor_wait_point", None),
                use_belief=self.pse_use_belief,
                use_exec_cost=self.pse_use_exec_cost,
                use_standby=self.pse_use_standby,
                pse_exec_cost_weight_effective=effective_weight,
                pse_exec_cost_schedule_factor=schedule_factor,
            )

    def _compute_pse_exec_cost_weight(self):
        base_weight = float(getattr(self, "pse_exec_cost_weight", 0.0))
        if not bool(getattr(self, "pse_use_exec_cost", True)):
            self.last_pse_exec_cost_weight_effective = 0.0
            self.last_pse_exec_cost_schedule_factor = 0.0
            return 0.0, 0.0
        if not bool(getattr(self, "pse_use_exec_cost_schedule", False)):
            self.last_pse_exec_cost_weight_effective = base_weight
            self.last_pse_exec_cost_schedule_factor = 1.0
            return base_weight, 1.0

        current_step = float(max(0, int(getattr(self, "step_count", 0))))
        warmup = float(max(1, int(getattr(self, "pse_exec_cost_schedule_warmup_steps", 1))))
        time_factor = float(np.clip(current_step / warmup, 0.0, 1.0))
        entropy = float(np.nan_to_num(getattr(self, "last_belief_entropy", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        entropy_low = float(getattr(self, "pse_exec_cost_entropy_low", 0.45))
        entropy_high = float(getattr(self, "pse_exec_cost_entropy_high", 0.95))
        denom = max(entropy_high - entropy_low, 1e-6)
        entropy_factor = float(np.clip((entropy_high - entropy) / denom, 0.0, 1.0))
        schedule_factor = max(time_factor, entropy_factor)
        w_min = float(getattr(self, "pse_exec_cost_weight_min", base_weight))
        w_max = float(getattr(self, "pse_exec_cost_weight_max", base_weight))
        if w_max < w_min:
            w_min, w_max = w_max, w_min
        effective = w_min + schedule_factor * (w_max - w_min)
        effective = float(np.nan_to_num(effective, nan=base_weight, posinf=w_max, neginf=w_min))
        self.last_pse_exec_cost_weight_effective = effective
        self.last_pse_exec_cost_schedule_factor = float(schedule_factor)
        return effective, float(schedule_factor)

    def _update_pse_belief(self, force_detection: bool = False):
        if not (bool(getattr(self, "use_pse_planner", False)) and bool(getattr(self, "pse_use_belief", True))):
            return
        if not hasattr(self.map_module, "belief_entropy"):
            return
        try:
            if bool(force_detection) or bool(getattr(self, "task_found", False)):
                if hasattr(self.map_module, "update_belief_detection"):
                    self.map_module.update_belief_detection(self._task_target)
            elif hasattr(self.map_module, "update_belief_negative"):
                ranges = self._sensor_range[: self.n_search] if hasattr(self, "_sensor_range") else self.detect_proximity_radius
                self.map_module.update_belief_negative(self._agent_pos[: self.n_search], sensor_ranges=ranges)
            self.last_belief_entropy = float(self.map_module.belief_entropy().item())
        except Exception:
            self.last_belief_entropy = float(getattr(self, "last_belief_entropy", 0.0))

    def _update_pse_executor_standby(self, force: bool = False):
        self.last_pse_standby_update_used = 0.0
        self.last_pse_lazy_standby_active = 0.0
        self.last_pse_standby_update_allowed = 1.0
        self.last_pse_standby_update_skipped_by_lazy_gate = 0.0
        interval = max(1, int(getattr(self, "planner_step_update_interval", 1)))
        lazy_active = bool(getattr(self, "pse_lazy_standby", False)) and not bool(getattr(self, "task_found", False))
        if lazy_active:
            interval = max(1, int(getattr(self, "pse_standby_update_interval_lazy", interval)))
            self.last_pse_lazy_standby_active = 1.0
        self.last_pse_standby_update_interval = float(interval)
        if not (
            bool(getattr(self, "use_pse_planner", False))
            and bool(getattr(self, "pse_use_standby", True))
            and not bool(getattr(self, "task_found", False))
        ):
            return
        if not hasattr(self.map_module, "plan_executor_standby"):
            return
        current_step = int(getattr(self, "step_count", 0))
        if current_step == int(getattr(self, "_last_pse_standby_update_step", -1)):
            return
        if (not bool(force)) and (current_step % interval != 0):
            return
        if lazy_active and not bool(force):
            entropy = float(np.nan_to_num(getattr(self, "last_belief_entropy", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
            prev_standby = getattr(self, "_executor_wait_point", None)
            has_valid_standby = prev_standby is not None and bool(torch.isfinite(prev_standby).all().item())
            allowed = (
                current_step >= int(getattr(self, "pse_standby_min_step", 0))
                or entropy <= float(getattr(self, "pse_standby_entropy_gate", 1.0))
                or not has_valid_standby
            )
            self.last_pse_standby_update_allowed = 1.0 if allowed else 0.0
            if not allowed:
                self.last_pse_standby_update_skipped_by_lazy_gate = 1.0
                return
        self._set_pse_planner_context()
        try:
            standby = self.map_module.plan_executor_standby(
                self._agent_pos[self.executor_idx],
                prev_standby=getattr(self, "_executor_wait_point", None),
                move_weight=getattr(self, "pse_standby_move_weight_lazy", None) if lazy_active else None,
                hysteresis_weight=getattr(self, "pse_standby_hysteresis_weight_lazy", None) if lazy_active else None,
            )
            standby = torch.as_tensor(standby, dtype=self.dtype, device=self.device).reshape(3)
            standby = torch.clamp(standby, min=self._sampling_margin_min, max=self.space_size - 0.5)
            if self.is_inside_obstacle(standby):
                return
            self._executor_wait_point.copy_(standby)
            self._last_pse_standby_update_step = current_step
            self.last_pse_standby_update_used = 1.0
            self.executor_wait_held = False
            self.current_target_arrived[self.executor_idx] = False
        except Exception:
            return

    def _sync_pse_diagnostics(self):
        if not bool(getattr(self, "use_pse_planner", False)):
            return
        planner = self.map_module
        self.last_belief_entropy = float(getattr(planner, "last_belief_entropy", getattr(self, "last_belief_entropy", 0.0)))
        self.last_exec_response_cost = float(getattr(planner, "last_exec_response_cost", 0.0))
        self.last_search_score_mean = float(getattr(planner, "last_search_score_mean", 0.0))
        self.last_pse_claim_overlap = float(getattr(planner, "last_claim_overlap", 0.0))
        self.last_pse_exec_cost_weight_effective = float(getattr(planner, "last_pse_exec_cost_weight_effective", getattr(self, "last_pse_exec_cost_weight_effective", self.pse_exec_cost_weight)))
        self.last_pse_exec_cost_schedule_factor = float(getattr(planner, "last_pse_exec_cost_schedule_factor", getattr(self, "last_pse_exec_cost_schedule_factor", 0.0)))
        standby = getattr(planner, "last_executor_standby", None)
        if hasattr(self, "_task_target"):
            if standby is not None:
                standby_t = torch.as_tensor(standby, dtype=self.dtype, device=self.device).reshape(3)
            elif hasattr(self, "_executor_wait_point"):
                standby_t = self._executor_wait_point
            else:
                standby_t = self._agent_pos[self.executor_idx]
            self.last_standby_to_target_dist = float(torch.norm(standby_t - self._task_target).item())
        if bool(getattr(self, "mission_complete", False)) and getattr(self, "_found_step", None) is not None:
            self.last_success_step_minus_found_step = float(max(0, int(self.step_count) - int(self._found_step)))

    def _reset_target_belief_memory(self):
        self._target_belief_valid = False
        self._target_belief_pos = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._target_belief_confidence = 0.0
        self._target_belief_age_steps = 0
        self._target_belief_source = -1
        self._target_belief_last_update_step = -1
        self._target_belief_executor_used = False
        self._target_belief_soft_assign_active = False
        self._target_belief_soft_executor_assigned = False
        self.last_target_belief_valid = 0.0
        self.last_target_belief_age = 0.0
        self.last_target_belief_confidence = 0.0
        self.last_target_belief_used = 0.0
        self.last_target_belief_soft_assign = 0.0
        self.last_target_belief_update = 0.0
        self.last_target_belief_error = 0.0

    def _record_target_belief(self, target_pos, source_id=-1, confidence=1.0, reason=""):
        if not bool(getattr(self, "use_target_belief_memory", False)):
            return False
        if target_pos is None:
            return False
        pos = target_pos.detach().to(device=self.device, dtype=self.dtype).reshape(3)
        if torch.any(torch.isnan(pos)):
            return False
        pos = torch.clamp(pos, min=self._lower_bound, max=self.space_size)

        old_valid = bool(getattr(self, "_target_belief_valid", False))
        old_conf = float(getattr(self, "_target_belief_confidence", 0.0))
        new_conf = float(np.clip(confidence, 0.0, 1.0))

        if old_valid and new_conf + 1e-6 < old_conf:
            return False

        self._target_belief_valid = True
        self._target_belief_pos.copy_(pos)
        self._target_belief_confidence = new_conf
        self._target_belief_age_steps = 0
        self._target_belief_source = int(source_id)
        self._target_belief_last_update_step = int(self.step_count)
        self.last_target_belief_update = 1.0

        if hasattr(self, "_task_target"):
            err = torch.linalg.vector_norm(self._target_belief_pos - self._task_target)
            self.last_target_belief_error = float(err.item())

        return True

    def _advance_target_belief_memory(self):
        if not bool(getattr(self, "use_target_belief_memory", False)):
            return
        if int(getattr(self, "_target_belief_last_update_step", -1)) != int(getattr(self, "step_count", 0)):
            self.last_target_belief_update = 0.0
        self.last_target_belief_used = 0.0
        self.last_target_belief_soft_assign = 0.0
        self._target_belief_executor_used = False
        self._target_belief_soft_assign_active = False

        if not bool(getattr(self, "_target_belief_valid", False)):
            self.last_target_belief_valid = 0.0
            self.last_target_belief_age = 0.0
            self.last_target_belief_confidence = 0.0
            return

        self._target_belief_age_steps += 1
        self._target_belief_confidence = max(
            0.0,
            float(self._target_belief_confidence) - float(self.target_belief_confidence_decay),
        )

        if (
            self._target_belief_age_steps > int(self.target_belief_max_age_steps)
            or self._target_belief_confidence < float(self.target_belief_min_confidence)
        ):
            self._target_belief_valid = False
            self._target_belief_confidence = 0.0

        self.last_target_belief_valid = 1.0 if self._target_belief_valid else 0.0
        self.last_target_belief_age = float(self._target_belief_age_steps)
        self.last_target_belief_confidence = float(self._target_belief_confidence)
        if self._target_belief_valid and hasattr(self, "_task_target"):
            err = torch.linalg.vector_norm(self._target_belief_pos - self._task_target)
            self.last_target_belief_error = float(err.item())

    def reset(self):
        self.step_count = 0
        self._sample_robust_disturbance_params()
        self._collision_flags.zero_(); self._agent_vel.zero_(); self._agent_acc.zero_(); self._prev_acc.zero_()
        self._agent_pos = self._sample_initial_positions(min_dist=2.8)
        # Keep planner grid/depth preferences synchronized with curriculum z_range.
        # Directly assigning map_module.z_range is not enough because the planner
        # caches 3-D grid centers used for waypoint sampling.
        if hasattr(self.map_module, "set_z_range"):
            self.map_module.set_z_range(self.random_z_range, rebuild_grid=True)
        else:
            self.map_module.z_range = self.random_z_range
        # 上层 map 不再直接读取所有搜索体当前位置；初始航点可理解为任务开始前预装。
        self.map_module.reset(None, self.obstacles)
        if self.use_pse_planner and hasattr(self.map_module, "reset_belief_map"):
            self.map_module.reset_belief_map()
        self.last_belief_entropy = float(getattr(self.map_module, "last_belief_entropy", 0.0))
        self.last_standby_to_target_dist = 0.0
        self.last_exec_response_cost = 0.0
        self.last_search_score_mean = 0.0
        self.last_pse_claim_overlap = 0.0
        self.last_pse_fallback_fast_path_used = 0.0
        self.last_pse_fallback_candidate_count = 0.0
        self.last_pse_exec_cost_weight_effective = 0.0 if not bool(getattr(self, "pse_use_exec_cost", True)) else float(getattr(self, "pse_exec_cost_weight", 0.0))
        self.last_pse_exec_cost_schedule_factor = 0.0
        self.last_pse_lazy_standby_active = 0.0
        self.last_pse_standby_update_allowed = 1.0
        self.last_pse_standby_update_skipped_by_lazy_gate = 0.0
        self.last_pse_standby_update_used = 0.0
        self.last_pse_standby_update_interval = float(max(1, int(self.planner_step_update_interval)))
        self._last_pse_standby_update_step = -1
        self.last_success_step_minus_found_step = float("nan")
        self.task_found = False; self.finder_idx = -1; self.mission_complete = False
        self.search_stage_complete = False; self.executor_target_assigned = False; self.executor_wait_held = False
        self.last_direct_task_assignment = 0.0
        self._found_event = False; self._mission_complete_event = False; self._executor_wait_hold_event = False
        self._target_assigned_bonus_given_v2 = False
        self.prev_exec_task_dist_v2 = None
        self.executor_hold_counter_v2 = 0
        self._reset_robust_residual_reward_diagnostics_v2()
        self._reset_point_control_reward_diagnostics_v3()
        self._agent_task_known.zero_(); self._agent_task_est.zero_(); self._upper_task_known = False; self._upper_task_est.zero_()
        self._pending_upper_upload.zero_(); self._waiting_upper_waypoint.zero_(); self._upper_wait_steps.zero_(); self._using_local_fallback.zero_()
        self._upper_uplink_queue = []; self._upper_downlink_queue = []
        self._found_step = None; self._executor_received_target_step = None
        self.last_handoff_delay = float("nan")
        self.last_upper_comm_density = 0.0; self.last_upper_uplink_success_rate = 0.0; self.last_upper_downlink_success_rate = 0.0
        self.last_upper_uplink_success_rate_valid = float("nan"); self.last_upper_downlink_success_rate_valid = float("nan")
        self.last_upper_uplink_has_attempt = 0.0; self.last_upper_downlink_has_attempt = 0.0
        self.last_upper_up_attempts = 0.0; self.last_upper_down_attempts = 0.0
        self.last_upper_up_successes = 0.0; self.last_upper_down_successes = 0.0
        self.last_upper_total_attempts = 0.0; self.last_upper_total_successes = 0.0; self.last_upper_has_attempt = 0.0
        self.last_upper_avg_delay_steps = 0.0; self.last_upper_map_uploads = 0; self.last_upper_waypoint_downlinks = 0; self.last_upper_target_downlinks = 0
        self.last_upper_comm_energy = 0.0; self.last_upper_msg_age = 0.0; self.last_upper_good_state_ratio = 1.0
        self.last_upper_belief_age = 0.0; self.last_local_fallback_ratio = 0.0
        self.last_island_duration = 0.0; self.last_island_agent_ratio = 0.0; self.last_island_count = 0.0
        self.last_reconnect_count = 0.0; self.last_reconnect_success_count = 0.0; self.last_avg_reconnect_time = 0.0; self.last_reconnect_lane_score = 0.0
        self.last_upper_role_topology_density = 0.0; self.last_upper_role_active_density = 0.0; self.last_upper_role_physical_density = 0.0
        self.last_lower_avg_delay_steps = 0.0; self.last_lower_msg_age = 0.0; self.last_lower_comm_energy = 0.0
        self.last_lower_good_state_ratio = 1.0; self.last_comm_energy_total = 0.0; self.last_comm_energy.zero_()
        self.last_edge_weight = 0.0; self.last_link_loss_prob = 0.0; self.last_link_snr_norm = 0.0
        self.last_link_bandwidth_bps = 0.0; self.last_comm_noise_mean = 0.0; self.last_depth_gap_mean = 0.0; self.last_flow_diff_mean = 0.0
        self.last_sem_target_msg_count = 0.0; self.last_sem_map_msg_count = 0.0; self.last_sem_waypoint_req_count = 0.0
        self.last_sem_handover_msg_count = 0.0; self.last_sem_heartbeat_count = 0.0; self.last_sem_risk_alert_count = 0.0
        self.last_sem_critical_count = 0.0; self.last_sem_payload_bits = 0.0
        self.last_sem_selected_count = 0.0; self.last_sem_critical_selected_count = 0.0
        self.last_sem_voi_score = 0.0; self.last_sem_dropped_by_budget = 0.0; self.last_sem_critical_dropped_by_budget = 0.0
        self.last_sem_voi_stress = 0.0; self.last_sem_voi_penalty_scale = 1.0
        self.last_sem_voi_adaptive_mix = 0.0; self.last_sem_voi_type_diversity_used = 0.0
        self.last_sem_voi_comm_pressure = 0.0; self.last_sem_voi_task_stage_id = 0.0
        self.last_sem_voi_stage_diversity_used = 0.0; self.last_sem_voi_upper_success_context = 1.0
        self.last_sem_voi_island_context = 0.0; self.last_sem_voi_critical_drop_context = 0.0
        self.last_sem_voi_search_diversity_disabled = 0.0
        self.last_sem_critical_delivery_rate = 0.0; self.last_sem_critical_drop_rate = 0.0
        self.last_sem_sent_packet_count = 0.0; self.last_sem_delivered_packet_count = 0.0; self.last_sem_arrived_packet_count = 0.0; self.last_sem_aggregated_packet_count = 0.0
        self.last_sem_multi_packet_edge_count = 0.0; self.last_sem_avg_packets_per_active_edge = 0.0
        self.last_sem_packet_delivery_rate = 0.0; self.last_sem_packet_drop_rate = 0.0
        self.last_sem_critical_packet_delivery_rate = 0.0; self.last_sem_critical_packet_drop_rate = 0.0
        self.last_role_topology_density = 1.0; self.last_role_active_density = 1.0; self.last_role_physical_density = 1.0
        self.last_role_pruned_ratio = 0.0; self.last_role_critical_link_density = 0.0; self.last_direct_handover_ready = 0.0
        self.last_direct_handover_bypass_enabled = 0.0
        self.last_direct_handover_bypass_ready = 0.0
        self.last_direct_handover_bypass_gate_pending = 0.0
        self.last_direct_handover_bypass_gate_pressure = 0.0
        self.last_adaptive_critical_filter_active = 0.0
        self.last_adaptive_critical_filter_score = 1.0
        self.last_adaptive_critical_filter_critical_boost = 0.0
        self.last_adaptive_critical_filter_ordinary_penalty = 0.0
        self.last_reconnect_voi_boost_active = 0.0
        self.last_guarded_adaptive_strength = 0.0
        self.last_guarded_medium_risk_active = 0.0
        self.last_guarded_extreme_guard_active = 0.0
        self.last_guarded_context_preserve_active = 0.0
        self.last_guarded_bypass_cooldown_active = 0.0
        self.last_direct_handover_attempts = 0.0; self.last_direct_handover_success = 0.0
        self.last_direct_handover_episode_attempts = 0.0; self.last_direct_handover_episode_success = 0.0; self.last_direct_handover_count = 0.0
        self.last_handover_success_rate = 0.0; self.last_handover_episode_success_rate = 0.0
        self._reset_residual_prior_diagnostics()
        self._role_topology_mask = self._non_diag_bool.to(dtype=self.dtype)
        self._role_score_matrix = self._role_topology_mask.clone()
        self._direct_handover_bypass_mask = torch.zeros((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        self._direct_handover_bypass_probs = torch.zeros((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        if hasattr(self, "_guarded_bypass_cooldown"):
            self._guarded_bypass_cooldown.zero_()
        if hasattr(self, "_guarded_bypass_window_count"):
            self._guarded_bypass_window_count.zero_()
        if hasattr(self, "_guarded_bypass_window_start_step"):
            self._guarded_bypass_window_start_step.fill_(int(self.step_count))
        if hasattr(self, "_guarded_bypass_last_trigger_step"):
            self._guarded_bypass_last_trigger_step.fill_(-1)
        self.last_pred_usage_ratio = 0.0; self.last_pred_generated_ratio = 0.0; self.last_pred_obs_ratio = 0.0; self.last_pred_fallback_ratio = 0.0
        self.last_pred_uncertainty = 0.0; self.last_pred_sigma = 0.0; self.last_pred_error = 0.0; self.last_repair_gain = 0.0
        self.last_avg_reliability = 1.0; self.last_conflict_score = 0.0; self.last_belief_disagreement = 0.0; self.last_quarantine_count = 0.0
        self.last_quarantine_active_count = 0.0; self.last_quarantine_release_count = 0.0; self.last_quarantine_expire_count = 0.0
        self.last_map_update_reliability = 1.0; self.last_prediction_reliability = 0.0; self.last_target_fusion_reliability = 0.0
        self._edge_features.zero_(); self._edge_weights.zero_(); self.comm_graph = {}
        self._upper_comm_attempts = {"up": 0, "down": 0, "up_success": 0, "down_success": 0, "reachable": 0}
        self._upper_delay_sum = 0.0; self._upper_delay_count = 0
        self._upper_last_contact_step.zero_(); self._upper_island_active.zero_(); self._upper_reconnect_active.zero_(); self._upper_reconnect_start_step.fill_(-1)
        self._upper_belief_pos.zero_(); self._upper_belief_vel.zero_(); self._upper_belief_step.fill_(-10_000); self._upper_belief_valid.zero_()
        self.waypoint_reached_counts.zero_(); self.hold_success_counts.zero_(); self.total_waypoints_per_agent.zero_()
        self.agent_finished.zero_(); self.just_reached_waypoint.zero_(); self.just_held_target.zero_()
        self.hold_counters.zero_(); self.current_target_arrived.zero_()
        self._task_target.copy_(self._sample_free_point(margin=1.0))
        self._reset_target_belief_memory()
        search_center = self._agent_pos[: self.n_search].mean(dim=0)
        self._executor_wait_point.copy_(torch.clamp(search_center, min=self._sampling_margin_min, max=self.space_size - 0.5))
        if self.is_inside_obstacle(self._executor_wait_point):
            self._executor_wait_point.copy_(self._sample_free_point(margin=1.0))
        self._set_pse_planner_context()
        self._update_pse_executor_standby(force=True)
        self._set_pse_planner_context()
        self._search_waypoints = self.map_module.initial_search_targets(self._agent_pos[: self.n_search])
        self._nav_targets[: self.n_search] = self._search_waypoints
        self._nav_targets[self.executor_idx] = self._executor_wait_point
        self._targets.copy_(self._nav_targets)
        self.total_waypoints_per_agent[: self.n_search] = 1
        self.total_waypoints_per_agent[self.executor_idx] = 1
        self._prev_nav_distances = self._compute_nav_distances()
        self._prev_coverage_ratio = self._current_coverage_ratio_internal()
        self._prev_search_task_min_dist = self._current_search_task_min_dist()
        self._reset_communication_state()
        return self._obs_to_public(self._get_obs())

    def _update_nav_targets(self):
        if not self.task_found:
            self._nav_targets[: self.n_search] = self._search_waypoints
            self._nav_targets[self.executor_idx] = self._executor_wait_point
        else:
            # 搜索阶段结束后搜索体停止；执行体只有收到上层目标下发后才切换到真实目标。
            self._nav_targets[: self.n_search] = self._agent_pos[: self.n_search]
            if self.executor_target_assigned:
                self._nav_targets[self.executor_idx] = self._agent_task_est[self.executor_idx]
                if bool(getattr(self, "_target_belief_soft_executor_assigned", False)):
                    self._target_belief_executor_used = True
                    self.last_target_belief_used = 1.0
            elif (
                bool(getattr(self, "use_target_belief_memory", False))
                and bool(getattr(self, "target_belief_executor_fallback", False))
                and bool(getattr(self, "_target_belief_valid", False))
                and float(getattr(self, "_target_belief_confidence", 0.0)) >= float(getattr(self, "target_belief_min_confidence", 0.25))
            ):
                self._nav_targets[self.executor_idx] = self._target_belief_pos
                self._target_belief_executor_used = True
                self.last_target_belief_used = 1.0

                if bool(getattr(self, "target_belief_mark_executor_soft_known", True)):
                    self._agent_task_est[self.executor_idx].copy_(self._target_belief_pos)
                    self._agent_task_known[self.executor_idx] = True
                    self._target_belief_soft_assign_active = True
                    self.last_target_belief_soft_assign = 1.0

                    if not self.executor_target_assigned:
                        self.executor_target_assigned = True
                        self._target_belief_soft_executor_assigned = True
                        self.current_target_arrived[self.executor_idx] = False
                        if self._found_step is not None and getattr(self, "_executor_received_target_step", None) is None:
                            self._executor_received_target_step = int(self.step_count)
                            self.last_handoff_delay = float(self._executor_received_target_step - self._found_step)
            else:
                self._nav_targets[self.executor_idx] = self._executor_wait_point
        self._targets.copy_(self._nav_targets)

    def _assign_executor_target_direct(self, target_pos=None, reason="direct_no_upper_comm"):
        try:
            exec_i = int(self.executor_idx)
            source = self._task_target if target_pos is None else target_pos
            target_est = torch.as_tensor(source, dtype=self.dtype, device=self.device).reshape(-1)
            if target_est.numel() != 3:
                return False
            target_est = torch.clamp(target_est[:3].clone(), min=self._lower_bound, max=self.space_size)
            if torch.any(torch.isnan(target_est)):
                return False

            was_assigned = bool(getattr(self, "executor_target_assigned", False))
            self._agent_task_known[exec_i] = True
            self._agent_task_est[exec_i].copy_(target_est)
            self.executor_target_assigned = True
            self.current_target_arrived[exec_i] = False
            self.hold_counters[exec_i] = 0
            self._target_belief_soft_executor_assigned = False
            self._nav_targets[exec_i].copy_(target_est)
            self._targets[exec_i].copy_(target_est)
            self.last_direct_task_assignment = 1.0

            if not was_assigned:
                self.total_waypoints_per_agent[exec_i] += 1
                self._executor_received_target_step = int(self.step_count)
                if self._found_step is not None:
                    self.last_handoff_delay = float(self._executor_received_target_step - self._found_step)
            return True
        except Exception:
            return False

    def _maybe_detect_task(self):
        if self.task_found:
            return
        dists = torch.norm(self._agent_pos[: self.n_search] - self._task_target.unsqueeze(0), dim=1)
        detect_mask = dists <= (self._sensor_range[: self.n_search] + self.detect_eps_bias)
        if torch.any(detect_mask):
            self.task_found = True
            self.finder_idx = int(torch.argmax(detect_mask.to(torch.int32)).item())
            self.search_stage_complete = True
            self.agent_finished[: self.n_search] = True
            self._found_event = True
            self._found_step = int(self.step_count)
            self._agent_task_known[self.finder_idx] = True
            self._agent_task_est[self.finder_idx].copy_(self._task_target)
            self._pending_upper_upload[self.finder_idx] = True
            if not bool(getattr(self, "use_upper_comm", False)):
                self._assign_executor_target_direct(
                    self._task_target,
                    reason="direct_no_upper_comm_after_detection",
                )
            if self.use_target_belief_memory:
                self._record_target_belief(
                    self._task_target,
                    source_id=self.finder_idx,
                    confidence=1.0,
                    reason="local_detect",
                )
            self._update_pse_belief(force_detection=True)
            self._sync_pse_diagnostics()

    def _hold_steps_for_agent(self, agent_id): return self.search_hold_steps if int(agent_id) < self.n_search else self.executor_hold_steps

    def _get_voi_task_stage(self) -> str:
        return "search" if not bool(getattr(self, "task_found", False)) else "execute"

    def _get_voi_comm_context(self) -> dict:
        return {"upper_success_rate": 1.0, "critical_drop_rate": 0.0, "island_ratio": 0.0}

    def _adaptive_critical_filter_context(self) -> dict:
        return {"handover_pending": 0.0, "comm_pressure": 0.0, "island_ratio": 0.0, "upper_success_context": 1.0}

    def _normalize_guarded_effective_diagnostics(self):
        self.last_guarded_adaptive_strength = 0.0
        self.last_guarded_medium_risk_active = 0.0
        self.last_guarded_extreme_guard_active = 0.0
        self.last_guarded_context_preserve_active = 0.0
        self.last_guarded_bypass_cooldown_active = 0.0

    def _make_comm_messages(self):
        """Build the basic fixed-size status message used by Chapter 3/4."""
        self._invalidate_message_cache()
        msgs = []
        zero_task = self._zero3
        for i in range(self.n_agents):
            known_task = self._agent_task_known[i].to(self.dtype).reshape(1)
            task_est = self._agent_task_est[i] if bool(self._agent_task_known[i].item()) else zero_task
            total_t = max(1, int(self.total_waypoints_per_agent[i].item()))
            progress_ratio = (self.waypoint_reached_counts[i].to(self.dtype) / float(total_t)).reshape(1)
            finished_flag = self.agent_finished[i].to(self.dtype).reshape(1)
            hold_ratio = torch.clamp(
                self.hold_counters[i].to(self.dtype) / max(1.0, float(self._hold_steps_for_agent(i))),
                0.0,
                1.0,
            ).reshape(1)
            msg_i = torch.cat([
                self._agent_pos[i], self._agent_vel[i], self._nav_targets[i],
                known_task, task_est, progress_ratio, finished_flag, hold_ratio,
                self.role_onehots[i]
            ]).to(dtype=self.dtype)
            msgs.append(msg_i)
        self._comm_messages = torch.stack(msgs, dim=0)
        self.last_sem_target_msg_count = 0.0
        self.last_sem_map_msg_count = 0.0
        self.last_sem_waypoint_req_count = 0.0
        self.last_sem_handover_msg_count = 0.0
        self.last_sem_heartbeat_count = float(self.n_agents)
        self.last_sem_risk_alert_count = 0.0
        self.last_sem_critical_count = 0.0
        self.last_sem_selected_count = 0.0
        self.last_sem_critical_selected_count = 0.0
        self.last_sem_payload_bits = float(self.lower_payload_bits * self.n_agents)
        self.last_sem_voi_score = 0.0
        self.last_sem_voi_stress = 0.0
        self.last_sem_voi_penalty_scale = 1.0
        self.last_sem_voi_adaptive_mix = 0.0
        self.last_sem_voi_type_diversity_used = 0.0
        self.last_sem_voi_comm_pressure = 0.0
        self.last_sem_voi_task_stage_id = 0.0
        self.last_sem_voi_stage_diversity_used = 0.0
        self.last_sem_voi_upper_success_context = 1.0
        self.last_sem_voi_island_context = 0.0
        self.last_sem_voi_critical_drop_context = 0.0
        self.last_sem_voi_search_diversity_disabled = 0.0
        self.last_sem_dropped_by_budget = 0.0
        self.last_sem_critical_dropped_by_budget = 0.0
        self.last_sem_critical_drop_rate = 0.0
        return self._comm_messages

    def _link_success_prob(self, dist, reliable_range, max_range, base_loss, attenuation, payload_bits=None):
        dist_t = dist if torch.is_tensor(dist) else self._scalar(dist)
        p_reliable = 1.0 - float(np.clip(base_loss, 0.0, 1.0))
        p = torch.zeros_like(dist_t, dtype=self.dtype, device=self.device)
        reliable_mask = dist_t <= float(reliable_range)
        fade_mask = (dist_t > float(reliable_range)) & (dist_t <= float(max_range))
        p = torch.where(reliable_mask, torch.full_like(p, p_reliable), p)
        fade_p = p_reliable * torch.exp(-float(attenuation) * (dist_t - float(reliable_range)))
        p = torch.where(fade_mask, fade_p, p)

        # Longer packets have larger packet-error probability. This is intentionally
        # a lightweight link-layer abstraction, not a waveform-level PHY simulator.
        if payload_bits is not None and self.payload_loss_scale > 0.0:
            bits = float(max(1, payload_bits))
            packet_factor = float(np.exp(-self.payload_loss_scale * max(0.0, bits - 80.0) / 160.0))
            p = p * packet_factor
        return torch.clamp(p, 0.0, 1.0)

    def _sample_next_burst_state(self, current_state):
        state = int(current_state)
        probs = self.burst_transition_matrix[state]
        return int(torch.multinomial(probs / probs.sum().clamp_min(self.eps), 1).item())

    def _advance_lower_link_states(self):
        if not self.use_burst_comm:
            self._lower_link_state.fill_(0)
            return
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    self._lower_link_state[i, j] = 0
                else:
                    self._lower_link_state[i, j] = self._sample_next_burst_state(self._lower_link_state[i, j].item())

    def _advance_upper_link_states(self):
        if not self.use_burst_comm:
            self._upper_link_state.fill_(0)
            return
        for i in range(self.n_agents):
            for d in range(2):
                self._upper_link_state[i, d] = self._sample_next_burst_state(self._upper_link_state[i, d].item())

    def _payload_tx_time(self, payload_bits=None, rate_bps=None):
        bits = float(self.lower_payload_bits if payload_bits is None else max(1, payload_bits))
        rate = float(self.lower_rate_bps if rate_bps is None else max(1.0, rate_bps))
        return bits / rate

    def _lower_delay_steps(self, dist, payload_bits=None, rate_bps=None):
        d = float(dist.item() if torch.is_tensor(dist) else dist)
        prop_delay = d / self.lower_effective_sound_speed
        tx_delay = self._payload_tx_time(payload_bits, rate_bps)
        delay = int(np.ceil((prop_delay + tx_delay) / max(self.dt, self.eps))) + self.lower_extra_delay_steps
        return int(np.clip(delay, 0, self.lower_max_delay_steps))

    def _upper_distance_to_agent(self, agent_id):
        return torch.norm(self._agent_pos[int(agent_id)] - self.upper_node_pos)

    def _upper_delay_steps(self, dist, payload_bits=None, rate_bps=None):
        d = float(dist.item() if torch.is_tensor(dist) else dist)
        bits = float(self.upper_state_payload_bits if payload_bits is None else max(1, payload_bits))
        rate = float(self.upper_rate_bps if rate_bps is None else max(1.0, rate_bps))
        prop_delay = d / self.upper_effective_sound_speed
        tx_delay = bits / rate
        delay = int(np.ceil((prop_delay + tx_delay) / max(self.dt, self.eps))) + self.upper_extra_delay_steps
        return int(max(0, delay))

    def _add_comm_energy(self, sender_id=None, receiver_id=None, payload_bits=None, rate_bps=None, tx=True, rx=True):
        if not self.use_comm_energy:
            return 0.0
        bits = float(self.lower_payload_bits if payload_bits is None else max(1, payload_bits))
        rate = float(self.lower_rate_bps if rate_bps is None else max(1.0, rate_bps))
        tx_time = bits / rate
        energy_total = 0.0
        if tx and sender_id is not None and 0 <= int(sender_id) < self.n_agents:
            e = self.comm_tx_power * tx_time + self.comm_idle_power * self.dt
            self.last_comm_energy[int(sender_id)] += float(e)
            energy_total += float(e)
        if rx and receiver_id is not None and 0 <= int(receiver_id) < self.n_agents:
            e = self.comm_rx_power * tx_time + self.comm_idle_power * self.dt
            self.last_comm_energy[int(receiver_id)] += float(e)
            energy_total += float(e)
        return energy_total

    def _reset_lower_delivery_state(self, current_msgs):
        self._lower_link_queues = [[[] for _ in range(self.n_agents)] for _ in range(self.n_agents)]
        self._lower_delivered_messages.zero_()
        self._lower_delivered_steps.fill_(-10_000)
        self._lower_delivered_quality.zero_()
        self._lower_delivered_state.fill_(2)
        self._lower_delivered_mask.zero_()
        mode = str(getattr(self, "lower_initial_sync_mode", "none"))
        sync_edges = 0
        if mode == "full":
            for receiver in range(self.n_agents):
                for sender in range(self.n_agents):
                    if receiver == sender:
                        continue
                    self._lower_delivered_messages[receiver, sender] = current_msgs[sender]
                    self._lower_delivered_steps[receiver, sender] = int(self.step_count)
                    self._lower_delivered_quality[receiver, sender] = 1.0
                    self._lower_delivered_state[receiver, sender] = 0
                    self._lower_delivered_mask[receiver, sender] = True
                    sync_edges += 1
        elif mode == "probabilistic":
            for receiver in range(self.n_agents):
                for sender in range(self.n_agents):
                    if receiver == sender:
                        continue
                    dist = torch.norm(self._agent_pos[sender] - self._agent_pos[receiver])
                    p = self._link_success_prob(
                        dist,
                        reliable_range=self.comm_reliable_range,
                        max_range=self.comm_max_range,
                        base_loss=self.comm_loss_prob,
                        attenuation=self.comm_attenuation,
                        payload_bits=self.lower_payload_bits,
                    )
                    if torch.rand((), dtype=self.dtype, device=self.device).item() >= float(p.item()):
                        continue
                    self._lower_delivered_messages[receiver, sender] = current_msgs[sender]
                    self._lower_delivered_steps[receiver, sender] = int(self.step_count)
                    self._lower_delivered_quality[receiver, sender] = float(torch.clamp(p, 0.0, 1.0).item())
                    self._lower_delivered_state[receiver, sender] = int(self.burst_initial_state)
                    self._lower_delivered_mask[receiver, sender] = True
                    sync_edges += 1
        elif mode != "none":
            raise ValueError(f"Unknown lower_initial_sync_mode={mode}")
        self.last_lower_initial_sync_edges = float(sync_edges)
        self.last_lower_initial_sync_mode = mode
        if mode == "none":
            self.last_comm_success_rate = 0.0
            self.last_avg_neighbor_num = 0.0
            self.last_lower_msg_age = 0.0
            self.last_lower_avg_delay_steps = 0.0
            self.last_comm_density = 0.0
        self._lower_link_state.fill_(self.burst_initial_state)
        for i in range(self.n_agents):
            self._lower_link_state[i, i] = 0
        self._upper_link_state.fill_(self.burst_initial_state)

    def _deliver_lower_queues(self):
        delivered_count = 0
        delivered_age_sum = 0.0
        aggregated_count = 0.0
        multi_packet_edges = 0.0
        active_edges = 0.0
        for sender in range(self.n_agents):
            for receiver in range(self.n_agents):
                if sender == receiver:
                    continue
                q = self._lower_link_queues[sender][receiver]
                if not q:
                    continue
                keep = []
                due = []
                for item in q:
                    record = self._packet_record_from_queue_item(item, sender, receiver)
                    if int(record["deliver_step"]) <= int(self.step_count):
                        due.append(record)
                    else:
                        keep.append(item)
                if due:
                    agg_msg, agg_quality, agg_state, agg_src_step, stats = self._aggregate_packets(receiver, sender, due)
                    self._lower_delivered_messages[receiver, sender] = agg_msg
                    self._lower_delivered_steps[receiver, sender] = int(agg_src_step)
                    self._lower_delivered_quality[receiver, sender] = float(agg_quality)
                    self._lower_delivered_state[receiver, sender] = int(agg_state)
                    self._lower_delivered_mask[receiver, sender] = True
                    self._maybe_apply_basic_task_handoff(
                        receiver_id=receiver,
                        sender_id=sender,
                        msg=agg_msg,
                        quality=float(agg_quality),
                        source_step=int(agg_src_step),
                    )
                    delivered_count += len(due)
                    delivered_age_sum += sum(max(0.0, float(self.step_count - int(r.get("src_step", self.step_count)))) for r in due)
                    aggregated_count += float(stats.get("packet_count", len(due)))
                    multi_packet_edges += float(stats.get("multi_packet_edge", 0.0))
                    active_edges += 1.0
                self._lower_link_queues[sender][receiver] = keep
        return delivered_count, delivered_age_sum, aggregated_count, multi_packet_edges, active_edges

    def _make_lower_comm_graph_config(self):
        return BasicCommConfig(
            n_agents=self.n_agents,
            space_z=float(self.space_size[2].item()),
            dt=self.dt,
            comm_reliable_range=self.comm_reliable_range,
            comm_max_range=self.comm_max_range,
            comm_base_loss=self.comm_loss_prob,
            comm_attenuation=self.comm_attenuation,
            effective_sound_speed=self.lower_effective_sound_speed,
            extra_delay_steps=self.lower_extra_delay_steps,
            max_delay_steps=self.lower_max_delay_steps,
            payload_bits=self.lower_payload_bits,
            rate_bps=self.lower_rate_bps,
            msg_ttl_steps=self.lower_msg_ttl_steps,
            payload_loss_scale=self.payload_loss_scale,
            use_burst_comm=self.use_burst_comm,
            edge_delay_lambda=self.edge_delay_lambda,
            edge_age_lambda=self.edge_age_lambda,
            edge_queue_lambda=self.edge_queue_lambda,
            device=self.device,
            dtype=self.dtype,
        )

    def _sync_lower_comm_graph_config(self):
        if not hasattr(self, "lower_comm_graph"):
            return
        self.lower_comm_graph.update_from_env(
            comm_reliable_range=self.comm_reliable_range,
            comm_max_range=self.comm_max_range,
            comm_base_loss=self.comm_loss_prob,
            comm_attenuation=self.comm_attenuation,
            effective_sound_speed=self.lower_effective_sound_speed,
            extra_delay_steps=self.lower_extra_delay_steps,
            max_delay_steps=self.lower_max_delay_steps,
            payload_bits=self.lower_payload_bits,
            rate_bps=self.lower_rate_bps,
            msg_ttl_steps=self.lower_msg_ttl_steps,
            payload_loss_scale=self.payload_loss_scale,
            use_burst_comm=self.use_burst_comm,
            use_dynamic_comm_graph=self.use_dynamic_comm_graph,
            use_depth_comm_loss=self.use_depth_comm_loss,
            use_flow_comm_loss=self.use_flow_comm_loss,
            use_noise_comm_loss=self.use_noise_comm_loss,
            use_snr_comm_model=self.use_snr_comm_model,
        )

    def _build_lower_edge_state(self):
        self._sync_lower_comm_graph_config()
        flow = self._flow_at(self._agent_pos)
        out = self.lower_comm_graph.build_lower_graph(
            agent_pos=self._agent_pos,
            step_count=int(self.step_count),
            delivered_steps=self._lower_delivered_steps,
            lower_link_state=self._lower_link_state,
            burst_state_pdr=self.burst_state_pdr,
            link_queues=self._lower_link_queues,
            flow=flow,
            use_comm=self.use_comm,
        )
        self._edge_features = out["edge_features"].to(device=self.device, dtype=self.dtype)
        self._edge_weights = out["edge_weights"].to(device=self.device, dtype=self.dtype)
        self.comm_graph = out
        stats = getattr(self.lower_comm_graph, "last_stats", {})
        self.last_edge_weight = float(stats.get("edge_weight", 0.0))
        self.last_link_loss_prob = float(stats.get("loss_prob", 0.0))
        self.last_link_snr_norm = float(stats.get("snr_norm", 0.0))
        self.last_link_bandwidth_bps = float(stats.get("bandwidth", 0.0))
        self.last_comm_noise_mean = float(stats.get("comm_noise", 0.0))
        self.last_depth_gap_mean = float(stats.get("depth_gap", 0.0))
        self.last_flow_diff_mean = float(stats.get("flow_diff", 0.0))
        return out

    def _invalidate_message_cache(self):
        self._message_for_receiver_cache = {}
        self._message_type_cache = {}

    def _message_for_receiver(self, sender_id, receiver_id):
        sender_id = int(sender_id)
        cache_key = (int(receiver_id), sender_id)
        cached = self._message_for_receiver_cache.get(cache_key)
        if cached is not None:
            return cached.clone()
        out = self._comm_messages[sender_id].to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim).clone()
        self._message_for_receiver_cache[cache_key] = out
        return out.clone()

    def _packet_from_vector(self, sender_id, vector, payload_bits=None):
        return {
            "vector": vector.to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim).clone(),
            "msg_type": "STATUS",
            "priority": 0.10,
            "critical": False,
            "payload_bits": int(max(1, self.lower_payload_bits if payload_bits is None else payload_bits)),
            "sender_id": int(sender_id),
        }

    def _messages_for_receiver(self, sender_id, receiver_id):
        return [self._packet_from_vector(sender_id, self._message_for_receiver(sender_id, receiver_id))]

    def _packet_record_from_queue_item(self, item, sender_id, receiver_id):
        if isinstance(item, dict):
            vector = item.get("vector", item.get("msg", self._comm_messages[int(sender_id)]))
            return {
                "deliver_step": int(item.get("deliver_step", self.step_count)),
                "vector": vector.to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim).clone(),
                "quality": float(item.get("quality", 1.0)),
                "state": int(item.get("state", 0)),
                "src_step": int(item.get("src_step", item.get("deliver_step", self.step_count))),
                "payload_bits": int(max(1, item.get("payload_bits", self.lower_payload_bits))),
                "sender_id": int(item.get("sender_id", sender_id)),
            }
        deliver_step, msg, quality, state, src_step = item
        return {
            "deliver_step": int(deliver_step),
            "vector": msg.to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim).clone(),
            "quality": float(quality),
            "state": int(state),
            "src_step": int(src_step),
            "payload_bits": int(self.lower_payload_bits),
            "sender_id": int(sender_id),
        }

    def _aggregate_packets(self, receiver_id, sender_id, packet_records):
        records = list(packet_records or [])
        if not records:
            fallback = self._packet_from_vector(sender_id, self._message_for_receiver(sender_id, receiver_id))
            records = [{**fallback, "quality": 1.0, "state": 0, "src_step": int(self.step_count)}]
        first = records[0]
        return (
            first["vector"].to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim).clone(),
            float(first.get("quality", 1.0)),
            int(first.get("state", 0)),
            int(first.get("src_step", self.step_count)),
            {"packet_count": float(len(records)), "multi_packet_edge": 1.0 if len(records) >= 2 else 0.0},
        )

    def _maybe_apply_basic_task_handoff(self, receiver_id, sender_id, msg, quality=1.0, source_step=None):
        receiver_id = int(receiver_id)
        sender_id = int(sender_id)
        if receiver_id != int(self.executor_idx) or sender_id < 0 or sender_id >= int(self.n_search):
            return False
        if bool(getattr(self, "executor_target_assigned", False)):
            return False
        msg = msg.to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim)
        if msg.numel() < 13 or float(msg[9].item()) <= 0.5:
            return False
        target_est = torch.clamp(msg[10:13].clone(), min=self._lower_bound, max=self.space_size)
        if torch.any(torch.isnan(target_est)):
            return False

        self._agent_task_known[self.executor_idx] = True
        self._agent_task_est[self.executor_idx].copy_(target_est)
        self._target_belief_soft_executor_assigned = False
        self.total_waypoints_per_agent[self.executor_idx] += 1
        self.executor_target_assigned = True
        self.current_target_arrived[self.executor_idx] = False
        self._executor_received_target_step = int(self.step_count if source_step is None else max(int(source_step), int(self.step_count)))
        if getattr(self, "_found_step", None) is not None:
            self.last_handoff_delay = float(int(self.step_count) - int(self._found_step))
        self.last_direct_handover_success += 1.0
        self.last_direct_handover_episode_success += 1.0
        self.last_direct_handover_count += 1.0
        self.last_handover_success_rate = 1.0
        self.last_handover_episode_success_rate = 1.0
        return True

    def _apply_basic_lower_graph_mask(self, reachable_matrix, probs):
        self._role_topology_mask = self._non_diag_bool.to(dtype=self.dtype)
        self._role_score_matrix = self._role_topology_mask.clone()
        self._direct_handover_bypass_mask.zero_()
        self._direct_handover_bypass_probs.zero_()
        self.last_role_topology_density = 1.0
        self.last_role_active_density = 1.0
        self.last_role_physical_density = 1.0
        self.last_role_pruned_ratio = 0.0
        self.last_role_critical_link_density = 0.0
        self.last_direct_handover_ready = 0.0
        self.last_direct_handover_bypass_enabled = 0.0
        self.last_direct_handover_bypass_ready = 0.0
        self.last_direct_handover_bypass_gate_pending = 0.0
        self.last_direct_handover_bypass_gate_pressure = 0.0
        self._normalize_guarded_effective_diagnostics()
        return reachable_matrix, probs

    def _apply_message_profile_degradation(self, receiver_id, sender_id, msg, src_step):
        return msg.to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim).clone(), int(src_step)

    def _build_comm_matrix(self):
        reachable_matrix = torch.zeros((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device)
        success_matrix = torch.zeros_like(reachable_matrix)
        delay_sum = 0.0
        delay_count = 0
        lower_energy = 0.0
        attempted_packet_count = 0
        scheduled_success_packet_count = 0
        arrived_packet_count = 0
        aggregated_packet_count = 0.0
        multi_packet_edge_count = 0.0
        active_packet_edges = 0.0
        self.last_direct_handover_attempts = 0.0
        self.last_direct_handover_success = 0.0

        self._advance_lower_link_states()
        delivered_count, delivered_age_sum, queue_aggregated, queue_multi_edges, queue_active_edges = self._deliver_lower_queues()
        arrived_packet_count += int(delivered_count)
        aggregated_packet_count += float(queue_aggregated)
        multi_packet_edge_count += float(queue_multi_edges)
        active_packet_edges += float(queue_active_edges)

        edge_out = self._build_lower_edge_state()
        reachable_matrix = edge_out["reachable"].to(device=self.device, dtype=self.dtype)
        probs = edge_out["success_prob"].to(device=self.device, dtype=self.dtype)
        delay_steps_matrix = edge_out["delay_steps"].to(device=self.device, dtype=self.dtype)
        reachable_matrix, probs = self._apply_basic_lower_graph_mask(reachable_matrix, probs)

        if self.use_comm:
            keep = (torch.rand((self.n_agents, self.n_agents), dtype=self.dtype, device=self.device) < probs).to(self.dtype)
            keep.fill_diagonal_(0.0)
            success_matrix = reachable_matrix * keep
            immediate_by_edge = {}
            for receiver in range(self.n_agents):
                for sender in range(self.n_agents):
                    if receiver == sender:
                        continue
                    packets = self._messages_for_receiver(sender, receiver)
                    if float(reachable_matrix[receiver, sender].item()) > 0.5:
                        attempted_packet_count += len(packets)
                    if float(success_matrix[receiver, sender].item()) <= 0.5:
                        continue
                    delay_steps = int(delay_steps_matrix[receiver, sender].item())
                    state = int(self._lower_link_state[receiver, sender].item())
                    quality = float(torch.clamp(self._edge_weights[receiver, sender], 0.0, 1.0).item())
                    src_step = int(self.step_count)
                    records = []
                    for packet in packets:
                        payload_bits = int(max(1, packet.get("payload_bits", self.lower_payload_bits)))
                        lower_energy += self._add_comm_energy(
                            sender_id=sender,
                            receiver_id=receiver,
                            payload_bits=payload_bits,
                            rate_bps=self.lower_rate_bps,
                            tx=True,
                            rx=True,
                        )
                        scheduled_success_packet_count += 1
                        records.append({
                            "deliver_step": int(src_step + delay_steps),
                            "vector": packet["vector"].to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim).clone(),
                            "quality": quality,
                            "state": state,
                            "src_step": src_step,
                            "payload_bits": payload_bits,
                            "sender_id": int(sender),
                        })
                        delay_sum += float(delay_steps)
                        delay_count += 1
                    if delay_steps <= 0:
                        immediate_by_edge.setdefault((receiver, sender), []).extend(records)
                    else:
                        self._lower_link_queues[sender][receiver].extend(records)

            for (receiver, sender), records in immediate_by_edge.items():
                agg_msg, agg_quality, agg_state, agg_src_step, stats = self._aggregate_packets(receiver, sender, records)
                self._lower_delivered_messages[receiver, sender] = agg_msg
                self._lower_delivered_steps[receiver, sender] = int(agg_src_step)
                self._lower_delivered_quality[receiver, sender] = float(agg_quality)
                self._lower_delivered_state[receiver, sender] = int(agg_state)
                self._lower_delivered_mask[receiver, sender] = True
                self._maybe_apply_basic_task_handoff(
                    receiver_id=receiver,
                    sender_id=sender,
                    msg=agg_msg,
                    quality=float(agg_quality),
                    source_step=int(agg_src_step),
                )
                arrived_packet_count += len(records)
                delivered_age_sum += sum(max(0.0, float(self.step_count - int(r.get("src_step", self.step_count)))) for r in records)
                aggregated_packet_count += float(stats.get("packet_count", len(records)))
                multi_packet_edge_count += float(stats.get("multi_packet_edge", 0.0))
                active_packet_edges += 1.0

        self.last_sem_sent_packet_count = float(attempted_packet_count)
        self.last_sem_delivered_packet_count = float(scheduled_success_packet_count)
        self.last_sem_arrived_packet_count = float(arrived_packet_count)
        self.last_sem_aggregated_packet_count = float(aggregated_packet_count)
        self.last_sem_multi_packet_edge_count = float(multi_packet_edge_count)
        self.last_sem_avg_packets_per_active_edge = float(aggregated_packet_count / max(1.0, active_packet_edges))
        if attempted_packet_count > 0:
            self.last_sem_packet_delivery_rate = float(np.clip(scheduled_success_packet_count / max(1.0, float(attempted_packet_count)), 0.0, 1.0))
            self.last_sem_packet_drop_rate = float(np.clip(1.0 - self.last_sem_packet_delivery_rate, 0.0, 1.0))
        else:
            self.last_sem_packet_delivery_rate = 0.0
            self.last_sem_packet_drop_rate = 0.0
        self.last_sem_critical_delivery_rate = 0.0
        self.last_sem_critical_drop_rate = 0.0
        self.last_sem_critical_packet_delivery_rate = 0.0
        self.last_sem_critical_packet_drop_rate = 0.0

        ages = (self.step_count - self._lower_delivered_steps.to(dtype=torch.int32)).to(dtype=self.dtype)
        valid = self._lower_delivered_mask & (ages <= float(self.lower_msg_ttl_steps))
        valid.fill_diagonal_(False)
        self._comm_matrix = valid.to(dtype=self.dtype)
        self._comm_range_matrix = reachable_matrix

        possible_edges = max(1, self.n_agents * (self.n_agents - 1))
        range_edges = float(reachable_matrix.sum().item())
        success_edges = float(success_matrix.sum().item())
        active_edges = float(self._comm_matrix.sum().item())
        self.last_comm_density = range_edges / possible_edges
        self.last_comm_success_rate = success_edges / max(1.0, range_edges)
        self.last_avg_neighbor_num = active_edges / max(1, self.n_agents)
        self.last_lower_avg_delay_steps = delay_sum / max(1, delay_count)
        active_age = ages[valid]
        self.last_lower_msg_age = float(active_age.mean().item()) if active_age.numel() > 0 else 0.0
        self.last_lower_comm_energy = float(lower_energy)
        non_diag = self._non_diag_bool
        self.last_lower_good_state_ratio = float((self._lower_link_state[non_diag] == 0).float().mean().item())
        self.last_comm_energy_total = float(self.last_comm_energy.sum().item())
        self.last_handover_success_rate = 0.0
        self.last_handover_episode_success_rate = 0.0
        return self._comm_matrix

    def _reset_communication_state(self):
        self._edge_features.zero_()
        self._edge_weights.zero_()
        self.comm_graph = {}
        self._invalidate_message_cache()
        current_msgs = self._make_comm_messages()
        hist_len = self.comm_delay_steps + 1
        self._comm_message_history = [current_msgs.clone() for _ in range(hist_len)]
        self._comm_messages = current_msgs.clone()
        self._delayed_comm_messages = current_msgs.clone()
        self._invalidate_message_cache()
        self.last_comm_energy.zero_()
        self._reset_lower_delivery_state(current_msgs)
        self._comm_matrix = self._build_comm_matrix()

    def _update_communication_state(self):
        self._invalidate_message_cache()
        current_msgs = self._make_comm_messages()
        self._comm_messages = current_msgs.clone()
        self._comm_message_history.append(current_msgs.clone())
        hist_len = self.comm_delay_steps + 1
        if len(self._comm_message_history) > hist_len:
            self._comm_message_history = self._comm_message_history[-hist_len:]
        self._delayed_comm_messages = self._comm_message_history[0].clone()
        self._invalidate_message_cache()
        self._comm_matrix = self._build_comm_matrix()

    def _message_piece_from_msg(self, receiver_id, sender_id, msg, *, delivered, quality, age_norm, state_norm, pred_flag=0.0, pred_uncertainty=0.0, pred_age_norm=0.0):
        receiver_id = int(receiver_id)
        sender_id = int(sender_id)
        msg = msg.to(device=self.device, dtype=self.dtype).reshape(self.comm_msg_dim)
        edge = self._edge_features[receiver_id, sender_id] if self._edge_features.numel() else torch.zeros(self.edge_feature_dim, dtype=self.dtype, device=self.device)
        sender_pos, sender_vel, nav = msg[0:3], msg[3:6], msg[6:9]
        known_task, task_est = msg[9:10], msg[10:13]
        progress, finished, _, role = msg[13:14], msg[14:15], msg[15:16], msg[16:20]
        main_pos = torch.where(known_task > 0.5, task_est, nav)
        rel_pos = sender_pos - self._agent_pos[receiver_id]
        rel_vel = sender_vel - self._agent_vel[receiver_id]
        rel_main = main_pos - self._agent_pos[receiver_id]
        edge_weight = edge[12:13] if edge.numel() >= 13 else torch.tensor([float(quality)], dtype=self.dtype, device=self.device)
        loss_prob = edge[2:3] if edge.numel() >= 13 else torch.tensor([1.0 - float(quality)], dtype=self.dtype, device=self.device)
        delay_norm = edge[3:4] if edge.numel() >= 13 else torch.tensor([0.0], dtype=self.dtype, device=self.device)
        snr_norm = edge[5:6] if edge.numel() >= 13 else torch.tensor([float(quality)], dtype=self.dtype, device=self.device)
        bw_norm = edge[6:7] if edge.numel() >= 13 else torch.tensor([1.0], dtype=self.dtype, device=self.device)
        reachable = edge[0:1] if edge.numel() >= 13 else torch.tensor([float(delivered)], dtype=self.dtype, device=self.device)
        role_score = self._role_score_matrix[receiver_id, sender_id].reshape(1) if hasattr(self, "_role_score_matrix") and self._role_score_matrix.numel() > 0 else torch.tensor([float(delivered)], dtype=self.dtype, device=self.device)
        reliability = torch.clamp(edge_weight.reshape(()), 0.0, 1.0)
        piece = torch.empty(self.comm_obs_per_neighbor_dim, dtype=self.dtype, device=self.device)
        piece[0] = 1.0 if delivered else 0.0
        piece[1:4] = rel_pos.reshape(3)
        piece[4:7] = rel_vel.reshape(3)
        piece[7:10] = rel_main.reshape(3)
        piece[10] = float(quality)
        piece[11] = 0.0
        piece[12] = 0.10
        piece[13] = progress.reshape(-1)[0]
        piece[14] = finished.reshape(-1)[0]
        piece[15] = 0.0
        piece[16] = known_task.reshape(-1)[0]
        piece[17:21] = role.reshape(4)
        piece[21] = float(quality)
        piece[22] = float(age_norm)
        piece[23] = float(state_norm)
        piece[24] = edge_weight.reshape(-1)[0]
        piece[25] = loss_prob.reshape(-1)[0]
        piece[26] = delay_norm.reshape(-1)[0]
        piece[27] = snr_norm.reshape(-1)[0]
        piece[28] = bw_norm.reshape(-1)[0]
        piece[29] = reachable.reshape(-1)[0]
        piece[30] = 0.0
        piece[31] = 0.0
        piece[32] = 0.0
        piece[33] = reliability.reshape(-1)[0]
        piece[34] = 0.0
        piece[35] = 0.0
        piece[36] = role_score.reshape(-1)[0]
        return piece

    def _apply_comm_obs_stats_aggregate(self, stats_list):
        if not stats_list:
            stats_list = []
        reliability_sum = sum(float(s.get("reliability_sum", 0.0)) for s in stats_list)
        reliability_count = sum(float(s.get("reliability_count", 0.0)) for s in stats_list)
        self.last_pred_usage_ratio = 0.0
        self.last_pred_generated_ratio = 0.0
        self.last_pred_obs_ratio = 0.0
        self.last_pred_uncertainty = 0.0
        self.last_pred_sigma = 0.0
        self.last_prediction_reliability = 0.0
        self.last_avg_reliability = float(reliability_sum / max(1.0, reliability_count)) if reliability_count > 0 else 1.0
        self.last_conflict_score = 0.0
        self.last_belief_disagreement = 0.0
        self.last_quarantine_count = 0.0
        self.last_quarantine_active_count = 0.0

    def _record_comm_obs_stats(self, stats):
        buffer = getattr(self, "_comm_obs_stats_buffer", None)
        if buffer is not None:
            buffer.append(stats)
        else:
            self._apply_comm_obs_stats_aggregate([stats])

    def _get_comm_obs_for_agent(self, i):
        pieces = []
        zero_piece = torch.zeros(self.comm_obs_per_neighbor_dim, dtype=self.dtype, device=self.device)
        reliabilities = []
        for j in range(self.n_agents):
            if j == i:
                continue
            if bool(self._comm_matrix[i, j].item() > 0.5):
                age = torch.clamp(
                    (self._scalar(float(self.step_count)) - self._lower_delivered_steps[i, j].to(dtype=self.dtype)) /
                    max(1.0, float(self.lower_msg_ttl_steps)),
                    0.0,
                    1.0,
                ).item()
                quality = float(torch.clamp(self._lower_delivered_quality[i, j], 0.0, 1.0).item())
                state_norm = float((self._lower_delivered_state[i, j].to(dtype=self.dtype) / 2.0).item())
                piece = self._message_piece_from_msg(i, j, self._lower_delivered_messages[i, j], delivered=True, quality=quality, age_norm=age, state_norm=state_norm)
            else:
                piece = zero_piece
            reliabilities.append(float(piece[33].item()))
            pieces.append(piece)
        self._record_comm_obs_stats({
            "reliability_sum": float(sum(reliabilities)),
            "reliability_count": float(len(reliabilities)),
        })
        return torch.cat(pieces, dim=0)

    def _get_obs(self):
        obs = []
        # Aggregate communication-observation diagnostics over all receivers.
        # Previously _get_comm_obs_for_agent() overwrote these fields once per
        # agent, so the logged value mostly reflected the last agent/executor.
        self._comm_obs_stats_buffer = []
        nearest_d_all = self._nearest_obstacle_distance(self._agent_pos)
        norm_near_d_all = torch.clamp(nearest_d_all / 10.0, 0.0, 1.0)
        for i in range(self.n_agents):
            phase_onehot = self._vec([0.0, 1.0]) if bool(self._agent_task_known[i].item()) else self._vec([1.0, 0.0])
            pos, vel, nav_tgt = self._agent_pos[i], self._agent_vel[i], self._nav_targets[i]
            rel_nav = nav_tgt - pos
            nav_norm = torch.norm(rel_nav).clamp_min(self.eps)
            unit_rel_nav = rel_nav / nav_norm
            nav_distance_scalar = torch.clamp(nav_norm / 10.0, 0.0, 1.0).unsqueeze(0)
            speed = torch.norm(vel).clamp_min(self.eps)
            speed_scalar = torch.clamp(speed / (self._v_xy_max[i] + self.eps), 0.0, 1.0).unsqueeze(0)
            closing_speed = torch.dot(vel, unit_rel_nav)
            closing_speed_scalar = torch.tanh((closing_speed / (self._v_xy_max[i] + self.eps)).unsqueeze(0))
            rel_task = self._agent_task_est[i] - pos if bool(self._agent_task_known[i].item()) else self._zero3
            total_t = max(1, int(self.total_waypoints_per_agent[i].item()))
            progress_ratio = self.waypoint_reached_counts[i].to(self.dtype) / float(total_t)
            finished_flag = self.agent_finished[i].to(self.dtype)
            hold_ratio = torch.clamp(self.hold_counters[i].to(self.dtype) / max(1.0, float(self._hold_steps_for_agent(i))), 0.0, 1.0)
            misc = torch.stack([norm_near_d_all[i], progress_ratio, finished_flag, hold_ratio])
            obs_i = torch.cat([pos, vel, rel_nav, unit_rel_nav, rel_task, nav_distance_scalar, speed_scalar, closing_speed_scalar, misc, self.role_onehots[i], phase_onehot, self._get_comm_obs_for_agent(i)]).to(dtype=self.dtype)
            if obs_i.numel() != self.obs_dim:
                raise RuntimeError(f"obs dim mismatch for agent {i}: expected {self.obs_dim}, got {obs_i.numel()}")
            obs.append(obs_i)
        self._apply_comm_obs_stats_aggregate(self._comm_obs_stats_buffer)
        self._comm_obs_stats_buffer = None
        return obs

    def step(self, actions): return self._step_mission(actions)
    def _blocked_mask(self): return self.agent_finished.clone()

    def _compute_waypoint_prior_acc(self):
        rel = self._nav_targets - self._agent_pos
        prior_acc = torch.zeros_like(self._agent_acc)
        rel_xy = rel[:, :2]
        dist_xy = torch.norm(rel_xy, dim=1, keepdim=True).clamp_min(self.eps)
        dir_xy = rel_xy / dist_xy
        slow_xy = torch.clamp(dist_xy / self.prior_slow_radius_xy, 0.0, 1.0)
        desired_vel_xy = dir_xy * self._v_xy_max.unsqueeze(1) * slow_xy
        prior_acc[:, :2] = self.prior_kv_xy * (desired_vel_xy - self._agent_vel[:, :2])
        desired_vel_z = torch.clamp(rel[:, 2] / self.prior_slow_radius_z, -1.0, 1.0) * self._v_z_max
        prior_acc[:, 2] = self.prior_kv_z * (desired_vel_z - self._agent_vel[:, 2])
        prior_acc[:, :2] = torch.clamp(prior_acc[:, :2], min=-self._a_xy_max.unsqueeze(1), max=self._a_xy_max.unsqueeze(1))
        prior_acc[:, 2] = torch.clamp(prior_acc[:, 2], min=-self._a_z_max, max=self._a_z_max)
        return prior_acc

    def _actions_to_residual_acc(self, actions):
        residual_acc = torch.empty_like(actions)
        residual_acc[:, :2] = actions[:, :2] * self._a_xy_max.unsqueeze(1)
        residual_acc[:, 2] = actions[:, 2] * self._a_z_max
        return residual_acc

    def _reset_residual_prior_diagnostics(self):
        self.last_prior_term_norm = 0.0
        self.last_residual_term_norm = 0.0
        self.last_final_acc_cmd_norm = 0.0
        self.last_residual_contribution_ratio = 0.0
        self.last_prior_contribution_ratio = 0.0
        self.last_prior_term_norm_search = 0.0
        self.last_residual_term_norm_search = 0.0
        self.last_residual_contribution_ratio_search = 0.0
        self.last_prior_term_norm_executor = 0.0
        self.last_residual_term_norm_executor = 0.0
        self.last_residual_contribution_ratio_executor = 0.0
        self.last_residual_prior_cosine_v2 = 0.0
        self._reset_residual_struct_diagnostics()

    def _reset_residual_struct_diagnostics(self):
        self.last_residual_struct_gate_mean = 0.0
        self.last_residual_struct_gate_executor = 0.0
        self.last_residual_struct_lateral_ratio = 0.0
        self.last_residual_struct_antitarget_ratio = 0.0
        self.last_residual_struct_parallel_pos_ratio = 0.0
        self.last_residual_struct_raw_norm = 0.0
        self.last_residual_struct_out_norm = 0.0
        self._reset_residual_hybrid_diagnostics()

    def _reset_residual_hybrid_diagnostics(self):
        self.last_residual_hybrid_exec_cartesian_blend = 0.0
        self.last_residual_hybrid_soft_gate = 0.0
        self.last_residual_hybrid_executor_cartesian_ratio = 0.0
        self.last_residual_hybrid_mode_active = 0.0

    def _reset_robust_residual_reward_diagnostics_v2(self):
        self.last_reward_exec_task_progress_v2 = 0.0
        self.last_reward_near_target_speed_v2 = 0.0
        self.last_reward_hold_dense_v2 = 0.0
        self.last_reward_target_assigned_v2 = 0.0
        self.last_reward_handoff_delay_penalty_v2 = 0.0
        self.last_reward_belief_age_penalty_v2 = 0.0
        self.last_reward_risk_aware_residual_penalty_v2 = 0.0
        self.last_reward_anti_prior_penalty_v2 = 0.0
        self.last_reward_near_target_residual_penalty_v2 = 0.0
        self.last_reward_robust_residual_total_v2 = 0.0
        self.last_residual_risk_score_v2 = 0.0
        self.last_residual_prior_cosine_v2 = 0.0
        self.last_exec_knows_task_v3 = 0.0
        self.last_reward_motion_gated_by_comm_v3 = 0.0
        self.last_residual_hybrid_soft_gate = 0.0

    def _reset_point_control_reward_diagnostics_v3(self):
        self.last_reward_point_vel_align_v3 = 0.0
        self.last_reward_point_lateral_vel_penalty_v3 = 0.0
        self.last_reward_point_speed_profile_penalty_v3 = 0.0
        self.last_reward_point_residual_align_v3 = 0.0
        self.last_reward_point_residual_antitarget_penalty_v3 = 0.0
        self.last_reward_point_shaping_total_v3 = 0.0
        self.last_point_vel_align_mean_v3 = 0.0
        self.last_point_lateral_speed_mean_v3 = 0.0
        self.last_point_speed_error_mean_v3 = 0.0
        self.last_point_target_dist_mean_v3 = 0.0

    def _compute_residual_risk_score_v2(self):
        def clip01(value):
            try:
                return float(np.clip(float(value), 0.0, 1.0))
            except Exception:
                return 0.0

        risk = 0.0
        risk += 0.30 * clip01(getattr(self, "last_link_loss_prob", 0.0))
        risk += 0.20 * clip01(getattr(self, "last_upper_avg_delay_steps", 0.0) / 5.0)
        risk += 0.20 * clip01(getattr(self, "last_upper_belief_age", 0.0) / 100.0)
        risk += 0.20 * clip01(getattr(self, "last_sem_voi_stress", 0.0))
        risk += 0.10 * clip01(abs(getattr(self, "flow_gain", 0.0)) / 0.7)
        return float(np.clip(risk, 0.0, 1.0))

    def _get_point_shaping_target_v3(self, agent_i):
        i = int(agent_i)
        if i < 0 or i >= self.n_agents:
            return None
        if i == int(self.executor_idx):
            knows_task = bool(getattr(self, "executor_target_assigned", False))
            if hasattr(self, "_agent_task_known"):
                knows_task = knows_task or bool(self._agent_task_known[i].item())
            if bool(getattr(self, "task_found", False)):
                return self._task_target
        if hasattr(self, "_nav_targets") and self._nav_targets is not None:
            return self._nav_targets[i]
        return None

    def _get_structured_residual_target_v1(self, agent_i):
        i = int(agent_i)
        if i < 0 or i >= self.n_agents:
            return None, False
        if i == int(self.executor_idx):
            knows_task = bool(getattr(self, "executor_target_assigned", False))
            if hasattr(self, "_agent_task_known"):
                knows_task = knows_task or bool(self._agent_task_known[i].item())
            if bool(getattr(self, "task_found", False)) and knows_task:
                if hasattr(self, "_agent_task_est") and bool(self._agent_task_known[i].item()):
                    return self._agent_task_est[i], True
                if bool(getattr(self, "executor_target_assigned", False)) and hasattr(self, "_nav_targets"):
                    return self._nav_targets[i], True
            if hasattr(self, "_nav_targets") and self._nav_targets is not None:
                return self._nav_targets[i], False
            return None, False
        if hasattr(self, "_nav_targets") and self._nav_targets is not None:
            return self._nav_targets[i], True
        return None, True

    def _structure_residual_acc_target_decomposed_v1(self, raw_residual_acc):
        raw = torch.nan_to_num(raw_residual_acc, nan=0.0, posinf=0.0, neginf=0.0)
        structured = torch.zeros_like(raw)
        gates = torch.zeros(self.n_agents, dtype=self.dtype, device=self.device)
        lateral_ratios = []
        antitarget_ratios = []
        parallel_pos_ratios = []

        try:
            risk = float(getattr(self, "last_residual_risk_score_v2", 0.0))
            if not np.isfinite(risk):
                risk = self._compute_residual_risk_score_v2()
        except Exception:
            risk = 0.0
        risk = float(np.clip(risk, 0.0, 1.0))
        risk_gate = (
            float(self.residual_struct_low_risk_gate) * (1.0 - risk)
            + float(self.residual_struct_high_risk_gate) * risk
        )

        blocked = self._blocked_mask() if hasattr(self, "_blocked_mask") else torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        for i in range(self.n_agents):
            raw_i = raw[i]
            raw_norm = torch.norm(raw_i).clamp_min(self.eps)
            target, executor_knows_target = self._get_structured_residual_target_v1(i)
            if target is None or bool(blocked[i].item()):
                lateral_ratios.append(torch.zeros((), dtype=self.dtype, device=self.device))
                antitarget_ratios.append(torch.zeros((), dtype=self.dtype, device=self.device))
                parallel_pos_ratios.append(torch.zeros((), dtype=self.dtype, device=self.device))
                continue

            target_vec = target - self._agent_pos[i]
            dist = torch.norm(target_vec)
            if (not bool(torch.isfinite(dist).item())) or float(dist.item()) <= self.eps:
                lateral_ratios.append(torch.zeros((), dtype=self.dtype, device=self.device))
                antitarget_ratios.append(torch.zeros((), dtype=self.dtype, device=self.device))
                parallel_pos_ratios.append(torch.zeros((), dtype=self.dtype, device=self.device))
                continue

            target_dir = target_vec / dist.clamp_min(self.eps)
            parallel_scalar = torch.sum(raw_i * target_dir)
            parallel_vec = parallel_scalar * target_dir
            lateral_vec = raw_i - parallel_vec
            positive_parallel = torch.clamp(parallel_scalar, min=0.0) * target_dir
            negative_parallel = torch.clamp(parallel_scalar, max=0.0) * target_dir

            is_exec = i == int(self.executor_idx)
            lateral_scale = self.residual_struct_lateral_scale_exec if is_exec else self.residual_struct_lateral_scale_search
            out_i = (
                positive_parallel
                + float(self.residual_struct_antitarget_scale) * negative_parallel
                + float(lateral_scale) * lateral_vec
            )

            gate = float(self.residual_struct_base_gate) * risk_gate
            near_radius = self.residual_struct_near_radius_exec if is_exec else self.residual_struct_near_radius_search
            if float(dist.item()) < float(near_radius):
                gate *= float(self.residual_struct_near_target_gate)
            if is_exec and not bool(executor_knows_target):
                gate *= float(self.residual_struct_unknown_target_gate_exec)
            gate = float(np.clip(gate, 0.0, 1.0))
            gates[i] = gate
            structured[i] = out_i * gate

            lateral_ratios.append(torch.norm(lateral_vec) / raw_norm)
            antitarget_ratios.append(torch.norm(negative_parallel) / raw_norm)
            parallel_pos_ratios.append(torch.norm(positive_parallel) / raw_norm)

        structured[:, :2] = torch.clamp(structured[:, :2], min=-self._a_xy_max.unsqueeze(1), max=self._a_xy_max.unsqueeze(1))
        structured[:, 2] = torch.clamp(structured[:, 2], min=-self._a_z_max, max=self._a_z_max)
        structured = torch.nan_to_num(structured, nan=0.0, posinf=0.0, neginf=0.0)

        def mean_or_zero(vals):
            return torch.stack(vals).mean() if vals else torch.zeros((), dtype=self.dtype, device=self.device)

        self.last_residual_struct_gate_mean = float(torch.nan_to_num(gates.mean(), nan=0.0).item())
        self.last_residual_struct_gate_executor = float(torch.nan_to_num(gates[int(self.executor_idx)], nan=0.0).item())
        self.last_residual_struct_lateral_ratio = float(torch.nan_to_num(mean_or_zero(lateral_ratios), nan=0.0).item())
        self.last_residual_struct_antitarget_ratio = float(torch.nan_to_num(mean_or_zero(antitarget_ratios), nan=0.0).item())
        self.last_residual_struct_parallel_pos_ratio = float(torch.nan_to_num(mean_or_zero(parallel_pos_ratios), nan=0.0).item())
        self.last_residual_struct_raw_norm = float(torch.norm(raw, dim=1).mean().item())
        self.last_residual_struct_out_norm = float(torch.norm(structured, dim=1).mean().item())
        return structured

    def _compute_v3_target_motion_gate(self, exec_knows_task):
        if self.reward_profile != "residual_point_v3":
            return 1.0
        if not bool(getattr(self, "residual_hybrid_use_soft_gate", True)):
            return 1.0 if bool(exec_knows_task) else 0.0
        if bool(exec_knows_task):
            return float(self.residual_hybrid_target_known_soft_gate)
        if bool(getattr(self, "task_found", False)):
            return float(self.residual_hybrid_target_found_soft_gate)
        return float(self.residual_hybrid_target_unknown_soft_gate)

    def _compute_v3_executor_dense_gate(self, motion_gate):
        if self.reward_profile != "residual_point_v3":
            return 1.0
        raw = float(motion_gate)
        if bool(getattr(self, "residual_hybrid_use_soft_gate", True)):
            return max(raw, 0.35)
        return raw

    def _safe_float_v3(self, value):
        try:
            value = float(value)
        except Exception:
            return 0.0
        return value if np.isfinite(value) else 0.0

    def _compute_point_control_shaping_v3(self, actions=None, prior_terms=None, residual_terms=None, final_acc_cmds=None):
        self._reset_point_control_reward_diagnostics_v3()
        if self.reward_profile != "residual_point_v3":
            return torch.zeros((), dtype=self.dtype, device=self.device)

        residual_terms = self._last_residual_term_v3 if residual_terms is None else residual_terms
        final_acc_cmds = self._last_final_acc_cmd_v3 if final_acc_cmds is None else final_acc_cmds
        if residual_terms is None and final_acc_cmds is not None and prior_terms is not None:
            residual_terms = final_acc_cmds - prior_terms

        reward_vel_vals = []
        reward_lat_vals = []
        reward_speed_vals = []
        reward_res_vals = []
        reward_anti_vals = []
        vel_align_norm_vals = []
        lateral_speed_vals = []
        speed_error_vals = []
        target_dist_vals = []
        point_vals = []

        for i in range(self.n_agents):
            target = self._get_point_shaping_target_v3(i)
            if target is None:
                continue
            target_vec = target - self._agent_pos[i]
            dist = torch.norm(target_vec)
            if not bool(torch.isfinite(dist).item()):
                continue
            target_dir = target_vec / dist.clamp_min(self.eps)
            vel = self._agent_vel[i]
            speed = torch.norm(vel)
            v_parallel = torch.sum(vel * target_dir)
            v_perp = vel - v_parallel * target_dir
            v_perp_norm = torch.norm(v_perp)

            v_xy = self._v_xy_max[i] if hasattr(self, "_v_xy_max") else torch.tensor(1.0, dtype=self.dtype, device=self.device)
            v_z = self._v_z_max[i] if hasattr(self, "_v_z_max") else torch.tensor(1.0, dtype=self.dtype, device=self.device)
            v_ref = torch.sqrt(v_xy * v_xy + v_z * v_z).clamp_min(self.eps)
            is_exec = i == int(self.executor_idx)
            point_gate = 1.0
            if is_exec:
                knows_task = bool(getattr(self, "executor_target_assigned", False))
                if hasattr(self, "_agent_task_known"):
                    knows_task = knows_task or bool(self._agent_task_known[i].item())
                raw_gate = float(self._compute_v3_target_motion_gate(knows_task))
                point_gate = self._compute_v3_executor_dense_gate(raw_gate)
            slow_radius = self.point_slow_radius_exec_v3 if is_exec else self.point_slow_radius_search_v3
            near_radius = self.point_near_radius_exec_v3 if is_exec else self.point_near_radius_search_v3
            desired_speed = v_ref * torch.clamp(dist / max(float(slow_radius), self.eps), 0.0, 1.0)

            vel_align_norm = torch.clamp(v_parallel / v_ref, -1.0, 1.0)
            gain_vel = self.point_vel_align_gain_exec_v3 if is_exec else self.point_vel_align_gain_search_v3
            reward_vel = gain_vel * torch.clamp(vel_align_norm, min=0.0)

            lateral_norm = v_perp_norm / v_ref
            gain_lat = self.point_lateral_vel_penalty_exec_v3 if is_exec else self.point_lateral_vel_penalty_search_v3
            reward_lat = -gain_lat * lateral_norm * lateral_norm

            speed_error_norm = (speed - desired_speed) / v_ref
            gain_speed = self.point_speed_profile_penalty_exec_v3 if is_exec else self.point_speed_profile_penalty_search_v3
            reward_speed = -gain_speed * speed_error_norm * speed_error_norm

            reward_res = torch.zeros((), dtype=self.dtype, device=self.device)
            reward_anti = torch.zeros((), dtype=self.dtype, device=self.device)
            if residual_terms is not None:
                residual_vec = residual_terms[i]
                residual_parallel = torch.sum(residual_vec * target_dir)
                a_xy = self._a_xy_max[i] if hasattr(self, "_a_xy_max") else torch.tensor(1.0, dtype=self.dtype, device=self.device)
                a_z = self._a_z_max[i] if hasattr(self, "_a_z_max") else torch.tensor(1.0, dtype=self.dtype, device=self.device)
                residual_norm_ref = torch.sqrt(a_xy * a_xy + a_z * a_z).clamp_min(self.eps)
                residual_align_norm = torch.clamp(residual_parallel / residual_norm_ref, -1.0, 1.0)
                far_gate = 1.0 if float(dist.item()) > float(near_radius) else 0.3
                gain_res = self.point_residual_align_gain_exec_v3 if is_exec else self.point_residual_align_gain_search_v3
                gain_anti = self.point_residual_antitarget_penalty_exec_v3 if is_exec else self.point_residual_antitarget_penalty_search_v3
                reward_res = gain_res * torch.clamp(residual_align_norm, min=0.0) * far_gate
                reward_anti = -gain_anti * torch.clamp(-residual_align_norm, min=0.0)

            point_i = reward_vel + reward_lat + reward_speed + reward_res + reward_anti
            if is_exec:
                point_i = point_i * point_gate
                reward_vel = reward_vel * point_gate
                reward_lat = reward_lat * point_gate
                reward_speed = reward_speed * point_gate
                reward_res = reward_res * point_gate
                reward_anti = reward_anti * point_gate
            finite_terms = (reward_vel, reward_lat, reward_speed, reward_res, reward_anti, point_i)
            if not all(bool(torch.isfinite(x).item()) for x in finite_terms):
                continue

            reward_vel_vals.append(reward_vel)
            reward_lat_vals.append(reward_lat)
            reward_speed_vals.append(reward_speed)
            reward_res_vals.append(reward_res)
            reward_anti_vals.append(reward_anti)
            vel_align_norm_vals.append(vel_align_norm)
            lateral_speed_vals.append(v_perp_norm)
            speed_error_vals.append(torch.abs(speed_error_norm))
            target_dist_vals.append(dist)
            point_vals.append(point_i)

        if not point_vals:
            return torch.zeros((), dtype=self.dtype, device=self.device)

        def mean_tensor(vals):
            return torch.stack(vals).mean() if vals else torch.zeros((), dtype=self.dtype, device=self.device)

        raw_total = torch.stack(point_vals).sum()
        clip = float(self.point_shaping_total_clip_v3)
        reward_point_total = torch.clamp(raw_total, -clip, clip) if clip > 0.0 else raw_total * 0.0

        self.last_reward_point_vel_align_v3 = self._safe_float_v3(torch.stack(reward_vel_vals).sum().item())
        self.last_reward_point_lateral_vel_penalty_v3 = self._safe_float_v3(torch.stack(reward_lat_vals).sum().item())
        self.last_reward_point_speed_profile_penalty_v3 = self._safe_float_v3(torch.stack(reward_speed_vals).sum().item())
        self.last_reward_point_residual_align_v3 = self._safe_float_v3(torch.stack(reward_res_vals).sum().item())
        self.last_reward_point_residual_antitarget_penalty_v3 = self._safe_float_v3(torch.stack(reward_anti_vals).sum().item())
        self.last_reward_point_shaping_total_v3 = self._safe_float_v3(reward_point_total.item())
        self.last_point_vel_align_mean_v3 = self._safe_float_v3(mean_tensor(vel_align_norm_vals).item())
        self.last_point_lateral_speed_mean_v3 = self._safe_float_v3(mean_tensor(lateral_speed_vals).item())
        self.last_point_speed_error_mean_v3 = self._safe_float_v3(mean_tensor(speed_error_vals).item())
        self.last_point_target_dist_mean_v3 = self._safe_float_v3(mean_tensor(target_dist_vals).item())
        return torch.nan_to_num(reward_point_total, nan=0.0, posinf=0.0, neginf=0.0)

    def _record_residual_prior_diagnostics(self, prior_term, residual_term, final_cmd):
        prior_norm = torch.norm(prior_term.detach(), dim=1)
        residual_norm = torch.norm(residual_term.detach(), dim=1)
        final_norm = torch.norm(final_cmd.detach(), dim=1)
        denom = prior_norm + residual_norm + 1e-6
        residual_ratio = residual_norm / denom
        prior_ratio = prior_norm / denom

        self.last_prior_term_norm = float(prior_norm.mean().item())
        self.last_residual_term_norm = float(residual_norm.mean().item())
        self.last_final_acc_cmd_norm = float(final_norm.mean().item())
        self.last_residual_contribution_ratio = float(residual_ratio.mean().item())
        self.last_prior_contribution_ratio = float(prior_ratio.mean().item())

        search_slice = slice(0, int(self.n_search))
        self.last_prior_term_norm_search = float(prior_norm[search_slice].mean().item()) if self.n_search > 0 else 0.0
        self.last_residual_term_norm_search = float(residual_norm[search_slice].mean().item()) if self.n_search > 0 else 0.0
        self.last_residual_contribution_ratio_search = float(residual_ratio[search_slice].mean().item()) if self.n_search > 0 else 0.0

        exec_idx = int(self.executor_idx)
        self.last_prior_term_norm_executor = float(prior_norm[exec_idx].item())
        self.last_residual_term_norm_executor = float(residual_norm[exec_idx].item())
        self.last_residual_contribution_ratio_executor = float(residual_ratio[exec_idx].item())

    def _blend_residual_prior_acc(self, residual_acc):
        if not self.use_residual_prior:
            prior_acc = torch.zeros_like(residual_acc)
            prior_term = torch.zeros_like(residual_acc)
            residual_term = residual_acc
            desired_acc = residual_acc
        else:
            prior_acc = self._compute_waypoint_prior_acc()
            prior_term = self._prior_strength.unsqueeze(1) * prior_acc
            residual_term = self._residual_scale.unsqueeze(1) * residual_acc
            desired_acc = prior_term + residual_term
        self._last_prior_acc_v2.copy_(prior_acc.detach())
        self._last_prior_term_v3.copy_(prior_term.detach())
        self._last_residual_term_v3.copy_(residual_term.detach())
        self._last_final_acc_cmd_v3.copy_(desired_acc.detach())
        self._record_residual_prior_diagnostics(prior_term, residual_term, desired_acc)
        desired_acc[:, :2] = torch.clamp(desired_acc[:, :2], min=-self._a_xy_max.unsqueeze(1), max=self._a_xy_max.unsqueeze(1))
        desired_acc[:, 2] = torch.clamp(desired_acc[:, 2], min=-self._a_z_max, max=self._a_z_max)
        return desired_acc

    def _apply_robust_actuator(self, desired_acc_cmd):
        """Apply actuator delay, lag, noise and disturbed acceleration limits."""
        if not self.use_robust_disturbance:
            return desired_acc_cmd
        cmd = desired_acc_cmd.clone()
        self._desired_acc_history.append(cmd.clone())
        hist_len = self.robust_action_delay_steps + 1
        if len(self._desired_acc_history) > hist_len:
            self._desired_acc_history = self._desired_acc_history[-hist_len:]
        if self.robust_action_delay_steps > 0 and len(self._desired_acc_history) >= hist_len:
            cmd = self._desired_acc_history[0].clone()
        if self.actuator_lag > 0.0:
            cmd = (1.0 - self.actuator_lag) * cmd + self.actuator_lag * self._prev_desired_acc_cmd
        self._prev_desired_acc_cmd.copy_(cmd.detach())
        if self._current_action_noise_std > 0.0:
            noise = torch.randn_like(cmd)
            noise[:, :2] *= self._a_xy_max_nominal.unsqueeze(1)
            noise[:, 2] *= self._a_z_max_nominal
            cmd = cmd + self._current_action_noise_std * noise
        cmd[:, :2] = cmd[:, :2] * self._actuator_scale_xy.unsqueeze(1)
        cmd[:, 2] = cmd[:, 2] * self._actuator_scale_z
        cmd[:, :2] = torch.clamp(cmd[:, :2], min=-self._dyn_a_xy_max.unsqueeze(1), max=self._dyn_a_xy_max.unsqueeze(1))
        cmd[:, 2] = torch.clamp(cmd[:, 2], min=-self._dyn_a_z_max, max=self._dyn_a_z_max)
        return cmd

    def _apply_agent_dynamics(self, actions):
        actions = torch.clamp(self._actions_to_tensor(actions), -1.0, 1.0)
        self._collision_flags.zero_()
        old_pos = self._agent_pos.clone()
        blocked_mask = self._blocked_mask()
        if torch.any(blocked_mask):
            actions = actions.clone(); actions[blocked_mask] = 0.0
        raw_residual_acc = self._actions_to_residual_acc(actions)
        if self.residual_action_mode == "target_decomposed_v1":
            self._reset_residual_hybrid_diagnostics()
            residual_acc = self._structure_residual_acc_target_decomposed_v1(raw_residual_acc)
        elif self.residual_action_mode == "hybrid_v4":
            struct_acc = self._structure_residual_acc_target_decomposed_v1(raw_residual_acc)
            residual_acc = struct_acc.clone()
            exec_i = int(self.executor_idx)
            exec_knows_task = bool(getattr(self, "executor_target_assigned", False))
            if hasattr(self, "_agent_task_known"):
                exec_knows_task = exec_knows_task or bool(self._agent_task_known[exec_i].item())
            if exec_knows_task:
                blend = float(self.residual_hybrid_exec_cartesian_blend_known)
            elif bool(getattr(self, "task_found", False)):
                blend = float(self.residual_hybrid_exec_cartesian_blend_found)
            else:
                blend = float(self.residual_hybrid_exec_cartesian_blend_unknown)
            max_cart_blend = 1.0 - float(self.residual_hybrid_exec_struct_blend_min)
            blend = float(np.clip(blend, 0.0, max(0.0, max_cart_blend)))
            residual_acc[exec_i] = blend * raw_residual_acc[exec_i] + (1.0 - blend) * struct_acc[exec_i]
            self.last_residual_hybrid_exec_cartesian_blend = blend
            self.last_residual_hybrid_executor_cartesian_ratio = blend
            self.last_residual_hybrid_mode_active = 1.0
        else:
            self._reset_residual_struct_diagnostics()
            residual_acc = raw_residual_acc
        self._last_residual_acc.copy_(residual_acc.detach())
        self.last_residual_norm = float(torch.norm(residual_acc, dim=1).mean().item())
        desired_acc_cmd = self._blend_residual_prior_acc(residual_acc)
        desired_acc = self._apply_robust_actuator(desired_acc_cmd)
        self._agent_acc.copy_(desired_acc)
        flow = self._flow_at(self._agent_pos)
        self._agent_vel[:, :2] += desired_acc[:, :2] * self.dt
        self._agent_vel[:, :2] += (-self._dyn_drag_xy.unsqueeze(1) * self._agent_vel[:, :2] + flow[:, :2]) * self.dt
        self._agent_vel[:, 2] += desired_acc[:, 2] * self.dt
        self._agent_vel[:, 2] += (-self._dyn_drag_z * self._agent_vel[:, 2] + self._dyn_buoyancy_bias) * self.dt
        xy_speed = torch.norm(self._agent_vel[:, :2], dim=1).clamp_min(self.eps)
        xy_scale = torch.minimum(torch.ones_like(xy_speed), self._dyn_v_xy_max / xy_speed)
        self._agent_vel[:, :2] *= xy_scale.unsqueeze(1)
        self._agent_vel[:, 2] = torch.clamp(self._agent_vel[:, 2], -self._dyn_v_z_max, self._dyn_v_z_max)
        self._agent_pos += self._agent_vel * self.dt
        clipped = torch.clamp(self._agent_pos, min=self._lower_bound, max=self.space_size)
        hit_wall = torch.any(clipped != self._agent_pos, dim=1)
        self._agent_pos.copy_(clipped)
        if torch.any(hit_wall): self._agent_vel[hit_wall] *= 0.4
        inside_obstacle = self._points_inside_obstacles(self._agent_pos)
        if torch.any(inside_obstacle):
            self._agent_pos[inside_obstacle] = old_pos[inside_obstacle]
            self._agent_vel[inside_obstacle] *= -0.2
            self._collision_flags[inside_obstacle] = True
        if torch.any(blocked_mask):
            self._agent_pos[blocked_mask] = old_pos[blocked_mask]
            self._agent_vel[blocked_mask] = 0.0
            self._agent_acc[blocked_mask] = 0.0
        self.step_count += 1
        if hasattr(self, "_guarded_bypass_cooldown"):
            self._guarded_bypass_cooldown.clamp_(min=0)
            self._guarded_bypass_cooldown.sub_(torch.where(
                self._guarded_bypass_cooldown > 0,
                torch.ones_like(self._guarded_bypass_cooldown),
                torch.zeros_like(self._guarded_bypass_cooldown),
            ))
            self._guarded_bypass_cooldown.clamp_(min=0)

    def _planner_step_update(self):
        # 上层地图更新不再直接读取所有搜索体位置，而是通过 _process_upper_layer_communication 中的 uplink 完成。
        self._set_pse_planner_context()
        if self.step_count % self.planner_step_update_interval == 0:
            self._update_pse_belief(force_detection=False)
        if not self.use_upper_comm:
            if self.step_count % self.planner_step_update_interval != 0:
                return
            self.map_module.update_from_searcher_positions(
                self._agent_pos[: self.n_search],
                apply_decay=False,
                suppress_only=self.planner_step_update_suppress_only,
                sensor_ranges=self._sensor_range[: self.n_search],
            )

    def _sample_diverse_waypoint(self, agent_id, reserved_positions=None):
        reserved_positions = reserved_positions or []
        current = self._agent_pos[agent_id]
        best_p = None; best_score = None
        reserved_t = torch.stack([r if torch.is_tensor(r) else self._vec(r) for r in reserved_positions], dim=0) if reserved_positions else None
        for _ in range(self.diverse_fallback_tries):
            p = self._sample_free_point(margin=1.0)
            travel_dist = torch.norm(p - current)
            travel_score = torch.exp(-0.5 * ((travel_dist - 6.5) / 3.2) ** 2)
            if reserved_t is not None:
                sep = torch.min(torch.norm(reserved_t - p.unsqueeze(0), dim=1))
                if sep < self.planner_min_waypoint_separation: continue
                sep_score = torch.clamp(sep / max(self.planner_min_waypoint_separation, self.eps), max=3.0)
            else:
                sep_score = self._scalar(1.0)
            score = 0.65 * sep_score + 0.35 * travel_score
            if best_score is None or float(score.item()) > float(best_score.item()):
                best_score, best_p = score, p
        return best_p if best_p is not None else self._sample_free_point(margin=1.0)

    def _choose_next_search_waypoint(self, agent_id, reserved_positions=None, current_pos=None):
        reserved_positions = reserved_positions or []
        self._set_pse_planner_context()
        planner_current = self._agent_pos[agent_id] if current_pos is None else current_pos
        if torch.rand((), dtype=self.dtype, device=self.device).item() < self.diverse_fallback_prob:
            # diverse fallback is a local safety behavior, so it uses the real local position.
            wp = self._sample_diverse_waypoint(agent_id, reserved_positions=reserved_positions)
            self.map_module.register_waypoint_claim(wp)
            return wp
        return self.map_module.sample_next_waypoint(
            agent_id=agent_id,
            current_pos=planner_current,
            reserved_positions=reserved_positions,
        )

    def _upper_link_quality_at(self, pos, payload_bits=None):
        pos_t = pos.to(device=self.device, dtype=self.dtype).reshape(3) if torch.is_tensor(pos) else self._vec(pos)
        dist = torch.norm(pos_t - self.upper_node_pos)
        return self._link_success_prob(
            dist,
            reliable_range=self.upper_comm_reliable_range,
            max_range=self.upper_comm_max_range,
            base_loss=self.upper_comm_base_loss,
            attenuation=self.upper_comm_attenuation,
            payload_bits=payload_bits,
        )

    def _record_upper_contact_success(self, agent_id):
        agent_id = int(agent_id)
        if agent_id < 0 or agent_id >= self.n_agents:
            return
        if agent_id < self.n_search and bool(self._upper_island_active[agent_id].item()):
            self._upper_island_active[agent_id] = False
            self._upper_reconnect_active[agent_id] = False
            self._upper_reconnect_start_step[agent_id] = -1
        self._upper_last_contact_step[agent_id] = int(self.step_count)

    def _update_upper_role_topology_stats(self):
        if not self.use_upper_comm:
            self.last_upper_role_topology_density = 0.0
            self.last_upper_role_active_density = 0.0
            self.last_upper_role_physical_density = 0.0
            return
        possible = float(max(1, 2 * self.n_search + 1))
        role_edges = 0.0
        physical_edges = 0.0
        active_edges = 0.0
        for i in range(self.n_search):
            reachable = bool(self._upper_distance_to_agent(i).item() <= self.upper_comm_max_range)
            physical_edges += 2.0 if reachable else 0.0
            up_active = bool(self._pending_upper_upload[i].item())
            down_active = bool(self._waiting_upper_waypoint[i].item()) and (not self.task_found)
            role_edges += float(up_active) + float(down_active)
            active_edges += (float(up_active) + float(down_active)) if reachable else 0.0
        exec_reachable = bool(self._upper_distance_to_agent(self.executor_idx).item() <= self.upper_comm_max_range)
        physical_edges += 1.0 if exec_reachable else 0.0
        task_down_active = bool(self._upper_task_known and (not self.executor_target_assigned))
        role_edges += float(task_down_active)
        active_edges += float(task_down_active) if exec_reachable else 0.0
        self.last_upper_role_topology_density = float(role_edges / possible)
        self.last_upper_role_physical_density = float(physical_edges / possible)
        self.last_upper_role_active_density = float(active_edges / possible)

    def _update_upper_island_state(self):
        if not self.use_upper_comm:
            self._upper_island_active[: self.n_search] = False
            self._upper_reconnect_active[: self.n_search] = False
            self._upper_reconnect_start_step[: self.n_search] = -1
            self.last_island_duration = 0.0
            self.last_island_agent_ratio = 0.0
            self.last_island_count = 0.0
            self.last_avg_reconnect_time = 0.0
            return
        active_durations = []
        for i in range(self.n_search):
            contact_age = max(0, int(self.step_count) - int(self._upper_last_contact_step[i].item()))
            if self._is_upper_island(i):
                self._upper_island_active[i] = True
                active_durations.append(float(contact_age))
            else:
                self._upper_island_active[i] = False
                self._upper_reconnect_active[i] = False
                self._upper_reconnect_start_step[i] = -1
        self.last_island_count = float(len(active_durations))
        self.last_island_agent_ratio = float(len(active_durations) / max(1, self.n_search))
        self.last_island_duration = float(np.mean(active_durations)) if active_durations else 0.0
        self.last_avg_reconnect_time = 0.0

    def _is_upper_island(self, agent_id):
        if not self.use_upper_comm:
            return False
        agent_id = int(agent_id)
        if agent_id < 0 or agent_id >= self.n_search:
            return False
        has_upper_need = bool(
            self._pending_upper_upload[agent_id].item()
            or self._waiting_upper_waypoint[agent_id].item()
            or self._using_local_fallback[agent_id].item()
        )
        contact_age = max(0, int(self.step_count) - int(self._upper_last_contact_step[agent_id].item()))
        return bool(has_upper_need and contact_age >= int(self.reconnect_island_timeout_steps))

    def _sample_reconnect_lane_waypoint(self, agent_id, reserved_positions=None):
        return None, 0.0

    def _upper_link_probability(self, agent_id, payload_bits=None, direction="up"):
        dist = self._upper_distance_to_agent(agent_id)
        return self._link_success_prob(
            dist,
            reliable_range=self.upper_comm_reliable_range,
            max_range=self.upper_comm_max_range,
            base_loss=self.upper_comm_base_loss,
            attenuation=self.upper_comm_attenuation,
            payload_bits=payload_bits,
        )

    def _upper_payload_bits_for_uplink(self, agent_id, reason="state"):
        if bool(self._agent_task_known[int(agent_id)].item()):
            return self.upper_task_payload_bits
        if str(reason).startswith("state"):
            return self.upper_state_payload_bits
        return self.upper_map_payload_bits

    def _upper_payload_bits_for_downlink(self, kind):
        return self.upper_task_payload_bits if str(kind) == "task" else self.upper_waypoint_payload_bits

    def _try_upper_link(self, agent_id, direction, payload_bits=None, rate_bps=None):
        if not self.use_upper_comm:
            return True, 0, 1.0, 0
        agent_id = int(agent_id)
        dist = self._upper_distance_to_agent(agent_id)
        reachable = bool(dist.item() <= self.upper_comm_max_range)
        if reachable:
            self._upper_comm_attempts["reachable"] += 1
        key = "up" if direction == "up" else "down"
        dir_idx = 0 if key == "up" else 1
        self._upper_comm_attempts[key] += 1

        p = self._upper_link_probability(agent_id, payload_bits=payload_bits, direction=direction)
        state = int(self._upper_link_state[agent_id, dir_idx].item())
        if self.use_burst_comm:
            p = p * self.burst_state_pdr[state]
        success = bool(torch.rand((), dtype=self.dtype, device=self.device).item() < float(p.item()))
        if success:
            self._upper_comm_attempts[f"{key}_success"] += 1
            delay_steps = self._upper_delay_steps(dist, payload_bits=payload_bits, rate_bps=rate_bps or self.upper_rate_bps)
            self._upper_delay_sum += float(delay_steps)
            self._upper_delay_count += 1
            self._record_upper_contact_success(agent_id)
            return True, delay_steps, float(torch.clamp(p, 0.0, 1.0).item()), state
        return False, 0, float(torch.clamp(p, 0.0, 1.0).item()), state

    def _update_upper_belief_from_payload(self, payload):
        i = int(payload["agent_id"])
        pos = payload["pos"].clone()
        if self.upper_belief_pos_noise_std > 0.0:
            pos = pos + torch.randn_like(pos) * self.upper_belief_pos_noise_std
        self._upper_belief_pos[i] = torch.clamp(pos, min=self._lower_bound, max=self.space_size)
        self._upper_belief_vel[i] = payload.get("vel", torch.zeros(3, dtype=self.dtype, device=self.device)).clone()
        self._upper_belief_step[i] = int(payload.get("source_step", self.step_count))
        self._upper_belief_valid[i] = True

    def _get_upper_planner_pos(self, agent_id):
        agent_id = int(agent_id)
        if (not self.use_upper_belief) or (not bool(self._upper_belief_valid[agent_id].item())):
            return self._agent_pos[agent_id]
        age_steps = max(0, int(self.step_count - int(self._upper_belief_step[agent_id].item())))
        est = self._upper_belief_pos[agent_id] + self._upper_belief_vel[agent_id] * (age_steps * self.dt)
        return torch.clamp(est, min=self._lower_bound, max=self.space_size)

    def _schedule_upper_uplink(self, agent_id, reason="state"):
        payload_bits = self._upper_payload_bits_for_uplink(agent_id, reason)
        success, delay_steps, quality, state = self._try_upper_link(
            agent_id,
            direction="up",
            payload_bits=payload_bits,
            rate_bps=self.upper_rate_bps,
        )
        self.last_upper_comm_energy += self._add_comm_energy(
            sender_id=int(agent_id),
            receiver_id=None,
            payload_bits=payload_bits,
            rate_bps=self.upper_rate_bps,
            tx=True,
            rx=False,
        )
        if not success:
            return False
        payload = {
            "agent_id": int(agent_id),
            "pos": self._agent_pos[int(agent_id)].clone(),
            "vel": self._agent_vel[int(agent_id)].clone(),
            "sensor_range": float(self._sensor_range[int(agent_id)].item()) if int(agent_id) < self.n_search else 0.0,
            "known_task": bool(self._agent_task_known[int(agent_id)].item()),
            "task_est": self._agent_task_est[int(agent_id)].clone() if bool(self._agent_task_known[int(agent_id)].item()) else None,
            "reason": reason,
            "payload_bits": int(payload_bits),
            "quality": float(quality),
            "burst_state": int(state),
            "source_step": int(self.step_count),
        }
        self._upper_uplink_queue.append((self.step_count + delay_steps, payload))
        return True

    def _schedule_upper_downlink(self, agent_id, kind, payload):
        payload_bits = self._upper_payload_bits_for_downlink(kind)
        success, delay_steps, quality, state = self._try_upper_link(
            agent_id,
            direction="down",
            payload_bits=payload_bits,
            rate_bps=self.upper_rate_bps,
        )
        self.last_upper_comm_energy += self._add_comm_energy(
            sender_id=None,
            receiver_id=int(agent_id),
            payload_bits=payload_bits,
            rate_bps=self.upper_rate_bps,
            tx=False,
            rx=True,
        )
        if not success:
            return False
        wrapped = {
            "data": payload.clone(),
            "payload_bits": int(payload_bits),
            "quality": float(quality),
            "burst_state": int(state),
            "source_step": int(self.step_count),
        }
        self._upper_downlink_queue.append((self.step_count + delay_steps, int(agent_id), str(kind), wrapped))
        return True

    def _deliver_upper_uplinks(self):
        if not self._upper_uplink_queue:
            return
        remaining = []
        delivered_ages = []
        for deliver_step, payload in self._upper_uplink_queue:
            if deliver_step > self.step_count:
                remaining.append((deliver_step, payload))
                continue
            i = int(payload["agent_id"])
            delivered_ages.append(max(0, self.step_count - int(payload.get("source_step", self.step_count))))
            self._update_upper_belief_from_payload(payload)
            if i < self.n_search:
                confidence = float(np.clip(payload.get("quality", 1.0), 0.0, 1.0))
                self.last_map_update_reliability = confidence
                if hasattr(self.map_module, "apply_uploaded_update"):
                    self.map_module.apply_uploaded_update(
                        center=payload["pos"].reshape(3),
                        sensor_range=float(payload["sensor_range"]),
                        confidence=confidence,
                    )
                else:
                    self.map_module.update_from_searcher_positions(
                        payload["pos"].reshape(1, 3),
                        apply_decay=False,
                        suppress_only=self.planner_step_update_suppress_only,
                        sensor_ranges=torch.as_tensor([payload["sensor_range"]], dtype=self.dtype, device=self.device),
                    )
                self.last_upper_map_uploads += 1
            if bool(payload.get("known_task", False)) and payload.get("task_est") is not None:
                target_est = torch.clamp(payload["task_est"].to(device=self.device, dtype=self.dtype).reshape(3), min=self._lower_bound, max=self.space_size)
                self._upper_task_known = True
                self._upper_task_est.copy_(target_est)
                self.last_target_fusion_reliability = float(np.clip(payload.get("quality", 1.0), 0.0, 1.0))
                if self.use_target_belief_memory:
                    self._record_target_belief(
                        target_est,
                        source_id=i,
                        confidence=self.last_target_fusion_reliability,
                        reason="upper_target_message",
                    )
            if i < self.n_search and bool(self._waiting_upper_waypoint[i].item()) and (not self.task_found):
                reserved = [self._search_waypoints[j] for j in range(self.n_search) if j != i]
                planner_pos = self._get_upper_planner_pos(i)
                wp = self._choose_next_search_waypoint(agent_id=i, reserved_positions=reserved, current_pos=planner_pos)
                self._schedule_upper_downlink(i, "waypoint", wp)
        self._upper_uplink_queue = remaining
        if delivered_ages:
            self.last_upper_msg_age = float(np.mean(delivered_ages))

    def _process_quarantine_buffer(self):
        self.last_quarantine_release_count = 0.0
        self.last_quarantine_expire_count = 0.0
        self.last_quarantine_active_count = 0.0
        self.last_quarantine_count = 0.0

    def _deliver_upper_downlinks(self):
        if not self._upper_downlink_queue:
            return
        remaining = []
        delivered_ages = []
        for deliver_step, agent_id, kind, wrapped in self._upper_downlink_queue:
            if deliver_step > self.step_count:
                remaining.append((deliver_step, agent_id, kind, wrapped))
                continue
            payload = wrapped["data"]
            delivered_ages.append(max(0, self.step_count - int(wrapped.get("source_step", self.step_count))))
            if kind == "waypoint" and agent_id < self.n_search and (not self.task_found):
                self._search_waypoints[agent_id] = payload
                self.total_waypoints_per_agent[agent_id] += 1
                self.current_target_arrived[agent_id] = False
                self._waiting_upper_waypoint[agent_id] = False
                self._upper_wait_steps[agent_id] = 0
                self._using_local_fallback[agent_id] = False
                self.last_upper_waypoint_downlinks += 1
            elif kind == "task" and agent_id == self.executor_idx:
                target_est = torch.clamp(payload.reshape(3), min=self._lower_bound, max=self.space_size)
                if self.use_target_belief_memory:
                    self._record_target_belief(
                        target_est,
                        source_id=-1,
                        confidence=float(wrapped.get("quality", 0.8)),
                        reason="upper_target_downlink",
                    )
                self._agent_task_known[agent_id] = True
                self._agent_task_est[agent_id].copy_(target_est)
                self._target_belief_soft_executor_assigned = False
                if not self.executor_target_assigned:
                    self.total_waypoints_per_agent[self.executor_idx] += 1
                    self.executor_target_assigned = True
                    self.current_target_arrived[self.executor_idx] = False
                    self._executor_received_target_step = int(self.step_count)
                    if self._found_step is not None:
                        self.last_handoff_delay = float(self._executor_received_target_step - self._found_step)
                self.last_upper_target_downlinks += 1
        self._upper_downlink_queue = remaining
        if delivered_ages:
            self.last_upper_msg_age = float(np.mean(delivered_ages))

    def _sample_prediction_guided_fallback_from_pse_candidates(
        self,
        agent_id,
        current,
        pred_t,
        reserved_t=None,
        reserved_positions=None,
    ):
        return None

    def _sample_prediction_guided_fallback_waypoint(self, agent_id, reserved_positions=None):
        return None

    def _apply_local_fallback_waypoints(self):
        if self.task_found:
            self.last_pred_fallback_ratio = 0.0
            return
        fallback_attempts = 0
        for i in range(self.n_search):
            if not bool(self._waiting_upper_waypoint[i].item()):
                continue
            self._upper_wait_steps[i] += 1
            if int(self._upper_wait_steps[i].item()) < self.upper_wait_timeout_steps:
                continue
            reserved = [self._search_waypoints[j] for j in range(self.n_search) if j != i]
            fallback_attempts += 1
            self._search_waypoints[i] = self._sample_diverse_waypoint(agent_id=i, reserved_positions=reserved)
            self.total_waypoints_per_agent[i] += 1
            self.current_target_arrived[i] = False
            self._waiting_upper_waypoint[i] = False
            self._upper_wait_steps[i] = 0
            self._using_local_fallback[i] = True
            self._pending_upper_upload[i] = True
        self.last_pred_fallback_ratio = 0.0
        self.last_reconnect_lane_score = 0.0

    def _publish_upper_attempt_stats(self):
        attempts = getattr(self, "_upper_comm_attempts", {}) or {}
        up = int(attempts.get("up", 0))
        down = int(attempts.get("down", 0))
        up_success = int(attempts.get("up_success", 0))
        down_success = int(attempts.get("down_success", 0))

        self.last_upper_up_attempts = float(up)
        self.last_upper_down_attempts = float(down)
        self.last_upper_up_successes = float(up_success)
        self.last_upper_down_successes = float(down_success)
        self.last_upper_total_attempts = float(up + down)
        self.last_upper_total_successes = float(up_success + down_success)
        self.last_upper_has_attempt = 1.0 if (up + down) > 0 else 0.0
        self.last_upper_uplink_has_attempt = 1.0 if up > 0 else 0.0
        self.last_upper_downlink_has_attempt = 1.0 if down > 0 else 0.0

        # Keep the legacy numeric fields compatible with old training logs.
        # The *_valid fields use NaN to mean this direction had no attempt, and
        # get_tail_risk_info uses attempt counts so no-attempt is not a failure.
        self.last_upper_uplink_success_rate = float(up_success / max(1, up))
        self.last_upper_downlink_success_rate = float(down_success / max(1, down))
        self.last_upper_uplink_success_rate_valid = float(up_success / max(1, up)) if up > 0 else float("nan")
        self.last_upper_downlink_success_rate_valid = float(down_success / max(1, down)) if down > 0 else float("nan")

    def _process_upper_layer_communication(self):
        self.last_upper_map_uploads = 0
        self.last_upper_waypoint_downlinks = 0
        self.last_upper_target_downlinks = 0
        self.last_upper_comm_energy = 0.0
        self.last_upper_msg_age = 0.0
        self.last_reconnect_count = 0.0
        self.last_reconnect_success_count = 0.0
        self.last_avg_reconnect_time = 0.0
        self.last_reconnect_lane_score = 0.0
        self._reconnect_time_sum_current = 0.0
        self._reconnect_time_count_current = 0
        self._upper_comm_attempts = {"up": 0, "down": 0, "up_success": 0, "down_success": 0, "reachable": 0}
        self._upper_delay_sum = 0.0
        self._upper_delay_count = 0
        self._advance_upper_link_states()
        self.last_pred_fallback_ratio = 0.0
        if not self.use_upper_comm:
            self._publish_upper_attempt_stats()
            self._process_quarantine_buffer()
            self._update_upper_role_topology_stats()
            self._update_upper_island_state()
            return

        self._deliver_upper_uplinks()
        self._deliver_upper_downlinks()

        if (self.step_count % self.upper_map_update_interval) == 0 and not self.task_found:
            self._pending_upper_upload[: self.n_search] = True

        for i in range(self.n_search):
            if bool(self._pending_upper_upload[i].item()):
                if self._schedule_upper_uplink(i, reason="periodic_or_arrival"):
                    self._pending_upper_upload[i] = False

        if self._upper_task_known and (not self.executor_target_assigned):
            self._schedule_upper_downlink(self.executor_idx, "task", self._upper_task_est)

        self._deliver_upper_uplinks()
        self._deliver_upper_downlinks()
        self._process_quarantine_buffer()
        self._update_upper_role_topology_stats()
        self._update_upper_island_state()
        self._apply_local_fallback_waypoints()
        self._update_upper_role_topology_stats()
        self._update_upper_island_state()

        self._publish_upper_attempt_stats()
        up = self._upper_comm_attempts["up"]
        down = self._upper_comm_attempts["down"]
        self.last_upper_comm_density = self._upper_comm_attempts["reachable"] / max(1, up + down)
        self.last_upper_avg_delay_steps = self._upper_delay_sum / max(1, self._upper_delay_count)
        self.last_upper_good_state_ratio = float((self._upper_link_state == 0).float().mean().item())
        if torch.any(self._upper_belief_valid):
            ages = (self.step_count - self._upper_belief_step[self._upper_belief_valid].to(dtype=torch.int32)).to(dtype=self.dtype)
            self.last_upper_belief_age = float(torch.clamp(ages, min=0).float().mean().item())
        else:
            self.last_upper_belief_age = 0.0
        self.last_local_fallback_ratio = float(self._using_local_fallback[: self.n_search].float().mean().item())
        self.last_comm_energy_total = float(self.last_comm_energy.sum().item())

    def _search_spread_reward(self):
        search_pos = self._agent_pos[: self.n_search]
        dists = torch.cdist(search_pos, search_pos); dists.fill_diagonal_(1e6)
        min_dist = torch.min(dists, dim=1).values
        spread = torch.clamp((min_dist - self.safe_dist) / max(self.safe_dist, 1e-6), 0.0, 1.0)
        return self.search_spread_reward_gain * spread

    def _update_search_waypoint_events(self, current_nav_distances, speeds):
        if self.task_found:
            self.hold_counters[: self.n_search] = 0
            return False
        updated = False
        search_in_arrive = (current_nav_distances[: self.n_search] < self.search_arrive_eps) & (~self.current_target_arrived[: self.n_search])
        for i in torch.nonzero(search_in_arrive, as_tuple=False).flatten().tolist():
            self.just_reached_waypoint[i] = True
            self.just_held_target[i] = False
            self.waypoint_reached_counts[i] += 1
            self.hold_counters[i] = 0
            self.current_target_arrived[i] = True
            if self.use_upper_comm:
                self._pending_upper_upload[i] = True
                self._waiting_upper_waypoint[i] = True
                self._upper_wait_steps[i] = 0
            else:
                self.map_module.register_visited_point(self._agent_pos[i], suppress_only=False)
                reserved = [self._search_waypoints[j] for j in range(self.n_search) if j != i]
                self._search_waypoints[i] = self._choose_next_search_waypoint(agent_id=i, reserved_positions=reserved)
                self.total_waypoints_per_agent[i] += 1
                self.current_target_arrived[i] = False
                updated = True
        return updated

    def _update_executor_hold_events(self, current_nav_distances, speeds):
        i = self.executor_idx
        stable_speed = speeds[i] < self.hold_speed_thresh
        if (not self.task_found) or (not self.executor_target_assigned):
            near_wait = current_nav_distances[i] < self.executor_hold_radius
            if bool(near_wait) and (not bool(self.current_target_arrived[i])):
                self.waypoint_reached_counts[i] += 1; self.current_target_arrived[i] = True
            if (not self.executor_wait_held) and bool(near_wait and stable_speed):
                self.hold_counters[i] += 1
            elif not self.executor_wait_held:
                self.hold_counters[i] = 0
            if (not self.executor_wait_held) and int(self.hold_counters[i].item()) >= self.executor_hold_steps:
                self.executor_wait_held = True; self.just_held_target[i] = True
                self.hold_success_counts[i] += 1; self.hold_counters[i] = self.executor_hold_steps
                self._executor_wait_hold_event = True
            return
        near_task = current_nav_distances[i] < self.executor_arrive_eps
        if bool(near_task) and (not bool(self.current_target_arrived[i])):
            self.waypoint_reached_counts[i] += 1; self.current_target_arrived[i] = True
        if bool(near_task and stable_speed):
            self.hold_counters[i] += 1
        else:
            self.hold_counters[i] = 0
        if (not self.mission_complete) and int(self.hold_counters[i].item()) >= self.executor_hold_steps:
            self.just_held_target[i] = True; self.hold_success_counts[i] += 1
            self.mission_complete = True; self.agent_finished[i] = True; self._mission_complete_event = True

    def _step_mission(self, actions):
        self._found_event = False; self._mission_complete_event = False; self._executor_wait_hold_event = False
        self.just_reached_waypoint.zero_(); self.just_held_target.zero_()
        self.last_comm_energy.zero_()
        self._update_pse_executor_standby()
        self._update_nav_targets()
        prev_nav_distances = self._compute_nav_distances()
        self._apply_agent_dynamics(actions)
        self._planner_step_update()
        self._maybe_detect_task()
        self._advance_target_belief_memory()
        self._update_nav_targets()
        current_nav_distances = self._compute_nav_distances()
        speeds = torch.norm(self._agent_vel, dim=1)
        if self._update_search_waypoint_events(current_nav_distances, speeds):
            self._update_nav_targets(); current_nav_distances = self._compute_nav_distances()
        self._process_upper_layer_communication()
        self._update_nav_targets(); current_nav_distances = self._compute_nav_distances()
        self._update_executor_hold_events(current_nav_distances, speeds)
        # 更新下层智能体间通信后再计算 reward，使本步通信能耗进入奖励。
        self._update_communication_state()
        rewards = self._calculate_mission_rewards(prev_nav_distances)
        obs = self._get_obs()
        dones = [(self.mission_complete or (self.step_count >= self.max_steps))] * self.n_agents
        self._prev_nav_distances = self._compute_nav_distances()
        self._prev_acc.copy_(self._agent_acc)
        self._prev_coverage_ratio = self._current_coverage_ratio_internal()
        self._prev_search_task_min_dist = self._current_search_task_min_dist()
        self._sync_pse_diagnostics()
        return self._obs_to_public(obs), self._rewards_to_public(rewards), dones

    def _common_reward_terms_all(self):
        rewards = torch.zeros(self.n_agents, dtype=self.dtype, device=self.device)
        pairwise = torch.cdist(self._agent_pos, self._agent_pos)
        sep_penalty = torch.clamp(self.safe_dist - pairwise, min=0.0); sep_penalty.fill_diagonal_(0.0)
        rewards -= self.sep_penalty_k * sep_penalty.sum(dim=1)
        rewards += self._collision_flags.to(self.dtype) * self.collision_penalty
        rewards -= self.time_penalty
        rewards -= self.lambda_a * self._energy_coeff * torch.sum(self._agent_acc ** 2, dim=1)
        rewards -= self.lambda_da * torch.sum((self._agent_acc - self._prev_acc) ** 2, dim=1)
        if self.use_comm_energy and self.lambda_comm_energy > 0.0:
            rewards -= self.lambda_comm_energy * self.last_comm_energy
        if self.reward_profile == "original" and self.use_residual_prior and self.residual_penalty > 0.0:
            rewards -= self.residual_penalty * torch.sum(self._last_residual_acc ** 2, dim=1)
        return rewards

    def _apply_robust_residual_rewards_v2(self, rewards):
        self._reset_robust_residual_reward_diagnostics_v2()
        if self.reward_profile not in ("robust_residual_v2", "residual_point_v3"):
            return rewards

        additions = torch.zeros_like(rewards)
        exec_i = int(self.executor_idx)
        search_idx = slice(0, self.n_search)
        exec_knows_task = False
        if hasattr(self, "executor_target_assigned"):
            exec_knows_task = exec_knows_task or bool(self.executor_target_assigned)
        if hasattr(self, "_agent_task_known"):
            exec_knows_task = exec_knows_task or bool(self._agent_task_known[exec_i].item())
        if self.reward_profile == "residual_point_v3":
            self.last_exec_knows_task_v3 = 1.0 if exec_knows_task else 0.0
            motion_gate = self._compute_v3_target_motion_gate(exec_knows_task)
            self.last_residual_hybrid_soft_gate = float(motion_gate)
            self.last_reward_motion_gated_by_comm_v3 = float(1.0 - motion_gate) if bool(self.task_found) else 0.0
        else:
            motion_gate = 1.0

        risk = self._compute_residual_risk_score_v2()
        self.last_residual_risk_score_v2 = risk
        penalty_eff = (
            self.low_risk_residual_penalty_v2 * (1.0 - risk)
            + self.high_risk_residual_penalty_v2 * risk
        )
        risk_pen = -penalty_eff * torch.sum(self._last_residual_acc ** 2, dim=1)
        additions += risk_pen
        self.last_reward_risk_aware_residual_penalty_v2 = float(risk_pen.mean().item())

        residual_norm = torch.norm(self._last_residual_acc, dim=1)
        prior_norm = torch.norm(self._last_prior_acc_v2, dim=1)
        denom = residual_norm * prior_norm + 1e-6
        cos = torch.sum(self._last_residual_acc * self._last_prior_acc_v2, dim=1) / denom
        valid = (residual_norm > 1e-6) & (prior_norm > 1e-6)
        cos = torch.where(valid, cos, torch.zeros_like(cos))
        anti_pen = -self.anti_prior_penalty_gain_v2 * torch.clamp(-cos, min=0.0) * residual_norm
        additions += anti_pen
        self.last_reward_anti_prior_penalty_v2 = float(anti_pen.mean().item())
        self.last_residual_prior_cosine_v2 = float(cos[valid].mean().item()) if torch.any(valid) else 0.0

        if self.task_found:
            exec_dist_t = None
            exec_dist = float("inf")
            exec_speed_t = None
            exec_speed = float("inf")
            exec_dist_t = torch.norm(self._agent_pos[exec_i] - self._task_target)
            exec_dist = float(exec_dist_t.item())
            exec_speed_t = torch.norm(self._agent_vel[exec_i])
            exec_speed = float(exec_speed_t.item())

            if self.prev_exec_task_dist_v2 is not None:
                val = float(motion_gate) * self.exec_task_progress_gain_v2 * (float(self.prev_exec_task_dist_v2) - exec_dist)
                additions[exec_i] += val
                self.last_reward_exec_task_progress_v2 = float(val)

            dense_gate = self._compute_v3_executor_dense_gate(motion_gate)

            if exec_dist < self.near_target_speed_radius_v2:
                val = -dense_gate * self.near_target_speed_penalty_gain_v2 * float((exec_speed_t ** 2).item())
                additions[exec_i] += val
                self.last_reward_near_target_speed_v2 = float(val)

            if exec_dist < self.executor_arrive_eps and exec_speed < self.hold_speed_thresh:
                self.executor_hold_counter_v2 = min(int(self.executor_hold_counter_v2) + 1, int(self.executor_hold_steps))
            else:
                self.executor_hold_counter_v2 = 0
            hold_progress = self.executor_hold_counter_v2 / max(1.0, float(self.executor_hold_steps))
            if hold_progress > 0.0:
                val = dense_gate * self.hold_dense_reward_gain_v2 * hold_progress
                additions[exec_i] += val
                self.last_reward_hold_dense_v2 = float(val)

            if not self._target_assigned_bonus_given_v2:
                nav_dist_to_task = float(torch.norm(self._nav_targets[exec_i] - self._task_target).item())
                has_task_est = bool(self.executor_target_assigned) or bool(self._agent_task_known[exec_i].item())
                if has_task_est and nav_dist_to_task <= 3.0:
                    additions[exec_i] += dense_gate * self.target_assigned_bonus_exec_v2
                    if self.n_search > 0:
                        additions[search_idx] += self.target_assigned_bonus_search_v2
                    self.last_reward_target_assigned_v2 = float(
                        dense_gate * self.target_assigned_bonus_exec_v2 + self.target_assigned_bonus_search_v2 * self.n_search
                    )
                    self._target_assigned_bonus_given_v2 = True

            handoff_delay = getattr(self, "last_handoff_delay", float("nan"))
            if np.isfinite(handoff_delay):
                val = -self.handoff_delay_penalty_gain_v2 * min(float(handoff_delay) / max(1.0, float(self.max_steps)), 1.0)
                additions[exec_i] += val
                self.last_reward_handoff_delay_penalty_v2 = float(val)

            belief_age = getattr(self, "last_upper_belief_age", 0.0)
            val = -self.belief_age_penalty_gain_v2 * min(float(belief_age) / self.belief_age_penalty_norm_v2, 1.0)
            additions[exec_i] += val
            self.last_reward_belief_age_penalty_v2 = float(val)

            if exec_dist < self.near_target_residual_penalty_radius_v2:
                val = -dense_gate * self.near_target_residual_penalty_gain_v2 * float(torch.sum(self._last_residual_acc[exec_i] ** 2).item())
                additions[exec_i] += val
                self.last_reward_near_target_residual_penalty_v2 = float(val)

        self.last_reward_robust_residual_total_v2 = float(additions.sum().item())
        return rewards + additions

    def _calculate_mission_rewards(self, prev_nav_distances):
        rewards = self._common_reward_terms_all()
        current_nav_distances = self._compute_nav_distances()
        delta_nav = prev_nav_distances - current_nav_distances
        search_idx = slice(0, self.n_search); exec_i = self.executor_idx
        if not self.task_found:
            rewards[search_idx] += self._progress_gain[search_idx] * delta_nav[search_idx]
            just_mask = self.just_reached_waypoint[: self.n_search]
            if torch.any(just_mask): rewards[: self.n_search][just_mask] += self._waypoint_bonus[: self.n_search][just_mask]
            coverage_delta = max(0.0, self._current_coverage_ratio_internal() - self._prev_coverage_ratio)
            if coverage_delta > 0.0: rewards[search_idx] += self.coverage_reward_gain * coverage_delta
            rewards[search_idx] += self._search_spread_reward()
            rewards[exec_i] += 8.0 * delta_nav[exec_i]
            if self._executor_wait_hold_event: rewards[exec_i] += self.executor_hold_bonus
        else:
            if self._found_event:
                rewards[search_idx] += self.team_find_bonus
                if 0 <= self.finder_idx < self.n_search:
                    rewards[self.finder_idx] += self._detect_bonus[self.finder_idx] + self.finder_extra_bonus
            if self.executor_target_assigned:
                rewards[exec_i] += self._progress_gain[exec_i] * delta_nav[exec_i]
                if self._mission_complete_event:
                    rewards[exec_i] += self.mission_complete_bonus
            else:
                # 执行体尚未收到上层目标时，只奖励保持/接近等待点，避免目标信息瞬时泄漏。
                rewards[exec_i] += 4.0 * delta_nav[exec_i]
            rewards[: self.n_search] = torch.where(self.agent_finished[: self.n_search], torch.zeros_like(rewards[: self.n_search]), rewards[: self.n_search])
        rewards = self._apply_robust_residual_rewards_v2(rewards)
        if self.reward_profile == "residual_point_v3":
            rewards = rewards + self._compute_point_control_shaping_v3()
        elif self.reward_profile != "residual_point_v3":
            self._reset_point_control_reward_diagnostics_v3()
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        if self.reward_profile in ("robust_residual_v2", "residual_point_v3") and self.task_found:
            if self.reward_profile == "residual_point_v3":
                self.prev_exec_task_dist_v2 = float(torch.norm(self._agent_pos[exec_i] - self._task_target).item())
            else:
                self.prev_exec_task_dist_v2 = float(torch.norm(self._agent_pos[exec_i] - self._task_target).item())
        return torch.tanh(rewards / self.reward_scale)

    def _compute_nav_distances(self): return torch.norm(self._agent_pos - self._nav_targets, dim=1)

    def _points_inside_obstacles(self, points):
        if self._obstacle_lower is None:
            return torch.zeros(points.shape[:-1], dtype=torch.bool, device=self.device)
        p = points.unsqueeze(-2)
        inside = ((p >= self._obstacle_lower) & (p <= self._obstacle_upper)).all(dim=-1)
        return inside.any(dim=-1)

    def _nearest_obstacle_distance(self, points):
        if self._obstacle_lower is None:
            return torch.full(points.shape[:-1], 10.0, dtype=self.dtype, device=self.device)
        p = points.unsqueeze(-2)
        zeros = torch.zeros_like(p)
        delta = torch.maximum(zeros, self._obstacle_lower - p) + torch.maximum(zeros, p - self._obstacle_upper)
        return torch.norm(delta, dim=-1).min(dim=-1).values

    def is_inside_obstacle(self, point):
        p = point if torch.is_tensor(point) else self._vec(point)
        return bool(self._points_inside_obstacles(p).item())

    @property
    def obs_dim(self): return int(self.base_obs_dim + (self.n_agents - 1) * self.comm_obs_per_neighbor_dim)
    @property
    def observation_space(self): return {f"agent_{i}": DummySpace(shape=(self.obs_dim,)) for i in range(self.n_agents)}
    @property
    def action_space(self): return {f"agent_{i}": DummySpace(shape=(3,)) for i in range(self.n_agents)}

    def get_tail_risk_info(self):
        link_loss = float(np.clip(getattr(self, "last_link_loss_prob", 0.0), 0.0, 1.0))
        edge_bad = float(np.clip(1.0 - getattr(self, "last_edge_weight", 1.0), 0.0, 1.0))
        lower_delay = float(np.clip(getattr(self, "last_lower_avg_delay_steps", 0.0) / max(1.0, getattr(self, "lower_max_delay_steps", 16.0)), 0.0, 1.0))
        lower_age = float(np.clip(getattr(self, "last_lower_msg_age", 0.0) / max(1.0, getattr(self, "lower_msg_ttl_steps", 24.0)), 0.0, 1.0))
        upper_rates = []
        up = float(getattr(self, "last_upper_up_attempts", 0.0))
        down = float(getattr(self, "last_upper_down_attempts", 0.0))
        up_success = float(getattr(self, "last_upper_up_successes", 0.0))
        down_success = float(getattr(self, "last_upper_down_successes", 0.0))
        if up > 0.0:
            upper_rates.append(up_success / max(1.0, up))
        if down > 0.0:
            upper_rates.append(down_success / max(1.0, down))
        # No upper communication attempt means no upper-link failure pressure for
        # this step; it should not raise comm_risk_mean or max-risk sources.
        upper_loss = 0.0 if len(upper_rates) == 0 else 1.0 - float(np.mean(upper_rates))
        upper_loss = float(np.clip(upper_loss, 0.0, 1.0))
        fallback = float(np.clip(getattr(self, "last_local_fallback_ratio", 0.0), 0.0, 1.0))
        island = float(np.clip(getattr(self, "last_island_agent_ratio", 0.0), 0.0, 1.0))
        pred_unc = float(np.clip(getattr(self, "last_pred_uncertainty", 0.0), 0.0, 1.0))
        conflict = float(np.clip(getattr(self, "last_conflict_score", 0.0), 0.0, 1.0))
        quarantine = float(np.clip(getattr(self, "last_quarantine_count", 0.0) / 4.0, 0.0, 1.0))
        reliability_bad = float(np.clip(1.0 - getattr(self, "last_avg_reliability", 1.0), 0.0, 1.0))
        critical_drop = float(np.clip(getattr(self, "last_sem_critical_drop_rate", 0.0), 0.0, 1.0))
        topology_pruned = float(np.clip(getattr(self, "last_role_pruned_ratio", 0.0), 0.0, 1.0))
        role_inactive = float(np.clip(1.0 - getattr(self, "last_role_active_density", 1.0), 0.0, 1.0))
        handover_pending = 1.0 if bool(getattr(self, "task_found", False)) and (not bool(getattr(self, "executor_target_assigned", False))) else 0.0
        handover_blocked = 0.0
        if handover_pending > 0.5:
            handover_blocked = float(np.clip(1.0 - getattr(self, "last_direct_handover_ready", 0.0), 0.0, 1.0))
        tail_score = float(np.clip(
            0.15 * link_loss + 0.12 * edge_bad + 0.10 * lower_delay + 0.08 * lower_age +
            0.12 * upper_loss + 0.08 * fallback + 0.05 * island + 0.07 * pred_unc + 0.05 * conflict +
            0.03 * quarantine + 0.03 * reliability_bad + 0.08 * critical_drop +
            0.03 * topology_pruned + 0.04 * handover_blocked + 0.04 * handover_pending +
            0.03 * role_inactive, 0.0, 1.0
        ))
        risk_components = {
            "risk_link_loss": link_loss,
            "risk_edge_bad": edge_bad,
            "risk_upper_loss": upper_loss,
            "risk_island": island,
            "risk_critical_drop": critical_drop,
            "risk_handover_blocked": handover_blocked,
            "risk_role_inactive": role_inactive,
        }
        risk_components = {k: float(np.clip(v, 0.0, 1.0)) for k, v in risk_components.items()}
        max_key = max(risk_components, key=risk_components.get)
        comm_risk_max = float(np.clip(risk_components[max_key], 0.0, 1.0))
        comm_risk_mean = float(np.mean(list(risk_components.values()))) if risk_components else 0.0

        active_keys = ["risk_link_loss", "risk_edge_bad", "risk_upper_loss", "risk_island", "risk_critical_drop"]
        if bool(getattr(self, "use_direct_handover_lane", False)):
            active_keys.append("risk_handover_blocked")
        if bool(getattr(self, "use_role_topology", False)):
            active_keys.append("risk_role_inactive")
        active_values = [risk_components[k] for k in active_keys if k in risk_components]
        active_components = {k: risk_components[k] for k in active_keys if k in risk_components}
        active_max_key = max(active_components, key=active_components.get) if active_components else max_key
        comm_risk_active_mean = float(np.mean(active_values)) if active_values else comm_risk_mean

        # comm_risk is the legacy max-risk indicator: it captures the worst active
        # communication risk, not an average loss rate. Use comm_risk_mean or
        # comm_risk_active_mean to explain average communication pressure.
        comm_risk = float(np.clip(max(link_loss, edge_bad, upper_loss, island, critical_drop, handover_blocked, role_inactive), 0.0, 1.0))
        scenario_id = int(getattr(self, "comm_scenario_id", 0))
        self.last_tail_score = tail_score
        self.last_comm_risk = comm_risk
        return {
            "tail_score": tail_score,
            "comm_risk": comm_risk,
            "comm_risk_max": comm_risk_max,
            "comm_risk_mean": comm_risk_mean,
            "comm_risk_active_mean": comm_risk_active_mean,
            "comm_risk_source": max_key.replace("risk_", "", 1),
            "comm_risk_active_source": active_max_key.replace("risk_", "", 1),
            "risk_components": risk_components,
            "upper_has_attempt": float(getattr(self, "last_upper_has_attempt", 0.0)),
            "upper_up_attempts": float(getattr(self, "last_upper_up_attempts", 0.0)),
            "upper_down_attempts": float(getattr(self, "last_upper_down_attempts", 0.0)),
            "upper_total_attempts": float(getattr(self, "last_upper_total_attempts", 0.0)),
            "upper_total_successes": float(getattr(self, "last_upper_total_successes", 0.0)),
            "scenario_id": scenario_id,
            "critical_drop": critical_drop,
            "island": island,
            "handover_pending": handover_pending,
            "handover_blocked": handover_blocked,
            "role_inactive": role_inactive,
        }

    def get_replay_comm_meta(self):
        def val(name, default=0.0, scale=1.0):
            try:
                x = float(getattr(self, name, default))
            except Exception:
                x = float(default)
            if not np.isfinite(x):
                x = 0.0
            return x / max(float(scale), 1e-6)

        graph_values = [
            val("last_comm_density"),
            val("last_comm_success_rate"),
            val("last_avg_neighbor_num"),
            val("last_lower_avg_delay_steps"),
            val("last_lower_msg_age"),
            val("last_edge_weight"),
            val("last_link_loss_prob"),
            val("last_link_snr_norm"),
            val("last_link_bandwidth_bps", scale=5000.0),
            val("last_comm_noise_mean"),
            val("last_depth_gap_mean", scale=float(self.space_size[2].item() if torch.is_tensor(self.space_size) else self.space_size[2])),
            val("last_flow_diff_mean"),
            val("last_role_topology_density"),
            val("last_role_active_density"),
            val("last_role_pruned_ratio"),
            val("last_island_agent_ratio"),
        ]
        message_values = [
            val("last_sem_target_msg_count"),
            val("last_sem_map_msg_count"),
            val("last_sem_waypoint_req_count"),
            val("last_sem_handover_msg_count"),
            val("last_sem_heartbeat_count"),
            val("last_sem_risk_alert_count"),
            val("last_sem_critical_count"),
            val("last_sem_payload_bits", scale=1000.0),
            val("last_sem_selected_count"),
            val("last_sem_critical_selected_count"),
            val("last_sem_voi_score"),
            val("last_sem_dropped_by_budget"),
            val("last_sem_critical_dropped_by_budget"),
            val("last_sem_critical_delivery_rate"),
            val("last_sem_critical_drop_rate"),
            val("last_direct_handover_ready"),
        ]
        belief_values = [
            val("last_pred_generated_ratio"),
            val("last_pred_obs_ratio"),
            val("last_pred_fallback_ratio"),
            val("last_pred_uncertainty"),
            val("last_pred_sigma"),
            val("last_pred_error"),
            val("last_repair_gain"),
            val("last_avg_reliability"),
            val("last_conflict_score"),
            val("last_belief_disagreement"),
            val("last_quarantine_count"),
            val("last_quarantine_active_count"),
            val("last_quarantine_release_count"),
            val("last_quarantine_expire_count"),
            val("last_map_update_reliability"),
            val("last_prediction_reliability"),
        ]

        def tensorize(values):
            t = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.clamp(t, -10.0, 10.0)

        return {
            "graph_meta": tensorize(graph_values),
            "message_meta": tensorize(message_values),
            "belief_meta": tensorize(belief_values),
        }


class DummySpace:
    def __init__(self, shape):
        self.shape = shape
