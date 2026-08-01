import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class PheromoneWaypointPlanner:
    """
    三维信息素/覆盖地图 + 航点规划器。

    优化点：
    1) 小网格上的覆盖/抑制更新改为展平网格张量化，避免 Python 三重循环；
    2) reserved_positions 很少时，不用 torch.cdist，直接做轻量距离计算；
    3) 缓存展平网格与 frontier，减少 reset/step 中重复构造。
    """

    def __init__(
        self,
        space_size,
        n_agents: int = 4,
        n_search: int = 3,
        executor_idx: int = 3,
        grid_size=(10, 10, 4),
        z_range=(0.8, 1.2),
        search_count_range=(2, 4),
        executor_count_range=(1, 1),
        visit_radius: int = 1,
        pheromone_decay: float = 0.992,
        suppression: float = 0.58,
        coverage_gain: float = 1.0,
        min_waypoint_separation: float = 2.0,
        pos_update_radius: int = 1,
        pos_update_suppression: float = 0.94,
        pos_update_gain: float = 0.08,
        frontier_weight: float = 0.45,
        coverage_weight: float = 0.85,
        claim_weight: float = 0.60,
        stochastic_topk: int = 6,
        stochastic_eps: float = 0.12,
        device=None,
        dtype=torch.float32,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.space_size = torch.as_tensor(space_size, dtype=dtype, device=self.device)
        self.n_agents = int(n_agents)
        self.n_search = int(n_search)
        self.executor_idx = int(executor_idx)

        if len(grid_size) == 2:
            grid_size = (int(grid_size[0]), int(grid_size[1]), 4)
        elif len(grid_size) != 3:
            raise ValueError("grid_size 必须是 (nx, ny, nz)")
        self.grid_size = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))

        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.search_count_range = tuple(int(x) for x in search_count_range)
        self.executor_count_range = tuple(int(x) for x in executor_count_range)
        self.visit_radius = int(visit_radius)
        self.pheromone_decay = float(pheromone_decay)
        self.suppression = float(suppression)
        self.coverage_gain = float(coverage_gain)
        self.min_waypoint_separation = float(min_waypoint_separation)
        self.pos_update_radius = int(pos_update_radius)
        self.pos_update_suppression = float(pos_update_suppression)
        self.pos_update_gain = float(pos_update_gain)
        self.frontier_weight = float(frontier_weight)
        self.coverage_weight = float(coverage_weight)
        self.claim_weight = float(claim_weight)
        self.stochastic_topk = max(1, int(stochastic_topk))
        self.stochastic_eps = float(max(0.0, min(1.0, stochastic_eps)))
        self.eps = 1e-6

        self.role_pref_radius = {
            0: (5.8, 2.2),
            1: (4.6, 1.9),
            2: (3.6, 1.5),
            3: (4.8, 2.8),
        }
        self._update_role_pref_z()

        self._grid_z_range = None
        self._build_grid()
        self.reset(None, [])

    def _update_role_pref_z(self):
        low_z, high_z = self.z_range
        span_z = max(high_z - low_z, 1e-3)
        self.role_pref_z = {
            0: low_z + 0.70 * span_z,
            1: low_z + 0.52 * span_z,
            2: low_z + 0.35 * span_z,
            3: low_z + 0.55 * span_z,
        }

    def set_z_range(self, z_range, rebuild_grid: bool = True):
        """Update curriculum z-range and rebuild cached 3-D grid when needed.

        The planner caches ``xyz_centers`` and ``flat_xyz_centers`` for speed.
        If training curriculum changes ``env.random_z_range`` without rebuilding
        these tensors, sampled waypoints can still come from the old depth band.
        This method keeps map waypoints, role-specific depth preferences and grid
        indices synchronized with the current environment depth range.
        """
        if z_range is None:
            return False
        z0, z1 = float(z_range[0]), float(z_range[1])
        if z1 < z0:
            z0, z1 = z1, z0
        z0 = max(0.0, min(float(self.space_size[2].item()), z0))
        z1 = max(0.0, min(float(self.space_size[2].item()), z1))
        new_range = (z0, z1)
        changed = any(abs(a - b) > 1e-6 for a, b in zip(new_range, self.z_range))
        self.z_range = new_range
        self._update_role_pref_z()
        if changed and rebuild_grid:
            self._build_grid()
        else:
            self._invalidate_cache()
        return changed

    def _build_grid(self):
        nx, ny, nz = self.grid_size
        sx, sy, _ = self.space_size.tolist()
        low_z, high_z = self.z_range

        xs = torch.linspace(sx / (2 * nx), sx - sx / (2 * nx), nx, dtype=self.dtype, device=self.device)
        ys = torch.linspace(sy / (2 * ny), sy - sy / (2 * ny), ny, dtype=self.dtype, device=self.device)
        zs = torch.linspace(low_z, high_z, nz, dtype=self.dtype, device=self.device)

        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        self.xyz_centers = torch.stack([xx, yy, zz], dim=-1)
        self.flat_xyz_centers = self.xyz_centers.reshape(-1, 3)
        self._all_valid_mask = torch.ones(self.grid_size, dtype=torch.bool, device=self.device)

        self.cell_dx = float(sx / nx)
        self.cell_dy = float(sy / ny)
        self.cell_dz = float((high_z - low_z) / max(nz - 1, 1))
        self.cell_diag = float(math.sqrt(self.cell_dx ** 2 + self.cell_dy ** 2 + self.cell_dz ** 2))
        self._grid_z_range = tuple(self.z_range)
        self._invalidate_cache()

    def reset(self, agent_positions=None, obstacles: Optional[Iterable] = None):
        if getattr(self, "_grid_z_range", None) != tuple(self.z_range):
            self._update_role_pref_z()
            self._build_grid()
        nx, ny, nz = self.grid_size
        self.pheromone = torch.ones((nx, ny, nz), dtype=self.dtype, device=self.device)
        self.coverage = torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device)
        self.claim_count = torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device)
        self.obstacles = list(obstacles) if obstacles is not None else []
        self.valid_mask = self._build_valid_mask()
        self.flat_valid_mask = self.valid_mask.reshape(-1)
        self.valid_mask_float = self.valid_mask.to(self.dtype)
        self.flat_valid_mask_float = self.flat_valid_mask.to(self.dtype)
        self.flat_valid_points = self.flat_xyz_centers[self.flat_valid_mask]

        cx = (self.xyz_centers[..., 0] / self.space_size[0] - 0.5) ** 2
        cy = (self.xyz_centers[..., 1] / self.space_size[1] - 0.5) ** 2
        low_z, high_z = self.z_range
        z_mid = 0.5 * (low_z + high_z)
        z_term = torch.abs(self.xyz_centers[..., 2] - z_mid) / max(high_z - low_z, self.eps)
        edge_bias = 0.22 * torch.sqrt(cx + cy) + 0.08 * z_term
        self.pheromone = (1.0 + edge_bias) * self.valid_mask.to(self.dtype)
        self._invalidate_cache()

        if agent_positions is not None:
            agent_positions = self._as_points(agent_positions)
            self.update_from_searcher_positions(agent_positions[: self.n_search], apply_decay=False, suppress_only=True)

    def _invalidate_cache(self):
        self._frontier_cache = None

    def _as_points(self, x):
        t = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=self.dtype, device=self.device)
        return t.to(device=self.device, dtype=self.dtype)

    def _build_valid_mask(self):
        if not self.obstacles:
            return self._all_valid_mask

        pts = self.flat_xyz_centers
        inside = torch.zeros((pts.shape[0],), dtype=torch.bool, device=self.device)
        for obs in self.obstacles:
            center = self._as_points(obs["center"])
            size = self._as_points(obs["size"])
            lower = center - size / 2.0
            upper = center + size / 2.0
            inside |= ((pts >= lower) & (pts <= upper)).all(dim=1)
        return (~inside).reshape(self.grid_size)

    def _sample_count(self, count_range: Tuple[int, int]):
        lo, hi = count_range
        lo, hi = min(lo, hi), max(lo, hi)
        if lo == hi:
            return lo
        return int(torch.randint(lo, hi + 1, (1,), device=self.device).item())

    def _grid_index_from_point(self, point):
        point = self._as_points(point)
        low_z, _ = self.z_range
        x = int(torch.clamp(torch.floor(point[0] / max(self.cell_dx, self.eps)), 0, self.grid_size[0] - 1).item())
        y = int(torch.clamp(torch.floor(point[1] / max(self.cell_dy, self.eps)), 0, self.grid_size[1] - 1).item())
        z_rel = (point[2] - low_z) / max(self.cell_dz, self.eps)
        z = int(torch.clamp(torch.round(z_rel), 0, self.grid_size[2] - 1).item())
        return x, y, z

    def _apply_negative_evidence(
        self,
        point,
        radius_m: float,
        suppression_factor: float,
        gain: float,
        claim_relief: float,
        suppress_only: bool = False,
        apply_decay: bool = False,
    ):
        point = self._as_points(point)
        radius_m = float(max(radius_m, 0.5 * min(self.cell_dx, self.cell_dy)))

        flat_pheromone = self.pheromone.reshape(-1)
        flat_coverage = self.coverage.reshape(-1)
        flat_claim_count = self.claim_count.reshape(-1)

        dist = torch.linalg.vector_norm(self.flat_xyz_centers - point.unsqueeze(0), dim=1)
        mask = (dist <= radius_m) & self.flat_valid_mask
        if torch.any(mask):
            weights = torch.clamp(1.0 - dist[mask] / max(radius_m, self.eps), min=0.0)
            flat_pheromone[mask] *= suppression_factor ** weights
            if not suppress_only:
                flat_coverage[mask] += gain * weights
                flat_claim_count[mask] = torch.clamp(flat_claim_count[mask] - claim_relief * weights, min=0.0)

        if apply_decay:
            self.pheromone.mul_(self.pheromone_decay)
        self.pheromone.mul_(self.valid_mask_float)
        self._invalidate_cache()

    def register_visited_point(self, point, suppress_only: bool = False):
        strong_radius = max(self.visit_radius * self.cell_diag, 1.2 * min(self.cell_dx, self.cell_dy))
        self._apply_negative_evidence(
            point=point,
            radius_m=strong_radius,
            suppression_factor=self.suppression,
            gain=self.coverage_gain,
            claim_relief=0.50,
            suppress_only=suppress_only,
            apply_decay=False,
        )

    def register_negative_observation(self, point, sensor_range: float, confidence: float = 1.0):
        radius_m = float(max(sensor_range, 1.2 * min(self.cell_dx, self.cell_dy)))
        local_supp = max(0.15, min(0.98, self.suppression ** max(confidence, 0.1)))
        self._apply_negative_evidence(
            point=point,
            radius_m=radius_m,
            suppression_factor=local_supp,
            gain=self.coverage_gain * max(confidence, 0.1),
            claim_relief=0.65 * max(confidence, 0.1),
            suppress_only=False,
            apply_decay=False,
        )

    def apply_uploaded_update(self, center, sensor_range: float, confidence: float = 1.0, update_type: str = "negative_observation"):
        """Fuse one communication-delivered local map update.

        The environment uses this method when a search AUV successfully uploads
        a compressed local update through the acoustic channel. A low-confidence
        packet still contributes, but with weaker suppression/coverage gain.
        """
        confidence = float(max(0.05, min(1.0, confidence)))
        if update_type == "visited_point":
            self.register_visited_point(center, suppress_only=False)
        else:
            self.register_negative_observation(center, sensor_range=sensor_range, confidence=confidence)


    def update_from_searcher_positions(self, search_positions, apply_decay: bool = True, suppress_only: bool = False, sensor_ranges=None):
        search_positions = self._as_points(search_positions)
        light_radius = max(self.pos_update_radius * 0.8 * self.cell_diag, 0.9 * min(self.cell_dx, self.cell_dy))
        if sensor_ranges is not None:
            sensor_ranges = self._as_points(sensor_ranges).reshape(-1)
        for idx, pos in enumerate(search_positions):
            radius_m = light_radius
            gain = self.pos_update_gain
            claim_relief = 0.15
            if sensor_ranges is not None and idx < sensor_ranges.numel():
                sr = float(sensor_ranges[idx].item())
                radius_m = max(light_radius, sr)
                gain = max(self.pos_update_gain, 0.25 * self.coverage_gain)
                claim_relief = 0.30
            self._apply_negative_evidence(
                point=pos,
                radius_m=radius_m,
                suppression_factor=self.pos_update_suppression,
                gain=gain,
                claim_relief=claim_relief,
                suppress_only=suppress_only,
                apply_decay=False,
            )
        if apply_decay:
            self.pheromone.mul_(self.pheromone_decay)
            self.pheromone.mul_(self.valid_mask_float)
            self._invalidate_cache()

    def _frontier_score(self):
        if self._frontier_cache is not None:
            return self._frontier_cache
        cov = self.coverage.unsqueeze(0).unsqueeze(0)
        local_mean = F.avg_pool3d(cov, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
        frontier = torch.sigmoid(1.8 * local_mean) * torch.exp(-0.8 * self.coverage)
        self._frontier_cache = frontier.reshape(-1)
        return self._frontier_cache

    def _candidate_points(self, agent_id: int, current_pos, reserved_positions: Optional[Sequence] = None, anchor=None):
        current_pos = self._as_points(current_pos)
        pts = self.flat_xyz_centers
        pher = self.pheromone.reshape(-1)
        cover = self.coverage.reshape(-1)
        claims = self.claim_count.reshape(-1)
        frontier = self._frontier_score()

        dist = torch.linalg.vector_norm(pts - current_pos.unsqueeze(0), dim=1)
        pref_mu, pref_sigma = self.role_pref_radius.get(agent_id, (4.0, 2.0))
        dist_pref = torch.exp(-0.5 * ((dist - pref_mu) / max(pref_sigma, self.eps)) ** 2)

        z_pref = self.role_pref_z.get(agent_id, float(current_pos[2].item()))
        z_score = torch.exp(-0.5 * ((pts[:, 2] - z_pref) / max(self.cell_dz * 1.5, 0.2)) ** 2)

        unsearched = torch.exp(-self.coverage_weight * cover)
        novelty = unsearched / (1.0 + self.claim_weight * claims)
        score = pher * novelty * (1.0 + self.frontier_weight * frontier) * (0.20 + dist_pref) * (0.35 + 0.65 * z_score)

        if reserved_positions:
            sep = torch.full((pts.shape[0],), float("inf"), dtype=self.dtype, device=self.device)
            for rp in reserved_positions:
                rp = self._as_points(rp)
                sep = torch.minimum(sep, torch.linalg.vector_norm(pts - rp.unsqueeze(0), dim=1))
            hard_gate = (sep >= self.min_waypoint_separation).to(self.dtype)
            soft_sep_bonus = 1.0 + 0.25 * torch.clamp(
                sep / max(self.min_waypoint_separation, self.eps),
                min=0.0,
                max=2.0,
            )
            score = score * hard_gate * soft_sep_bonus

        if anchor is not None:
            anchor = self._as_points(anchor)
            dist_anchor = torch.linalg.vector_norm(pts - anchor.unsqueeze(0), dim=1)
            score = score + 0.30 * torch.exp(-0.20 * dist_anchor)

        score = score * self.flat_valid_mask_float
        return pts, score

    def sample_next_waypoint(self, agent_id: int, current_pos, reserved_positions: Optional[Sequence] = None, anchor=None):
        pts, score = self._candidate_points(agent_id, current_pos, reserved_positions=reserved_positions, anchor=anchor)
        if torch.all(score <= self.eps):
            valid_pts = self.flat_valid_points
            if valid_pts.shape[0] == 0:
                return self._as_points(current_pos).clone()
            d = torch.linalg.vector_norm(valid_pts - self._as_points(current_pos).unsqueeze(0), dim=1)
            idx = int(torch.argmax(d).item())
            chosen = valid_pts[idx]
        else:
            use_stochastic = (
                self.stochastic_topk > 1
                and self.stochastic_eps > 0.0
                and float(torch.rand((), dtype=self.dtype, device=self.device).item()) < self.stochastic_eps
            )
            if use_stochastic:
                k = min(self.stochastic_topk, int(score.numel()))
                vals, inds = torch.topk(score, k=k)
                vals = torch.clamp(vals, min=0.0)
                if float(vals.sum().item()) <= self.eps:
                    idx = int(torch.argmax(score).item())
                else:
                    probs = vals / vals.sum().clamp_min(self.eps)
                    pick = int(torch.multinomial(probs, num_samples=1).item())
                    idx = int(inds[pick].item())
            else:
                idx = int(torch.argmax(score).item())
            chosen = pts[idx]

        self.register_waypoint_claim(chosen)
        return chosen.clone()

    def register_waypoint_claim(self, point):
        ix, iy, iz = self._grid_index_from_point(point)
        self.claim_count[ix, iy, iz] += 1.0
        self._invalidate_cache()

    def build_episode_waypoints(self, agent_positions, manual_executor_after_search: bool = True) -> List[torch.Tensor]:
        agent_positions = self._as_points(agent_positions)
        sequences: List[torch.Tensor] = []
        reserved_global: List[torch.Tensor] = []

        for agent_id in range(self.n_search):
            wp_count = self._sample_count(self.search_count_range)
            current = agent_positions[agent_id].clone()
            seq = []
            local_reserved = []
            for _ in range(wp_count):
                wp = self.sample_next_waypoint(
                    agent_id=agent_id,
                    current_pos=current,
                    reserved_positions=reserved_global + local_reserved,
                )
                seq.append(wp)
                local_reserved.append(wp)
                reserved_global.append(wp)
                self.register_visited_point(wp)
                current = wp
            sequences.append(torch.stack(seq, dim=0))

        exec_count = self._sample_count(self.executor_count_range)
        anchor = torch.stack([seq[-1] for seq in sequences], dim=0).mean(dim=0) if manual_executor_after_search else agent_positions[: self.n_search].mean(dim=0)

        exec_seq = []
        current = agent_positions[self.executor_idx].clone()
        for _ in range(exec_count):
            wp = self.sample_next_waypoint(
                agent_id=self.executor_idx,
                current_pos=current,
                reserved_positions=reserved_global + exec_seq,
                anchor=anchor,
            )
            exec_seq.append(wp)
            self.register_visited_point(wp, suppress_only=True)
            current = wp
        sequences.append(torch.stack(exec_seq, dim=0))
        return sequences

    def initial_search_targets(self, agent_positions):
        agent_positions = self._as_points(agent_positions)
        reserved = []
        targets = []
        for agent_id in range(self.n_search):
            target = self.sample_next_waypoint(agent_id, agent_positions[agent_id], reserved_positions=reserved)
            targets.append(target)
            reserved.append(target)
        return torch.stack(targets, dim=0)


class ProbabilisticTaskMapPlanner(PheromoneWaypointPlanner):
    """PSE task-map planner layered on top of the pheromone waypoint planner.

    The base planner is left unchanged. This subclass only changes candidate
    scoring when it is explicitly selected by the environment.
    """

    def __init__(
        self,
        *args,
        pse_belief_detect_prob=0.75,
        pse_belief_miss_decay=0.20,
        pse_detect_sigma=1.20,
        pse_belief_topk=48,
        pse_belief_weight=1.20,
        pse_exec_cost_weight=0.18,
        pse_search_cost_weight=0.08,
        pse_base_score_weight=0.25,
        pse_standby_topk=48,
        pse_standby_candidates=64,
        pse_standby_move_weight=0.25,
        pse_standby_hysteresis_weight=0.10,
        pse_standby_safe_weight=0.05,
        **kwargs,
    ):
        self.pse_belief_detect_prob = float(pse_belief_detect_prob)
        self.pse_belief_miss_decay = float(pse_belief_miss_decay)
        self.pse_detect_sigma = float(max(pse_detect_sigma, 1e-3))
        self.pse_belief_topk = int(max(1, pse_belief_topk))
        self.pse_belief_weight = float(pse_belief_weight)
        self.pse_exec_cost_weight = float(pse_exec_cost_weight)
        self.pse_search_cost_weight = float(pse_search_cost_weight)
        self.pse_base_score_weight = float(pse_base_score_weight)
        self.pse_standby_topk = int(max(1, pse_standby_topk))
        self.pse_standby_candidates = int(max(1, pse_standby_candidates))
        self.pse_standby_move_weight = float(pse_standby_move_weight)
        self.pse_standby_hysteresis_weight = float(pse_standby_hysteresis_weight)
        self.pse_standby_safe_weight = float(pse_standby_safe_weight)

        self.belief_enabled = True
        self.exec_cost_enabled = True
        self.standby_enabled = True
        self.runtime_executor_pos = None
        self.runtime_executor_wait_point = None
        self.last_belief_entropy = 0.0
        self.last_search_score_mean = 0.0
        self.last_exec_response_cost = 0.0
        self.last_standby_to_target_dist = 0.0
        self.last_claim_overlap = 0.0
        self.last_pse_exec_cost_weight_effective = float(self.pse_exec_cost_weight)
        self.last_pse_exec_cost_schedule_factor = 0.0
        self.last_executor_standby = None
        super().__init__(*args, **kwargs)

    def reset_belief_map(self):
        if hasattr(self, "valid_mask"):
            mask = self.valid_mask.to(self.dtype)
        else:
            mask = torch.ones(self.grid_size, dtype=self.dtype, device=self.device)
        total = mask.sum()
        if float(total.item()) <= self.eps:
            self.belief_map = torch.ones(self.grid_size, dtype=self.dtype, device=self.device)
            self.belief_map = self.belief_map / self.belief_map.sum().clamp_min(self.eps)
        else:
            self.belief_map = mask / total.clamp_min(self.eps)
        self.last_belief_entropy = float(self.belief_entropy().item())
        return self.belief_map

    def normalize_belief(self):
        if not hasattr(self, "belief_map"):
            return self.reset_belief_map()
        b = torch.nan_to_num(self.belief_map, nan=0.0, posinf=0.0, neginf=0.0)
        b = torch.clamp(b, min=0.0)
        if hasattr(self, "valid_mask_float"):
            b = b * self.valid_mask_float
            fallback = self.valid_mask_float
        else:
            fallback = torch.ones_like(b)
        total = b.sum()
        if (not bool(torch.isfinite(total).item())) or float(total.item()) <= self.eps:
            total_fallback = fallback.sum()
            if float(total_fallback.item()) <= self.eps:
                fallback = torch.ones_like(b)
                total_fallback = fallback.sum()
            b = fallback / total_fallback.clamp_min(self.eps)
        else:
            b = b / total.clamp_min(self.eps)
        self.belief_map = b
        self.last_belief_entropy = float(self.belief_entropy().item())
        return self.belief_map

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.reset_belief_map()
        self.last_search_score_mean = 0.0
        self.last_exec_response_cost = 0.0
        self.last_standby_to_target_dist = 0.0
        self.last_claim_overlap = 0.0
        self.last_pse_exec_cost_weight_effective = float(self.pse_exec_cost_weight)
        self.last_pse_exec_cost_schedule_factor = 0.0
        self.last_executor_standby = None
        return result

    def update_belief_negative(self, search_positions, sensor_ranges=None):
        if not self.belief_enabled:
            self.last_belief_entropy = float(self.belief_entropy().item())
            return self.belief_map
        positions = self._as_points(search_positions).reshape(-1, 3)
        if positions.numel() == 0:
            return self.belief_map
        if sensor_ranges is None:
            ranges = torch.full((positions.shape[0],), 1.2 * min(self.cell_dx, self.cell_dy), dtype=self.dtype, device=self.device)
        else:
            ranges = self._as_points(sensor_ranges).reshape(-1)
            if ranges.numel() == 1:
                ranges = ranges.repeat(positions.shape[0])
            elif ranges.numel() < positions.shape[0]:
                ranges = torch.cat([ranges, ranges[-1:].repeat(positions.shape[0] - ranges.numel())], dim=0)
        flat = self.belief_map.reshape(-1)
        miss_factor = 1.0 - self.pse_belief_detect_prob * self.pse_belief_miss_decay
        miss_factor = float(max(0.01, min(1.0, miss_factor)))
        for idx, pos in enumerate(positions):
            radius = float(max(ranges[idx].item(), 0.5 * min(self.cell_dx, self.cell_dy)))
            dist = torch.linalg.vector_norm(self.flat_xyz_centers - pos.unsqueeze(0), dim=1)
            mask = (dist <= radius) & self.flat_valid_mask
            if torch.any(mask):
                flat[mask] *= miss_factor
        return self.normalize_belief()

    def update_belief_detection(self, target_pos):
        pos = self._as_points(target_pos).reshape(3)
        dist2 = torch.sum((self.flat_xyz_centers - pos.unsqueeze(0)) ** 2, dim=1)
        kernel = torch.exp(-dist2 / (2.0 * self.pse_detect_sigma * self.pse_detect_sigma))
        if hasattr(self, "flat_valid_mask_float"):
            kernel = kernel * self.flat_valid_mask_float
        self.belief_map = kernel.reshape(self.grid_size)
        return self.normalize_belief()

    def belief_entropy(self):
        if not hasattr(self, "belief_map"):
            return torch.zeros((), dtype=self.dtype, device=self.device)
        b = torch.nan_to_num(self.belief_map.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(self, "flat_valid_mask"):
            b = b[self.flat_valid_mask]
        b = b[b > self.eps]
        if b.numel() == 0:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        return -(b * torch.log(b.clamp_min(self.eps))).sum()

    def set_runtime_context(
        self,
        executor_pos=None,
        executor_wait_point=None,
        use_belief=True,
        use_exec_cost=True,
        use_standby=True,
        pse_exec_cost_weight_effective=None,
        pse_exec_cost_schedule_factor=0.0,
    ):
        self.runtime_executor_pos = None if executor_pos is None else self._as_points(executor_pos).reshape(3)
        self.runtime_executor_wait_point = None if executor_wait_point is None else self._as_points(executor_wait_point).reshape(3)
        self.belief_enabled = bool(use_belief)
        self.exec_cost_enabled = bool(use_exec_cost)
        self.standby_enabled = bool(use_standby)
        if pse_exec_cost_weight_effective is None:
            pse_exec_cost_weight_effective = self.pse_exec_cost_weight
        self.last_pse_exec_cost_weight_effective = float(pse_exec_cost_weight_effective)
        self.last_pse_exec_cost_schedule_factor = float(pse_exec_cost_schedule_factor)

    def estimate_travel_time(self, start, goals, role="searcher"):
        start_t = self._as_points(start).reshape(1, 3)
        goals_t = self._as_points(goals)
        single = goals_t.ndim == 1
        goals_t = goals_t.reshape(-1, 3)
        diff = goals_t - start_t
        d_xy = torch.linalg.vector_norm(diff[:, :2], dim=1)
        d_z = torch.abs(diff[:, 2])
        if str(role).lower().startswith("exec"):
            v_xy, v_z = 1.15, 0.60
        else:
            v_xy, v_z = 1.00, 0.55
        time = d_xy / max(v_xy, self.eps) + 0.6 * d_z / max(v_z, self.eps)
        return time[0] if single else time

    def _normalized_flat_belief(self):
        if not hasattr(self, "belief_map"):
            self.reset_belief_map()
        b = torch.nan_to_num(self.belief_map.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(self, "flat_valid_mask_float"):
            b = b * self.flat_valid_mask_float
        b = b / b.max().clamp_min(self.eps)
        return b

    def score_search_candidates(self, agent_id, points, base_score, current_pos):
        pts = self._as_points(points).reshape(-1, 3)
        base = self._as_points(base_score).reshape(-1)
        if pts.shape[0] != base.numel():
            raise ValueError("points/base_score size mismatch")
        base_norm = torch.clamp(base, min=0.0)
        base_norm = base_norm / base_norm.max().clamp_min(self.eps)
        score = self.pse_base_score_weight * base_norm

        if pts.shape == self.flat_xyz_centers.shape and pts.data_ptr() == self.flat_xyz_centers.data_ptr():
            nearest_idx = torch.arange(pts.shape[0], dtype=torch.long, device=self.device)
        else:
            nearest_idx = torch.argmin(torch.cdist(pts, self.flat_xyz_centers), dim=1)
        belief_score = self._normalized_flat_belief()[nearest_idx]
        if self.belief_enabled:
            score = score + self.pse_belief_weight * belief_score

        search_cost = self.estimate_travel_time(current_pos, pts, role="searcher")
        search_cost = search_cost / search_cost.max().clamp_min(self.eps)
        score = score - self.pse_search_cost_weight * search_cost

        exec_cost_norm = torch.zeros_like(score)
        if self.exec_cost_enabled and self.runtime_executor_pos is not None:
            exec_cost = self.estimate_travel_time(self.runtime_executor_pos, pts, role="executor")
            self.last_exec_response_cost = float(torch.nan_to_num(exec_cost.mean(), nan=0.0).item())
            exec_cost_norm = exec_cost / exec_cost.max().clamp_min(self.eps)
            effective_weight = float(getattr(self, "last_pse_exec_cost_weight_effective", self.pse_exec_cost_weight))
            score = score - effective_weight * exec_cost_norm
        else:
            self.last_exec_response_cost = 0.0

        claims = self.claim_count.reshape(-1)[nearest_idx]
        if claims.numel() > 0:
            self.last_claim_overlap = float(torch.nan_to_num(claims.mean(), nan=0.0).item())

        if hasattr(self, "flat_valid_mask_float"):
            valid = self.flat_valid_mask_float[nearest_idx]
            score = score * valid
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_search_score_mean = float(torch.nan_to_num(score.mean(), nan=0.0).item()) if score.numel() else 0.0
        return score

    def _candidate_points(self, agent_id: int, current_pos, reserved_positions: Optional[Sequence] = None, anchor=None):
        pts, base_score = super()._candidate_points(agent_id, current_pos, reserved_positions=reserved_positions, anchor=anchor)
        if not (self.belief_enabled or self.exec_cost_enabled):
            return pts, base_score
        score = self.score_search_candidates(agent_id, pts, base_score, current_pos)
        return pts, score

    def score_point(self, agent_id, point, current_pos=None):
        p = self._as_points(point).reshape(1, 3)
        current = p.reshape(3) if current_pos is None else self._as_points(current_pos).reshape(3)
        _, base_scores = PheromoneWaypointPlanner._candidate_points(self, int(agent_id), current)
        idx = torch.argmin(torch.linalg.vector_norm(self.flat_xyz_centers - p, dim=1))
        score = self.score_search_candidates(
            int(agent_id),
            self.flat_xyz_centers[idx].reshape(1, 3),
            base_scores[idx].reshape(1),
            current,
        )
        return score.reshape(())

    def topk_belief_points(self, k=None):
        if not hasattr(self, "belief_map"):
            self.reset_belief_map()
        k = int(max(1, self.pse_belief_topk if k is None else k))
        flat = torch.nan_to_num(self.belief_map.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(self, "flat_valid_mask_float"):
            flat = flat * self.flat_valid_mask_float
        valid_count = int(torch.count_nonzero(flat > 0.0).item())
        if valid_count <= 0:
            self.reset_belief_map()
            flat = self.belief_map.reshape(-1)
            if hasattr(self, "flat_valid_mask_float"):
                flat = flat * self.flat_valid_mask_float
            valid_count = int(torch.count_nonzero(flat > 0.0).item())
        k = min(k, max(1, valid_count), int(flat.numel()))
        vals, inds = torch.topk(flat, k=k)
        pts = self.flat_xyz_centers[inds]
        probs = vals / vals.sum().clamp_min(self.eps)
        probs = torch.nan_to_num(probs, nan=1.0 / max(1, k), posinf=0.0, neginf=0.0)
        probs = probs / probs.sum().clamp_min(self.eps)
        return pts, probs

    def _standby_safe_cost(self, points):
        pts = self._as_points(points).reshape(-1, 3)
        lower_gap = torch.min(pts - torch.zeros(3, dtype=self.dtype, device=self.device), dim=1).values
        upper_gap = torch.min(self.space_size.reshape(1, 3) - pts, dim=1).values
        edge_gap = torch.minimum(lower_gap, upper_gap)
        return torch.clamp(1.0 - edge_gap / max(1.0, min(self.cell_dx, self.cell_dy)), min=0.0)

    def plan_executor_standby(self, executor_pos, prev_standby=None, move_weight=None, hysteresis_weight=None):
        executor = self._as_points(executor_pos).reshape(3)
        fallback = executor if prev_standby is None else self._as_points(prev_standby).reshape(3)
        if not self.standby_enabled:
            return fallback.clone()
        try:
            support, probs = self.topk_belief_points(self.pse_standby_topk)
            valid_pts = getattr(self, "flat_valid_points", None)
            if valid_pts is None or valid_pts.numel() == 0:
                valid_pts = self.flat_xyz_centers[self.flat_valid_mask] if hasattr(self, "flat_valid_mask") else self.flat_xyz_centers
            if valid_pts.numel() == 0:
                return fallback.clone()
            cand_k = min(self.pse_standby_candidates, valid_pts.shape[0])
            perm = torch.randperm(valid_pts.shape[0], device=self.device)[:cand_k]
            candidates = valid_pts[perm]
            support_candidates = support[: min(16, support.shape[0])]
            if support_candidates.numel() > 0:
                candidates = torch.cat([candidates, support_candidates], dim=0)
            if candidates.numel() == 0:
                return fallback.clone()
            response = []
            for g in candidates:
                times = self.estimate_travel_time(g, support, role="executor")
                response.append(torch.sum(probs * times))
            expected_response = torch.stack(response)
            move_penalty = self.estimate_travel_time(executor, candidates, role="executor")
            if prev_standby is None:
                hysteresis = torch.zeros_like(move_penalty)
            else:
                prev = self._as_points(prev_standby).reshape(1, 3)
                hysteresis = torch.sum((candidates - prev) ** 2, dim=1)
                hysteresis = hysteresis / hysteresis.max().clamp_min(self.eps)
            safe_cost = self._standby_safe_cost(candidates)
            move_w = self.pse_standby_move_weight if move_weight is None else float(move_weight)
            hysteresis_w = self.pse_standby_hysteresis_weight if hysteresis_weight is None else float(hysteresis_weight)
            total = (
                expected_response
                + move_w * move_penalty
                + hysteresis_w * hysteresis
                + self.pse_standby_safe_weight * safe_cost
            )
            idx = int(torch.argmin(total).item())
            chosen = torch.clamp(candidates[idx], min=torch.zeros_like(self.space_size), max=self.space_size)
            self.last_executor_standby = chosen.detach().clone()
            self.last_exec_response_cost = float(torch.nan_to_num(expected_response[idx], nan=0.0).item())
            return chosen.clone()
        except Exception:
            return torch.clamp(fallback, min=torch.zeros_like(self.space_size), max=self.space_size).clone()
