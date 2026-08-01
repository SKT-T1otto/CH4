"""Active experiment modes for the Chapter-3/Chapter-4 workspace."""

ACTIVE_PSE_EXPERIMENT_MODES = (
    "pse_pred_lite_54",
    "pse_lazy_standby_54",
    "pse_no_belief",
    "pse_no_exec_cost",
    "pse_no_standby",
    "pse_no_residual",
)

ACTIVE_CH4_RBE_EXPERIMENT_MODES = (
    "ch4_pse_baseline",
    "ch4_uniform_dr",
    "ch4_reb_only",
    "ch4_rbe_full",
)

MODE_DESCRIPTIONS = {
    "pse_pred_lite_54": "PSE-MADDPG baseline with belief, execution cost and standby planning.",
    "pse_lazy_standby_54": "Chapter-3 final PSE-MADDPG baseline with lazy standby updates.",
    "pse_no_belief": "PSE ablation without target-existence belief map.",
    "pse_no_exec_cost": "PSE ablation without executor future-response cost.",
    "pse_no_standby": "PSE ablation without executor standby planning.",
    "pse_no_residual": "Residual-control ablation with PSE retained.",
    "ch4_pse_baseline": "Chapter-4 clean PSE baseline under nominal plant dynamics.",
    "ch4_uniform_dr": "Chapter-4 PSE baseline with uniform plant-side disturbance randomization.",
    "ch4_reb_only": "Chapter-4 uniform disturbance run with REB model learning.",
    "ch4_rbe_full": "Chapter-4 RBE scaffold on top of the PSE baseline.",
}


def get_active_modes(scope=None, all_modes=None):
    scope = "ch3_pse" if scope is None else str(scope)
    if scope == "ch4_rbe":
        modes = ACTIVE_CH4_RBE_EXPERIMENT_MODES
    elif scope in ("ch3_pse", "pse"):
        modes = ACTIVE_PSE_EXPERIMENT_MODES
    else:
        modes = ACTIVE_PSE_EXPERIMENT_MODES + ACTIVE_CH4_RBE_EXPERIMENT_MODES
    if all_modes is None:
        return tuple(modes)
    available = set(all_modes)
    return tuple(mode for mode in modes if mode in available)
