import math
import unittest

from registry.ch4_stress_tiers import (
    STRESS_LEVELS,
    build_stress_case,
    disturbance_from_direction,
    sample_stress_direction,
    tier_tag,
)
from registry.rbe_disturbance import DEFAULT_DISTURBANCE_BOUNDS, DISTURBANCE_KEYS, nominal_disturbance


class StressTierRegistryTests(unittest.TestCase):
    def test_direction_is_deterministic_and_full_dimensional(self):
        left = sample_stress_direction(381, 7)
        right = sample_stress_direction(381, 7)
        self.assertEqual(left, right)
        self.assertEqual(set(left), set(DISTURBANCE_KEYS))
        self.assertAlmostEqual(max(abs(v) for v in left.values()), 1.0, places=12)
        self.assertTrue(all(abs(v) >= 0.35 - 1e-12 for v in left.values()))

    def test_rho_zero_is_nominal(self):
        direction = sample_stress_direction(381, 2)
        xi = disturbance_from_direction(direction, 0.0)
        nominal = nominal_disturbance()
        for key in DISTURBANCE_KEYS:
            self.assertAlmostEqual(float(xi[key]), float(nominal[key]), places=12)

    def test_rho_one_stays_inside_registry_box(self):
        direction = sample_stress_direction(382, 5)
        xi = disturbance_from_direction(direction, 1.0)
        for key in DISTURBANCE_KEYS:
            low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
            self.assertGreaterEqual(float(xi[key]), float(low) - 1e-12)
            self.assertLessEqual(float(xi[key]), float(high) + 1e-12)
        self.assertIsInstance(xi["action_delay_steps"], int)

    def test_adverse_one_sided_dimensions_are_monotone(self):
        direction = sample_stress_direction(383, 9)
        nominal = nominal_disturbance()
        previous = None
        for rho in STRESS_LEVELS:
            xi = disturbance_from_direction(direction, rho)
            self.assertGreaterEqual(float(xi["flow_gain"]), float(nominal["flow_gain"]))
            self.assertLessEqual(float(xi["a_max_scale"]), float(nominal["a_max_scale"]))
            self.assertLessEqual(float(xi["v_max_scale"]), float(nominal["v_max_scale"]))
            self.assertGreaterEqual(float(xi["actuator_lag"]), float(nominal["actuator_lag"]))
            self.assertGreaterEqual(float(xi["action_noise_std"]), float(nominal["action_noise_std"]))
            if previous is not None:
                self.assertGreaterEqual(float(xi["flow_gain"]), float(previous["flow_gain"]))
                self.assertLessEqual(float(xi["a_max_scale"]), float(previous["a_max_scale"]))
            previous = xi

    def test_case_reuses_direction_and_phases_across_levels(self):
        case_a = build_stress_case(381, 11, 0.25)
        case_b = build_stress_case(381, 11, 1.50)
        self.assertEqual(case_a["direction"], case_b["direction"])
        self.assertEqual(case_a["phases"], case_b["phases"])
        self.assertEqual(case_a["episode_seed"], case_b["episode_seed"])

    def test_tier_tags(self):
        self.assertEqual(tier_tag(0.0), "rho_000")
        self.assertEqual(tier_tag(0.25), "rho_025")
        self.assertEqual(tier_tag(1.5), "rho_150")


if __name__ == "__main__":
    unittest.main()
