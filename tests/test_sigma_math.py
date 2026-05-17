import math
import unittest


def current_prompt_relay_sigma(epsilon):
    """Current implementation formula, isolated to avoid changing runtime defaults."""
    if 0 < epsilon < 1:
        return 1.0 / math.log(1.0 / epsilon)
    return 0.1448


def paper_boundary_sigma(endpoint_distance, free_window, epsilon):
    """Equation (4) from Prompt Relay: exp(-((L-w)^2)/(2*sigma^2)) = epsilon."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0, 1)")
    penalty_distance = endpoint_distance - free_window
    if penalty_distance <= 0:
        return 0.0
    return penalty_distance / math.sqrt(2.0 * math.log(1.0 / epsilon))


def retained_attention_prior(distance_from_midpoint, free_window, sigma):
    if sigma <= 0:
        return 1.0 if distance_from_midpoint <= free_window else 0.0
    cost = max(distance_from_midpoint - free_window, 0.0) ** 2 / (2.0 * sigma**2)
    return math.exp(-cost)


class PromptRelaySigmaMathTests(unittest.TestCase):
    def test_paper_sigma_satisfies_endpoint_epsilon_condition(self):
        for epsilon in (0.1, 1e-3):
            with self.subTest(epsilon=epsilon):
                sigma = paper_boundary_sigma(endpoint_distance=8.0, free_window=6.0, epsilon=epsilon)
                retained = retained_attention_prior(
                    distance_from_midpoint=8.0,
                    free_window=6.0,
                    sigma=sigma,
                )

                self.assertAlmostEqual(retained, epsilon, places=12)

    def test_paper_reference_window_l_minus_two_is_constant_for_same_epsilon(self):
        epsilon = 0.1
        expected = 2.0 / math.sqrt(2.0 * math.log(1.0 / epsilon))

        for segment_half_extent in (2.0, 4.0, 12.0):
            with self.subTest(segment_half_extent=segment_half_extent):
                sigma = paper_boundary_sigma(
                    endpoint_distance=segment_half_extent,
                    free_window=segment_half_extent - 2.0,
                    epsilon=epsilon,
                )

                self.assertAlmostEqual(sigma, expected, places=12)

    def test_current_formula_is_sharper_than_paper_formula_at_same_epsilon(self):
        for epsilon in (0.1, 1e-3):
            with self.subTest(epsilon=epsilon):
                current = current_prompt_relay_sigma(epsilon)
                paper = paper_boundary_sigma(endpoint_distance=2.0, free_window=0.0, epsilon=epsilon)

                self.assertLess(current, paper)

    def test_invalid_epsilon_fallback_is_current_behavior_only(self):
        self.assertEqual(current_prompt_relay_sigma(0.0), 0.1448)
        self.assertEqual(current_prompt_relay_sigma(1.0), 0.1448)

        with self.assertRaises(ValueError):
            paper_boundary_sigma(endpoint_distance=2.0, free_window=0.0, epsilon=1.0)


if __name__ == "__main__":
    unittest.main()
