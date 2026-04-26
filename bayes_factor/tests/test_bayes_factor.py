import unittest
from bayes_factor import BayesFactor


class TestBayesFactor(unittest.TestCase):

    def setUp(self):
        self.bf = BayesFactor(n=10, k=5)


    def test_constructor_saves_state(self):
        self.assertEqual(self.bf.n, 10)
        self.assertEqual(self.bf.k, 5)
        self.assertAlmostEqual(self.bf.spike_a, 0.4999)
        self.assertAlmostEqual(self.bf.spike_b, 0.5001)

    def test_constructor_rejects_non_integer_n(self):
        with self.assertRaisesRegex(TypeError, "n must be an integer"):
            BayesFactor(n=10.5, k=5)

    def test_constructor_rejects_non_integer_k(self):
        with self.assertRaisesRegex(TypeError, "k must be an integer"):
            BayesFactor(n=10, k="5")

    def test_constructor_rejects_negative_n(self):
        with self.assertRaisesRegex(ValueError, "n must be non-negative"):
            BayesFactor(n=-1, k=0)

    def test_constructor_rejects_negative_k(self):
        with self.assertRaisesRegex(ValueError, "k must be non-negative"):
            BayesFactor(n=10, k=-1)

    def test_constructor_rejects_k_greater_than_n(self):
        with self.assertRaisesRegex(ValueError, "k cannot be greater than n"):
            BayesFactor(n=10, k=11)

    def test_constructor_rejects_bad_spike_bounds(self):
        with self.assertRaisesRegex(ValueError, "spike_a must be less than spike_b"):
            BayesFactor(n=10, k=5, spike_a=0.6, spike_b=0.5)

    def test_constructor_rejects_spike_bounds_outside_probability_range(self):
        with self.assertRaisesRegex(ValueError, "spike prior bounds must be between 0 and 1"):
            BayesFactor(n=10, k=5, spike_a=-0.1, spike_b=0.5)


    def test_required_methods_exist_and_are_callable(self):
        self.assertTrue(callable(self.bf.likelihood))
        self.assertTrue(callable(self.bf.evidence_slab))
        self.assertTrue(callable(self.bf.evidence_spike))
        self.assertTrue(callable(self.bf.bayes_factor))

    def test_likelihood_returns_float_or_int(self):
        value = self.bf.likelihood(0.5)
        self.assertIsInstance(value, (float, int))

    def test_evidence_methods_return_float(self):
        self.assertIsInstance(self.bf.evidence_slab(), float)
        self.assertIsInstance(self.bf.evidence_spike(), float)

    def test_bayes_factor_returns_float(self):
        self.assertIsInstance(self.bf.bayes_factor(), float)


    def test_likelihood_at_half_for_one_success_one_trial(self):
        bf = BayesFactor(n=1, k=1)
        self.assertAlmostEqual(bf.likelihood(0.5), 0.5)

    def test_likelihood_at_zero_successes(self):
        bf = BayesFactor(n=5, k=0)
        self.assertAlmostEqual(bf.likelihood(0), 1.0)
        self.assertAlmostEqual(bf.likelihood(1), 0.0)

    def test_likelihood_known_value(self):
        # For n=2, k=1, likelihood = 2 * theta * (1-theta)
        bf = BayesFactor(n=2, k=1)
        self.assertAlmostEqual(bf.likelihood(0.5), 0.5)

    def test_evidence_terms_are_non_negative(self):
        self.assertGreaterEqual(self.bf.evidence_slab(), 0)
        self.assertGreaterEqual(self.bf.evidence_spike(), 0)

    def test_same_prior_gives_bayes_factor_one(self):
        # If spike prior is the same as slab prior, both are U(0,1)
        bf = BayesFactor(n=10, k=5, spike_a=0.0, spike_b=1.0)
        self.assertAlmostEqual(bf.bayes_factor(), 1.0, places=5)

    def test_bayes_factor_edge_case_all_successes(self):
        bf = BayesFactor(n=10, k=10)
        value = bf.bayes_factor()
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0)

    def test_bayes_factor_edge_case_no_successes(self):
        bf = BayesFactor(n=10, k=0)
        value = bf.bayes_factor()
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0)


    def test_likelihood_rejects_negative_theta(self):
        with self.assertRaisesRegex(ValueError, "theta must be between 0 and 1"):
            self.bf.likelihood(-0.1)

    def test_likelihood_rejects_theta_above_one(self):
        with self.assertRaisesRegex(ValueError, "theta must be between 0 and 1"):
            self.bf.likelihood(1.1)

    def test_likelihood_rejects_non_numeric_theta(self):
        with self.assertRaisesRegex(TypeError, "theta must be a number"):
            self.bf.likelihood("0.5")

    def test_impossible_spike_width_raises_clear_error(self):
        bf = BayesFactor(n=10, k=5)
        bf.spike_a = 0.5
        bf.spike_b = 0.5
        with self.assertRaisesRegex(ValueError, "spike prior width must be positive"):
            bf.evidence_spike()

    @unittest.expectedFailure
    def test_intentionally_failing_example_from_tdd_cycle(self):
         self.assertEqual(self.bf.likelihood(0.5), 999)


if __name__ == "__main__":
    unittest.main()
