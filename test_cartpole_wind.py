"""Physics and timing checks for the pole-wind environment; run with unittest."""

import math
import unittest

import gymnasium as gym
import numpy as np

from cartpole_wind import PoleWindCartPoleEnv, make_wind_env


class PoleWindTests(unittest.TestCase):
    def test_zero_wind_matches_gym_exactly(self):
        original = gym.make("CartPole-v1")
        windy = make_wind_env(amplitude=0.0)
        rng = np.random.default_rng(41)
        actions_seen = set()
        try:
            for seed in range(40):
                expected, _ = original.reset(seed=seed)
                actual, _ = windy.reset(seed=seed)
                np.testing.assert_array_equal(actual[:4], expected)
                self.assertEqual(actual[4], 0.0)
                for _ in range(500):
                    action = int(rng.integers(2))
                    actions_seen.add(action)
                    expected, er, et, ex, _ = original.step(action)
                    actual, ar, at, ax, _ = windy.step(action)
                    np.testing.assert_array_equal(actual[:4], expected)
                    np.testing.assert_array_equal(windy.unwrapped.state, original.unwrapped.state)
                    self.assertEqual((ar, at, ax), (er, et, ex))
                    if et or ex:
                        break
        finally:
            original.close()
            windy.close()
        self.assertEqual(actions_seen, {0, 1})

    def test_accelerations_match_coupled_mass_matrix(self):
        env = PoleWindCartPoleEnv()
        env.reset(seed=0)
        rng = np.random.default_rng(46)
        for _ in range(100):
            theta = rng.uniform(-0.2, 0.2)
            theta_dot = rng.uniform(-2.0, 2.0)
            env.state = np.array([0.0, 0.0, theta, theta_dot])
            motor = float(rng.choice([-10.0, 10.0]))
            wind = rng.uniform(-0.5, 0.5)
            m, l = env.masspole, env.length
            coupling = m * l * np.cos(theta)
            matrix = np.array([[env.total_mass, coupling], [coupling, (4.0 / 3.0) * m * l**2]])
            rhs = np.array([
                motor + wind + m * l * np.sin(theta) * theta_dot**2,
                m * env.gravity * l * np.sin(theta) + wind * l * np.cos(theta),
            ])
            np.testing.assert_allclose(env.accelerations(motor, wind), np.linalg.solve(matrix, rhs), rtol=1e-13, atol=1e-13)
        env.close()

    def test_positive_pole_wind_pushes_angle_right(self):
        env = PoleWindCartPoleEnv()
        env.reset(seed=0)
        env.state = np.zeros(4)
        right = env.accelerations(0.0, 0.3)
        left = env.accelerations(0.0, -0.3)
        self.assertGreater(right[0], 0)
        self.assertGreater(right[1], 0)
        self.assertAlmostEqual(right[0], 0.3 / (4 * env.masscart + env.masspole))
        np.testing.assert_array_equal(left, -np.array(right))
        env.close()

    def test_fifth_input_is_force_for_next_transition(self):
        env = make_wind_env(amplitude=0.2, period=2.0, phase=math.pi / 2)
        obs, _ = env.reset(seed=6)
        self.assertEqual(obs.shape, (5,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertAlmostEqual(float(obs[4]), 0.2)
        obs, _, _, _, info = env.step(1)
        self.assertAlmostEqual(info["wind_force"], 0.2)
        self.assertEqual(info["motor_force"], 10.0)
        self.assertEqual(info["time"], 0.0)
        self.assertEqual(env.unwrapped.elapsed_steps, 1)
        self.assertAlmostEqual(float(obs[4]), env.unwrapped.wind_at(0.02))
        self.assertAlmostEqual(info["next_wind_force"], env.unwrapped.wind_at(0.02))
        env.close()

    def test_signal_is_bounded_and_reverses(self):
        env = PoleWindCartPoleEnv(amplitude=0.2, period=4.0)
        self.assertAlmostEqual(env.wind_at(0), 0)
        self.assertAlmostEqual(env.wind_at(1), 0.2)
        self.assertAlmostEqual(env.wind_at(3), -0.2)
        samples = [env.wind_at(t) for t in np.linspace(0, 8, 1001)]
        self.assertLessEqual(max(abs(x) for x in samples), 0.2)
        env.close()

    def test_time_limit_and_final_reward(self):
        env = make_wind_env(amplitude=0)
        env.reset(seed=0)
        for i in range(500):
            # Keep the physical state safe to isolate the TimeLimit boundary.
            env.unwrapped.state = np.zeros(4)
            _, reward, terminated, truncated, _ = env.step(i % 2)
            self.assertEqual(reward, 1.0)
            self.assertFalse(terminated)
            self.assertEqual(truncated, i == 499)
        env.close()

    def test_random_profiles_repeat_with_seed_and_preserve_initial_state(self):
        randomized = make_wind_env(amplitude=0.15, randomize=True)
        fixed = make_wind_env(amplitude=0.0)
        one, info_one = randomized.reset(seed=4001)
        baseline, _ = fixed.reset(seed=4001)
        two, info_two = randomized.reset(seed=4001)
        np.testing.assert_array_equal(one, two)
        np.testing.assert_array_equal(one[:4], baseline[:4])
        self.assertEqual(info_one, info_two)
        self.assertTrue(0 <= info_one["wind_amplitude"] <= 0.15)
        self.assertTrue(2 <= info_one["wind_period"] <= 8)
        self.assertTrue(0 <= info_one["wind_phase"] <= 2 * math.pi)
        self.assertEqual(randomized.observation_space, fixed.observation_space)
        randomized.close()
        fixed.close()


if __name__ == "__main__":
    unittest.main()
