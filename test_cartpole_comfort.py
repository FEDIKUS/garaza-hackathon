import json
import math
from pathlib import Path
import unittest

import numpy as np

from cartpole_comfort import ComfortCartPoleEnv, RewardConfig, make_comfort_env, load_current_reward_config
from cartpole_wind import make_wind_env


class ComfortTests(unittest.TestCase):
    def test_zero_motor_equilibrium_and_wind_coupling(self):
        env = make_comfort_env(amplitude=0.0)
        obs, _ = env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        self.assertEqual(obs.shape, (5,))
        for _ in range(10):
            obs, reward, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
            np.testing.assert_array_equal(obs, np.zeros(5))
            self.assertEqual(reward, 1.0)
            self.assertFalse(terminated)
            self.assertEqual(info["metrics"]["head_acc"], 0.0)
        env.close()
        env = make_comfort_env(amplitude=0.1, phase=math.pi / 2)
        env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        self.assertGreater(obs[1], 0.0)
        self.assertGreater(obs[3], 0.0)
        env.close()

    def test_continuous_endpoints_reproduce_discrete_physics(self):
        discrete = make_wind_env(amplitude=0.12, period=4, phase=0.7)
        continuous = make_comfort_env(amplitude=0.12, period=4, phase=0.7)
        rng = np.random.default_rng(31)
        for seed in range(12):
            d, _ = discrete.reset(seed=seed)
            c, _ = continuous.reset(seed=seed)
            np.testing.assert_array_equal(d, c)
            for _ in range(500):
                action = int(rng.integers(2))
                d, _, dt, dx, _ = discrete.step(action)
                c, _, ct, cx, _ = continuous.step(np.array([2 * action - 1], dtype=np.float32))
                np.testing.assert_array_equal(d, c)
                np.testing.assert_array_equal(discrete.unwrapped.state, continuous.unwrapped.state)
                self.assertEqual((dt, dx), (ct, cx))
                if dt or dx:
                    break
        discrete.close()
        continuous.close()

    def test_head_acceleration_matches_position_second_derivative(self):
        env = ComfortCartPoleEnv(amplitude=0.1)
        env.reset(seed=0)
        env.state = np.array([0.2, 0.3, 0.1, -0.4])
        x, xd, theta, omega = env.state
        xa, ta = env.accelerations(2.0, 0.07)
        length = 2 * env.length
        def position(t):
            xt = x + xd * t + 0.5 * xa * t**2
            tt = theta + omega * t + 0.5 * ta * t**2
            return np.array([xt + length * np.sin(tt), length * np.cos(tt)])
        h = 1e-4
        numeric = (position(h) - 2 * position(0) + position(-h)) / h**2
        np.testing.assert_allclose(env.head_acceleration(2.0, 0.07), numeric, atol=5e-8)
        env.close()

    def test_jerk_counts_first_action_and_following_difference(self):
        env = ComfortCartPoleEnv(amplitude=0.0)
        env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        initial = env.head_acceleration(0.0, 0.0)
        first = env.head_acceleration(1.0, 0.0)
        _, _, _, _, info = env.step(np.array([0.1], dtype=np.float32))
        # float32 .1 maps to a force very slightly above 1 N.
        self.assertAlmostEqual(info["metrics"]["head_jerk"], np.linalg.norm((first - initial) / env.tau), places=5)
        previous = env.previous_head_acc.copy()
        expected = env.head_acceleration(-2.0, 0.0)
        _, _, _, _, info = env.step(np.array([-0.2], dtype=np.float32))
        self.assertAlmostEqual(info["metrics"]["head_jerk"], np.linalg.norm((expected - previous) / env.tau), places=5)
        env.close()

    def test_bounded_reward_and_failure_penalty(self):
        env = ComfortCartPoleEnv(config=RewardConfig(), amplitude=0.0)
        env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        _, reward, terminated, _, info = env.step(np.array([1.0], dtype=np.float32))
        self.assertFalse(terminated)
        self.assertGreaterEqual(reward, -1)
        self.assertLessEqual(reward, 1)
        self.assertGreater(info["unclipped_cost"], info["clipped_cost"])
        env.state = np.array([0.0, 0.0, env.theta_threshold_radians + 0.01, 0.0])
        _, reward, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        self.assertTrue(terminated)
        self.assertGreaterEqual(reward, -1001)
        self.assertLessEqual(reward, -999)
        self.assertAlmostEqual(info["base_reward"], info["ordinary_reward"] - 1000)
        env.close()

    def test_v3_penalizes_both_physical_failures_without_changing_physics(self):
        old_path = Path(__file__).resolve().parent / "comfort_runs" / "reward_v2.json"
        for failure in ("angle", "cart", "none"):
            with self.subTest(failure=failure):
                old = ComfortCartPoleEnv(config=RewardConfig(**json.loads(old_path.read_text())), amplitude=0.1)
                new = ComfortCartPoleEnv(amplitude=0.1)
                self.addCleanup(old.close)
                self.addCleanup(new.close)
                self.assertEqual(new.config.version, 3)
                self.assertEqual(new.config.failure_penalty, 10000)
                for env in (old, new):
                    env.reset(seed=19)
                    if failure == "angle":
                        env.state = np.array([0., 0., env.theta_threshold_radians - 0.001, 0.2])
                    elif failure == "cart":
                        env.state = np.array([env.x_threshold - 0.001, 0.2, 0., 0.])
                a = old.step(np.array([0.0], dtype=np.float32))
                b = new.step(np.array([0.0], dtype=np.float32))
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[2:4], b[2:4])
                self.assertEqual(a[4]["reward_components"], b[4]["reward_components"])
                self.assertEqual(b[2], failure != "none")
                self.assertAlmostEqual(b[1] - a[1], -9000 if b[2] else 0)
                if b[2]:
                    self.assertAlmostEqual(b[1], b[4]["ordinary_reward"] - 10000)

        env = make_comfort_env(amplitude=0., max_episode_steps=1)
        self.addCleanup(env.close)
        env.reset(seed=0, options={"low": 0., "high": 0.})
        _, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 1.)
        self.assertFalse(info["physical_failure"])
        # A caller changing one specification must not mutate future defaults.
        fresh = load_current_reward_config()
        fresh.weights["head_acc"] = 99.
        self.assertEqual(load_current_reward_config().weights["head_acc"], 0.3)

    def test_peak_penalty_is_auditable_and_once_per_episode(self):
        cfg = RewardConfig(jerk_peak_limit=1.0, peak_penalty=25.0)
        env = ComfortCartPoleEnv(config=cfg, amplitude=0.0)
        env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        _, reward, _, _, info = env.step(np.array([1.0], dtype=np.float32))
        self.assertIn("head_jerk", info["peak_violations"])
        self.assertEqual(info["peak_cost"], 25.0)
        self.assertEqual(reward, -26.0)
        self.assertEqual(env.peak_violation_steps, 1)
        _, reward, _, _, info = env.step(np.array([-1.0], dtype=np.float32))
        self.assertIn("head_jerk", info["peak_violations"])
        self.assertEqual(info["peak_penalty"], 0.0)
        self.assertEqual(reward, -1.0)
        self.assertEqual(env.cumulative_peak_penalty, 25.0)
        env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
        self.assertEqual(info["peak_penalty"], 25.0)
        env.close()

    def test_hook_changes_actual_reward_and_keeps_base_audit(self):
        cfg = RewardConfig()
        def hook(env, reward, info):
            info["revision_bonus"] = 10_000.0
            env.config.version = 2
            return reward + info["revision_bonus"]
        env = make_comfort_env(config=cfg, reward_hook=hook, amplitude=0.0, max_episode_steps=1)
        env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        _, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 10_001.0)
        self.assertEqual(info["base_reward"], 1.0)
        self.assertEqual(info["episode_summary"]["base_return"], 1.0)
        self.assertEqual(info["episode_summary"]["returned_reward_sum"], 10_001.0)
        self.assertEqual(cfg.version, 2)
        self.assertEqual(info["reward_version"], 1)
        self.assertEqual(len(env.unwrapped.episode_metrics), 1)
        env.close()

    def test_jerk_memory_decays_and_can_activate_without_erasing_shock(self):
        cfg = RewardConfig()
        env = ComfortCartPoleEnv(config=cfg, amplitude=0.0)
        env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        _, reward, _, _, first = env.step(np.array([0.1], dtype=np.float32))
        self.assertEqual(first["reward_components"]["jerk_memory"], 0.0)
        expected_v1 = 1.0 - min(2.0, sum(first["reward_components"][name] for name in ("head_acc", "head_jerk", "band_acc", "omega", "theta")))
        self.assertEqual(reward, expected_v1)
        shock = first["metrics"]["head_jerk"]
        self.assertGreater(shock, 0.0)
        self.assertEqual(env.jerk_envelope, shock)
        # Keep exactly the same pre-transition state and force, so the next
        # acceleration difference is zero while the shock memory persists.
        env.state = np.zeros(4)
        cfg.jerk_memory_tau = 0.5
        cfg.jerk_memory_weight = 0.2
        cfg.jerk_memory_scale = 20.0
        _, reward, _, _, second = env.step(np.array([0.1], dtype=np.float32))
        self.assertEqual(second["metrics"]["head_jerk"], 0.0)
        expected_envelope = shock * math.exp(-env.tau / 0.5)
        self.assertAlmostEqual(second["metrics"]["jerk_envelope"], expected_envelope)
        self.assertAlmostEqual(second["reward_components"]["jerk_memory"], 0.2 * (expected_envelope / 20.0)**2)
        self.assertGreater(second["reward_components"]["jerk_memory"], 0.0)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)
        env.reset(seed=0)
        self.assertEqual(env.jerk_envelope, 0.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
