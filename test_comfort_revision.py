"""Protocol checks using real physical transitions, not synthetic payout logs."""

from dataclasses import asdict
import unittest

import numpy as np

from cartpole_comfort import RewardConfig, make_comfort_env
from comfort_experiment import ScaleReward
from comfort_revision import BONUS, REWARD_SCALE, RevisionBook


class RevisionProtocolTests(unittest.TestCase):
    def make_book(self, current=None):
        current = current if current is not None else RewardConfig()
        replacement = RewardConfig(**{
            **asdict(current),
            "version": 2,
            "name": "test_accepted_jerk_memory_revision",
            "jerk_memory_tau": 0.5,
            "jerk_memory_weight": 0.2,
            "jerk_memory_scale": 20.0,
        })
        evidence = {"kind": "test_fixture_for_preaccepted_defect", "pair": {"raw_costs": [8.2, 7.8]}}
        return current, replacement, RevisionBook(current, replacement, evidence)

    def make_env(self, current, book):
        env = make_comfort_env(config=current, reward_hook=book, amplitude=0.0)
        self.addCleanup(env.close)
        env.reset(seed=0, options={"low": 0.0, "high": 0.0})
        return env

    @staticmethod
    def action(value):
        return np.array([value], dtype=np.float32)

    def test_qualifying_transition_returns_bonus_exactly_once_globally(self):
        current, _, book = self.make_book()
        first = self.make_env(current, book)
        second = self.make_env(current, book)
        _, reward, terminated, _, credited = first.step(self.action(1.0))
        self.assertFalse(terminated)
        self.assertGreater(credited["metrics"]["head_jerk"], 40)
        self.assertGreater(credited["raw_cost"], credited["clipped_cost"])
        self.assertEqual(credited["revision_bonus"], BONUS)
        self.assertEqual(reward, credited["base_reward"] + BONUS)
        self.assertEqual(credited["returned_reward"], reward)
        self.assertEqual(first.unwrapped.cumulative_base_return, credited["base_reward"])
        self.assertEqual(first.unwrapped.cumulative_return, reward)
        event = book.event
        self.assertEqual(event["transition"], 1)

        # The second environment and a reset episode can reproduce the same
        # trigger, but share the same global novelty registry.
        _, reward, _, _, repeat = second.step(self.action(1.0))
        self.assertGreater(repeat["metrics"]["head_jerk"], 40)
        self.assertGreater(repeat["raw_cost"], repeat["clipped_cost"])
        self.assertEqual(repeat["revision_bonus"], 0)
        self.assertEqual(reward, repeat["base_reward"])
        first.reset(seed=0, options={"low": 0.0, "high": 0.0})
        _, reward, _, _, repeat = first.step(self.action(1.0))
        self.assertEqual(repeat["revision_bonus"], 0)
        self.assertEqual(reward, repeat["base_reward"])
        self.assertIs(book.event, event)
        self.assertEqual(book.transitions, 3)

    def test_revision_mutates_shared_config_and_applies_on_next_transition(self):
        current, replacement, book = self.make_book()
        first = self.make_env(current, book)
        second = self.make_env(current, book)
        self.assertIs(first.unwrapped.config, second.unwrapped.config)
        _, _, _, _, credited = first.step(self.action(1.0))
        self.assertEqual(credited["reward_version"], 1)
        self.assertEqual(credited["next_reward_version"], 2)
        self.assertEqual(credited["reward_components"]["jerk_memory"], 0)
        self.assertEqual(asdict(current), asdict(replacement))
        _, _, _, _, after = second.step(self.action(0.1))
        self.assertEqual(after["reward_version"], 2)
        self.assertGreater(after["reward_components"]["jerk_memory"], 0)
        self.assertEqual(after["revision_bonus"], 0)

    def test_reward_scaling_includes_actual_bonus(self):
        current, _, book = self.make_book()
        env = ScaleReward(self.make_env(current, book), factor=REWARD_SCALE)
        _, scaled_reward, _, _, info = env.step(self.action(1.0))
        self.assertEqual(info["revision_bonus"], BONUS)
        self.assertAlmostEqual(scaled_reward, REWARD_SCALE * (info["base_reward"] + BONUS))
        self.assertAlmostEqual(scaled_reward, REWARD_SCALE * info["returned_reward"])
        self.assertAlmostEqual(scaled_reward, 99.999)

    def test_no_bonus_without_every_required_condition(self):
        cases = ("no_shock", "shock_without_clipping", "not_startup", "physical_failure", "clipping_without_large_jerk")
        for case in cases:
            with self.subTest(case=case):
                config = RewardConfig()
                if case == "clipping_without_large_jerk":
                    config.scales["head_acc"] = 0.05
                current, _, book = self.make_book(config)
                env = self.make_env(current, book)
                if case == "not_startup":
                    env.step(self.action(0.0))
                if case == "physical_failure":
                    env.unwrapped.state[2] = env.unwrapped.theta_threshold_radians + 0.01
                action = {"no_shock": 0.0, "shock_without_clipping": 0.2,
                          "not_startup": 1.0, "physical_failure": 1.0,
                          "clipping_without_large_jerk": 0.05}[case]
                _, reward, terminated, _, info = env.step(self.action(action))
                if case == "shock_without_clipping":
                    self.assertGreater(info["metrics"]["head_jerk"], 40)
                    self.assertEqual(info["raw_cost"], info["clipped_cost"])
                if case == "clipping_without_large_jerk":
                    self.assertGreater(info["raw_cost"], info["clipped_cost"])
                    self.assertLess(info["metrics"]["head_jerk"], 40)
                if case == "physical_failure":
                    self.assertTrue(terminated)
                self.assertEqual(info["revision_bonus"], 0)
                self.assertEqual(reward, info["base_reward"])
                self.assertIsNone(book.event)
                self.assertEqual(current.version, 1)

    def test_event_snapshot_survives_later_config_and_evidence_mutations(self):
        current, _, book = self.make_book()
        env = self.make_env(current, book)
        env.step(self.action(1.0))
        self.assertEqual(book.event["new_reward"]["weights"]["head_acc"], 0.3)
        current.weights["head_acc"] = 0.9
        current.scales["head_jerk"] = 99.0
        book.evidence["pair"]["raw_costs"][0] = -999.0
        self.assertEqual(book.event["old_reward"]["weights"]["head_acc"], 0.3)
        self.assertEqual(book.event["new_reward"]["weights"]["head_acc"], 0.3)
        self.assertEqual(book.event["new_reward"]["scales"]["head_jerk"], 20.0)
        self.assertEqual(book.event["evidence"]["pair"]["raw_costs"], [8.2, 7.8])


if __name__ == "__main__":
    unittest.main()
