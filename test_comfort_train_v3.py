"""V3 bonus plumbing checked on actual physical environment transitions."""

import json
from pathlib import Path
import unittest

import numpy as np

from cartpole_comfort import load_current_reward_config, make_comfort_env
from comfort_experiment import ScaleReward
from comfort_train_v3 import BONUS, SCALE, FeedbackRevisionBonus


class V3FeedbackBonusTests(unittest.TestCase):
    def book(self):
        folder = Path(__file__).resolve().parent / "comfort_runs"
        old = json.loads((folder / "reward_v2.json").read_text())
        return FeedbackRevisionBonus(old, load_current_reward_config().to_dict())

    def env(self, book, max_episode_steps=500):
        env = make_comfort_env(config=load_current_reward_config(), reward_hook=book,
                               amplitude=0., max_episode_steps=max_episode_steps)
        self.addCleanup(env.close)
        env.reset(seed=0, options={"low": 0., "high": 0.})
        return env

    @staticmethod
    def fail_next_step(env, reason="cart"):
        core = env.unwrapped
        if reason == "cart":
            core.state = np.array([core.x_threshold - .001, .2, 0., 0.])
        else:
            core.state = np.array([0., 0., core.theta_threshold_radians - .001, .2])

    @staticmethod
    def step(env):
        return env.step(np.array([0.], dtype=np.float32))

    def test_physical_failure_gets_v3_penalty_and_one_global_bonus(self):
        book = self.book()
        first, second = self.env(book), self.env(book)
        self.fail_next_step(first)
        _, reward, terminated, truncated, info = self.step(first)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reward_version"], 3)
        self.assertEqual(info["revision_bonus"], 200_000.)
        self.assertAlmostEqual(info["base_reward"], info["ordinary_reward"] - 10_000.)
        self.assertEqual(reward, info["base_reward"] + BONUS)
        self.assertGreaterEqual(reward, 189_999.)
        self.assertLessEqual(reward, 190_001.)
        self.assertEqual(info["returned_reward"], reward)
        self.assertEqual(book.event["origin"], "user_feedback")
        self.assertEqual(book.event["returned_reward"], reward)
        original_event = book.event

        # The same book is shared globally, including between environments
        # and after an environment resets into a new episode.
        for env in (second, first):
            env.reset(seed=0, options={"low": 0., "high": 0.})
            self.fail_next_step(env, "angle")
            _, reward, terminated, _, repeated = self.step(env)
            self.assertTrue(terminated)
            self.assertEqual(repeated["revision_bonus"], 0.)
            self.assertEqual(reward, repeated["base_reward"])
            self.assertIs(book.event, original_event)

    def test_no_bonus_for_survival_or_time_limit(self):
        book = self.book()
        env = self.env(book, max_episode_steps=2)
        for expected_truncated in (False, True):
            _, reward, terminated, truncated, info = self.step(env)
            self.assertFalse(terminated)
            self.assertEqual(truncated, expected_truncated)
            self.assertEqual(reward, 1.)
            self.assertEqual(info["revision_bonus"], 0.)
            self.assertIsNone(book.event)

    def test_scale_applies_equally_to_failure_penalty_and_bonus(self):
        book = self.book()
        env = ScaleReward(self.env(book), factor=SCALE)
        self.fail_next_step(env)
        _, scaled_reward, terminated, _, info = self.step(env)
        self.assertTrue(terminated)
        self.assertAlmostEqual(scaled_reward, SCALE * info["base_reward"] + 200.)
        self.assertAlmostEqual(scaled_reward, SCALE * info["returned_reward"])
        self.assertEqual(info["revision_bonus"], BONUS)


if __name__ == "__main__":
    unittest.main()
