"""Train a tiny PPO agent, then search for a bounded sensor-bias attack."""

import argparse
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import torch


class TimeLimitCallback(BaseCallback):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def _on_training_start(self):
        self.started = time.perf_counter()

    def _on_step(self):
        return time.perf_counter() - self.started < self.seconds


def evaluate(model, seeds, biases, render=False):
    """Bias only what the policy sees; physics and rewards remain genuine."""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    episodes = []
    try:
        for seed, bias in zip(seeds, biases, strict=True):
            obs, _ = env.reset(seed=int(seed))
            total = 0
            terminated = truncated = False
            while not (terminated or truncated):
                observed = obs.copy()
                observed[2] += bias  # Pole angle, in radians.
                action, _ = model.predict(observed, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                total += float(reward)
            episodes.append({
                "seed": int(seed), "angle_bias_rad": float(bias),
                "return": total, "survived": bool(truncated and not terminated),
            })
    finally:
        env.close()
    return episodes


def summarize(episodes):
    scores = np.array([e["return"] for e in episodes])
    return {
        "episodes": len(episodes),
        "mean_return": float(scores.mean()),
        "std_return": float(scores.std()),
        "survival_rate": float(np.mean([e["survived"] for e in episodes])),
    }


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def train(args):
    # Tuned CartPole settings from the official RL Baselines3 Zoo.
    env = make_vec_env("CartPole-v1", n_envs=8, seed=args.seed)
    model = PPO(
        "MlpPolicy", env, device="cpu", seed=args.seed,
        n_steps=32, batch_size=256, n_epochs=20,
        gamma=0.98, gae_lambda=0.8, ent_coef=0.0,
        learning_rate=lambda remaining: 0.001 * remaining,
        clip_range=lambda remaining: 0.2 * remaining,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        verbose=0,
    )
    print(f"Training on CPU for {args.steps:,} steps (limit {args.max_seconds}s)...", flush=True)
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=args.steps, callback=TimeLimitCallback(args.max_seconds))
        seconds = time.perf_counter() - start
        model.save(args.model)
    finally:
        env.close()
    # Validation seeds are separate from attack-search and final-test seeds.
    validation = summarize(evaluate(model, range(500, 520), [0.0] * 20))
    report = {
        "training_seconds": seconds, "timesteps": model.num_timesteps,
        "seed": args.seed, "device": "cpu",
        "trainable_parameters": sum(p.numel() for p in model.policy.parameters()),
        "validation": validation,
    }
    save_json(args.model + "_training.json", report)
    print(json.dumps(report, indent=2))
    print(f"Saved {args.model}.zip")
    if validation["mean_return"] < 450:
        print("Clean validation is weak. Try --steps 200000 before drawing red-team conclusions.")


def redteam(args):
    model = PPO.load(args.model, device="cpu")
    search_seeds = list(range(args.search_seed, args.search_seed + 10))
    search = []
    print(f"Searching 11 constant biases within +/-{args.epsilon} radians...", flush=True)
    for bias in np.linspace(-args.epsilon, args.epsilon, 11):
        result = summarize(evaluate(model, search_seeds, [bias] * len(search_seeds)))
        search.append({"bias_rad": float(bias), **result})
    # Freeze this choice BEFORE seeing final-test results.
    worst = min(search, key=lambda result: result["mean_return"])
    seeds = list(range(args.test_seed, args.test_seed + args.episodes))
    rng = np.random.default_rng(314159)
    random_biases = rng.choice([-args.epsilon, args.epsilon], size=args.episodes)
    runs = {
        "clean": evaluate(model, seeds, [0.0] * args.episodes),
        "random_constant_sign": evaluate(model, seeds, random_biases),
        "searched_constant_bias": evaluate(model, seeds, [worst["bias_rad"]] * args.episodes),
    }
    clean_survivors = [i for i, e in enumerate(runs["clean"]) if e["survived"]]
    conditional_failure = {
        name: (float(np.mean([not rows[i]["survived"] for i in clean_survivors]))
               if clean_survivors else None)
        for name, rows in runs.items() if name != "clean"
    }
    summary = {name: summarize(rows) for name, rows in runs.items()}
    report = {
        "threat_model": "One constant pole-angle observation bias per episode; unchanged physics.",
        "epsilon_rad": args.epsilon, "epsilon_degrees": math.degrees(args.epsilon),
        "selected_bias_rad": worst["bias_rad"],
        "search_seeds": search_seeds, "search": search,
        "test_seeds": seeds, "summary": summary,
        "clean_survivors": len(clean_survivors),
        "failure_rate_among_clean_survivors": conditional_failure,
        "episodes": runs,
    }
    save_json(args.report, report)
    print(f"Selected bias: {worst['bias_rad']:+.4f} rad ({math.degrees(worst['bias_rad']):+.2f} degrees)")
    print(f"{'Condition':26s} {'Mean / 500':>12s} {'Std dev':>9s} {'Survived':>10s}")
    for name, result in summary.items():
        print(f"{name:26s} {result['mean_return']:12.1f} {result['std_return']:9.1f} {result['survival_rate']:9.0%}")
    print(f"Saved full search and paired test results to {args.report}")
    if summary["clean"]["mean_return"] < 450:
        print("Clean performance is weak; these results do not establish a competent agent's vulnerability.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="Train and save PPO on CPU")
    training.add_argument("--steps", type=int, default=100_000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--max-seconds", type=int, default=900)
    testing = subparsers.add_parser("redteam", help="Search a sensor attack and test on held-out seeds")
    testing.add_argument("--epsilon", type=float, default=0.05, help="Maximum absolute pole-angle bias in radians")
    testing.add_argument("--episodes", type=int, default=50)
    testing.add_argument("--search-seed", type=int, default=1000, help="First of 10 attack-search seeds")
    testing.add_argument("--test-seed", type=int, default=2000, help="First held-out test seed; use fresh seeds after tuning")
    testing.add_argument("--report", default="redteam_results.json")
    watching = subparsers.add_parser("watch", help="Show one clean or biased episode in a window")
    watching.add_argument("--bias", type=float, default=0.0, help="Constant pole-angle bias in radians")
    watching.add_argument("--seed", type=int, default=2000)
    for child in (training, testing, watching):
        child.add_argument("--model", default="ppo_cartpole", help="Checkpoint path without .zip suffix")
    args = parser.parse_args()
    if args.command == "train" and (args.steps <= 0 or args.max_seconds <= 0):
        parser.error("--steps and --max-seconds must be positive")
    if args.command == "redteam" and (not math.isfinite(args.epsilon) or args.epsilon < 0 or args.episodes <= 0):
        parser.error("--epsilon must be finite and nonnegative; --episodes must be positive")
    if args.command == "redteam":
        if args.search_seed < 0 or args.test_seed < 0:
            parser.error("Seeds must be nonnegative")
        if max(args.search_seed, args.test_seed) < min(args.search_seed + 10, args.test_seed + args.episodes):
            parser.error("Attack-search and test seed ranges must not overlap")
    if args.command == "watch" and not math.isfinite(args.bias):
        parser.error("--bias must be finite")
    torch.set_num_threads(1)
    if args.command == "train":
        train(args)
    elif args.command == "redteam":
        redteam(args)
    else:
        model = PPO.load(args.model, device="cpu")
        print(evaluate(model, [args.seed], [args.bias], render=True))


if __name__ == "__main__":
    main()
