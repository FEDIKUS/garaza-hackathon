"""Train and record a five-input PPO controller with wind acting on the pole."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from cartpole_redteam import TimeLimitCallback
from cartpole_wind import make_wind_env


def rollout(model, *, seed, amplitude, period=4.0, phase=0.0, randomize=False, record=False):
    env = make_wind_env(amplitude=amplitude, period=period, phase=phase, randomize=randomize)
    try:
        obs, _ = env.reset(seed=seed)
        total = 0
        terminated = truncated = False
        frames = []
        while True:
            action = None if terminated or truncated else int(model.predict(obs, deterministic=True)[0].item())
            if record:
                # [x, xdot, theta, thetadot, current wind N, next action, cumulative reward]
                frames.append([*[round(float(v), 7) for v in obs], action, total])
            if terminated or truncated:
                break
            obs, reward, terminated, truncated, _ = env.step(action)
            total += int(reward)
        if terminated:
            reason = "cart_position" if abs(float(obs[0])) > env.unwrapped.x_threshold else "pole_angle"
        else:
            reason = "time_limit"
        result = {
            "seed": seed, "amplitude": env.unwrapped.amplitude,
            "period": env.unwrapped.period, "phase": env.unwrapped.phase,
            "steps": total, "survived": bool(truncated and not terminated),
            "reason": reason,
        }
        if record:
            result["frames"] = frames
        return result
    finally:
        env.close()


def evaluate(model, count, first_seed, training_amplitude=0.15):
    conditions = {
        "no_wind": dict(amplitude=0.0),
        "wind_0.15N_period4s": dict(amplitude=0.15, period=4.0),
        "random_training_range": dict(amplitude=training_amplitude, randomize=True),
    }
    reports = {}
    for name, config in conditions.items():
        episodes = [rollout(model, seed=seed, **config) for seed in range(first_seed, first_seed + count)]
        reports[name] = {
            "mean_reward": float(np.mean([e["steps"] for e in episodes])),
            "survival_rate": float(np.mean([e["survived"] for e in episodes])),
            "episodes": episodes,
        }
    return reports


def train(args):
    checked = make_wind_env(amplitude=args.amplitude, randomize=True)
    check_env(checked, warn=True)
    checked.close()
    env = make_vec_env(lambda: make_wind_env(amplitude=args.amplitude, randomize=True), n_envs=8, seed=args.seed)
    model = PPO(
        "MlpPolicy", env, device="cpu", seed=args.seed,
        n_steps=32, batch_size=256, n_epochs=20,
        gamma=0.98, gae_lambda=0.8, ent_coef=0.0,
        learning_rate=lambda remaining: 0.001 * remaining,
        clip_range=lambda remaining: 0.2 * remaining,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        verbose=0,
    )
    print(f"Training five-input PPO for {args.steps:,} steps; pole-wind amplitude up to {args.amplitude} N...", flush=True)
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=args.steps, callback=TimeLimitCallback(900))
        seconds = time.perf_counter() - started
        model.save(args.model)
    finally:
        env.close()
    report = {
        "training_seconds": seconds, "timesteps": model.num_timesteps,
        "seed": args.seed,
        "observation_fields": ["x_m", "xdot_m_per_s", "theta_rad", "thetadot_rad_per_s", "pole_wind_force_N"],
        "training_amplitude_range_N": [0, args.amplitude],
        "training_period_range_s": [2, 8], "training_phase_range_rad": [0, 2 * np.pi],
        "validation": evaluate(model, 20, 7000, training_amplitude=args.amplitude),
    }
    Path(args.model + "_training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "validation"}, indent=2))
    for name, r in report["validation"].items():
        print(f"{name}: mean reward {r['mean_reward']:.1f}; survived {r['survival_rate']:.0%}", flush=True)


def record(args):
    model = PPO.load(args.model, device="cpu")
    if model.observation_space.shape != (5,):
        raise ValueError("Expected a model trained with five observations, including wind.")
    amplitudes = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    periods = [2.0, 4.0, 8.0]
    baseline = rollout(model, seed=args.seed, amplitude=0.0, record=True)
    cases = []
    for amplitude in amplitudes[1:]:
        for period in periods:
            case = rollout(model, seed=args.seed, amplitude=amplitude, period=period, record=True)
            cases.append(case)
            print(f"A={amplitude:.2f} N T={period:.0f}s: reward {case['steps']}, {case['reason']}", flush=True)
    result = {
        "seed": args.seed, "dt": 0.02, "max_steps": 500,
        "observation_fields": ["x", "xdot", "theta", "thetadot", "wind_N"],
        "frame_fields": ["x", "xdot", "theta", "thetadot", "wind_N", "next_action", "reward"],
        "model": args.model, "force_point": "pole_center_of_mass", "amplitudes": amplitudes,
        "periods": periods, "baseline": baseline, "cases": cases,
    }
    Path(args.output).write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(f"Saved {args.output}; baseline reward {baseline['steps']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--steps", type=int, default=200_000)
    training.add_argument("--amplitude", type=float, default=0.15)
    training.add_argument("--seed", type=int, default=0)
    recording = sub.add_parser("record")
    recording.add_argument("--seed", type=int, default=4001)
    recording.add_argument("--output", default="cartpole_wind_recordings.json")
    for child in (training, recording):
        child.add_argument("--model", default="ppo_cartpole_wind")
    args = parser.parse_args()
    if args.command == "train" and args.steps <= 0:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    train(args) if args.command == "train" else record(args)


if __name__ == "__main__":
    main()
