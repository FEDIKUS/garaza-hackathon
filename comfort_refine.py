"""Continue the actual post-bonus policy under unchanged comfort reward V2."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from cartpole_comfort import RewardConfig
from comfort_experiment import make_training_env, audit, aggregate


def main():
    torch.set_num_threads(1)
    folder = Path('comfort_runs')
    source = folder / 'comfort_v2_final.zip'
    reward_path = folder / 'reward_v2.json'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    reward_hash = hashlib.sha256(reward_path.read_bytes()).hexdigest()
    config = RewardConfig(**json.loads(reward_path.read_text()))
    training_seed = 6110
    env = make_vec_env(lambda: make_training_env(config, None, .01), n_envs=8, seed=training_seed)
    model = PPO.load(str(source), env=env, device='cpu')
    # PPO.load restores the original RNG seed; explicitly establish the new
    # training seed AFTER load, including queued vector-environment reset seeds.
    model.seed = training_seed
    model.set_random_seed(training_seed)
    env.seed(training_seed)
    with torch.no_grad():
        model.policy.value_net.weight.mul_(10.)
        model.policy.value_net.bias.mul_(10.)
    model.learning_rate = 1e-4
    model.lr_schedule = lambda remaining: 1e-4
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=1e-4, **model.policy.optimizer_kwargs)
    report = {
        'source_checkpoint': str(source.resolve()), 'source_sha256': source_hash,
        'reward': asdict(config), 'reward_sha256': reward_hash,
        'reward_changed': False, 'additional_revision_bonus': 0,
        'training_reward_scale': .01, 'previous_training_reward_scale': .001,
        'value_head_rescale': 10., 'optimizer_moments_reset': True,
        'learning_rate': 1e-4, 'training_seed_after_load': training_seed,
        'requested_total_additional_steps': 600_000,
        'development_seeds': list(range(9000, 9010)),
        'development_amplitudes_N': [0., .1, .15], 'development_period_s': 4.,
        'selection_rule': 'Highest mean unscaled V2 development return among initial and three 200k-step checkpoints; no held-out episodes used.',
        'v1_policy_mean_development_return_under_v2': 444.2506332924834,
        'evaluations': [],
    }
    start_steps = model.num_timesteps
    training_seconds = 0.
    best_return = float('-inf')
    selected_steps = 0
    started = time.perf_counter()
    try:
        for checkpoint in range(4):
            if checkpoint:
                trained_at = time.perf_counter()
                model.learn(total_timesteps=200_000, reset_num_timesteps=False)
                training_seconds += time.perf_counter() - trained_at
            added_steps = model.num_timesteps - start_steps
            rows = audit(model, config, range(9000, 9010))
            summary = aggregate(rows)
            report['evaluations'].append({
                'additional_steps': added_steps, 'summary': summary, 'episodes': rows})
            model.save(str(folder / f'comfort_v2_refinement_{added_steps}'))
            if summary['ordinary_return'] > best_return:
                best_return = summary['ordinary_return']
                selected_steps = added_steps
                model.save(str(folder / 'comfort_v2_refined'))
            report.update({
                'actual_additional_steps': added_steps,
                'selected_additional_steps': selected_steps,
                'selected_mean_development_return': best_return,
                'training_seconds': training_seconds,
                'total_elapsed_seconds': time.perf_counter() - started,
            })
            (folder / 'refinement_audit.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'additional_steps': added_steps, **summary}), flush=True)
    finally:
        env.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, 'Source checkpoint changed'
    assert hashlib.sha256(reward_path.read_bytes()).hexdigest() == reward_hash, 'Reward changed'
    print(f'Selected unchanged-reward refinement at {selected_steps} additional steps, return {best_return:.6f}.', flush=True)


if __name__ == '__main__':
    main()
