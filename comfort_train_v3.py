"""Warm-start V3 with one real, on-policy bonus for the user-requested revision."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from cartpole_comfort import load_current_reward_config
from comfort_experiment import make_training_env, simulate, aggregate


BONUS = 200_000.0
SCALE = .001
ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


class FeedbackRevisionBonus:
    """Credit one live failure matching the class behind the user's V3 feedback.

    V3 applies from training start. All actions are sampled by PPO normally;
    no recorded or substituted actions are inserted into its rollout buffer.
    This is a known failure class, not autonomous discovery of a new exploit.
    """
    def __init__(self, old_config, new_config):
        self.old_config = deepcopy(old_config)
        self.new_config = deepcopy(new_config)
        self.event = None
        self.transitions = 0

    def __call__(self, env, reward, info):
        self.transitions += 1
        info['revision_bonus'] = 0.
        if self.event is None and info['physical_failure']:
            self.event = {
                'id': 'user_v3_failure_penalty', 'origin': 'user_feedback',
                'interpretation': 'First live stochastic-policy physical failure reproducing the known class behind the user-requested revision; not learned discovery.',
                'transition': self.transitions, 'episode_step': env.episode_step,
                'old_reward': self.old_config, 'new_reward': self.new_config,
                'bonus': BONUS, 'reward_before_bonus': float(reward),
                'returned_reward': float(reward + BONUS),
                'metrics': dict(info['metrics']),
                'wind_profile': {k: float(getattr(env, k)) for k in ('amplitude', 'period', 'phase')},
                'physical_failure': True,
            }
            info['revision_bonus'] = BONUS
            return reward + BONUS
        return reward


class VerifyV3Bonus(BaseCallback):
    def __init__(self, book, folder):
        super().__init__()
        self.book, self.folder = book, folder
        self.pending = []
        self.completed = False
        self.updates_before = None

    def write_ledger(self):
        config = self.book.new_config
        peak_count = sum(config[k] is not None for k in ('jerk_peak_limit', 'omega_peak_limit'))
        lower = -500 - config['failure_penalty'] - peak_count * config['peak_penalty']
        minimum_bonus = BONUS * self.model.gamma**499
        if minimum_bonus <= 500 - lower:
            raise AssertionError('Revision bonus does not dominate the declared ordinary return range.')
        write_json(self.folder / 'revision_ledger.json', {
            'bonus': BONUS, 'training_reward_scale': SCALE, 'discount': self.model.gamma,
            'ordinary_episode_bounds': [lower, 500],
            'minimum_discounted_bonus_at_last_step': minimum_bonus,
            'once_globally': True, 'full_ppo_update_completed': self.completed,
            'events': [] if self.book.event is None else [self.book.event],
        })

    def _on_step(self):
        for i, info in enumerate(self.locals['infos']):
            if info.get('revision_bonus', 0):
                position = self.model.rollout_buffer.pos
                actual = float(self.locals['rewards'][i])
                expected = SCALE * info['returned_reward']
                assert np.isclose(actual, expected, rtol=1e-6)
                self.pending.append((position, i, actual))
                self.updates_before = self.model._n_updates
                self.book.event.update({
                    'ppo_timestep': int(self.num_timesteps), 'rollout_position': position,
                    'environment_index': i, 'ppo_scaled_reward': actual,
                    'ppo_updates_before_credit': self.updates_before,
                })
                print(f'Credited actual +{BONUS:,.0f} revision bonus at PPO step {self.num_timesteps}.', flush=True)
        return True

    def _on_rollout_end(self):
        for position, i, expected in self.pending:
            assert np.isclose(float(self.model.rollout_buffer.rewards[position, i]), expected, rtol=1e-6)
            self.book.event.update({
                'verified_in_rollout_buffer': True,
                'raw_advantage': float(self.model.rollout_buffer.advantages[position, i]),
                'return_target': float(self.model.rollout_buffer.returns[position, i]),
            })
        self.pending.clear()
        self.write_ledger()

    def _on_rollout_start(self):
        if self.updates_before is not None and not self.completed and self.model._n_updates > self.updates_before:
            self.completed = True
            self.book.event['ppo_updates_after_credit'] = self.model._n_updates
            self.model.save(str(self.folder / 'after_bonus_update'))
            self.write_ledger()

    def _on_training_end(self):
        self.write_ledger()


def cases(start, count, random_seed):
    rng = np.random.default_rng(random_seed)
    return [{'seed': seed, 'amplitude': amplitude, 'period': float((2, 4, 8)[i % 3]),
             'phase': float(rng.uniform(0, 2*np.pi))}
            for amplitude in (.05, .1, .15) for i, seed in enumerate(range(start, start+count))]


def evaluate(model, config, conditions):
    rows = [simulate(model, config, **case) for case in conditions]
    return {'summary': aggregate(rows), 'episodes': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--output', type=Path, default=ROOT/'comfort_runs'/'v3')
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training_report.json').exists():
        raise RuntimeError('Output already contains a run; choose a new --output directory.')
    source = ROOT/'comfort_runs'/'comfort_v2_refined.zip'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    old_config = json.loads((ROOT/'comfort_runs'/'reward_v2.json').read_text())
    config = load_current_reward_config()
    assert config.version == 3 and config.failure_penalty == 10000
    write_json(folder/'reward_v3.json', config.to_dict())
    book = FeedbackRevisionBonus(old_config, config.to_dict())
    env = make_vec_env(lambda: make_training_env(config, book, SCALE), n_envs=8, seed=6210)
    model = PPO.load(str(source), env=env, device='cpu')
    previous = PPO.load(str(source), device='cpu')
    model.seed = 6210
    model.set_random_seed(6210)
    env.seed(6210)
    with torch.no_grad():
        model.policy.value_net.weight.mul_(.1)
        model.policy.value_net.bias.mul_(.1)
    model.learning_rate = 5e-5
    model.lr_schedule = lambda remaining: 5e-5
    model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(), lr=5e-5, **model.policy.optimizer_kwargs)
    callback = VerifyV3Bonus(book, folder)
    development = cases(12000, 12, 8831)
    report = {
        'source_checkpoint': str(source), 'source_sha256': source_hash,
        'reward': config.to_dict(), 'training_seed_after_load': 6210,
        'learning_rate': 5e-5, 'reward_scale': SCALE,
        'value_head_rescale': .1, 'optimizer_reset': True,
        'requested_additional_steps': args.steps,
        'selection_rule': 'Only post-bonus updated checkpoints; highest development survival, then highest V3 mean return. No holdout selection.',
        'development_conditions': development,
        'previous_policy_development': evaluate(previous, config, development),
        'checkpoints': [],
    }
    started = time.perf_counter()
    training_seconds = 0.
    first_step = model.num_timesteps
    best_key = (-1, -float('inf'))
    try:
        for chunk in range(4):
            tick = time.perf_counter()
            model.learn(total_timesteps=max(2048, args.steps//4), callback=callback, reset_num_timesteps=False)
            training_seconds += time.perf_counter() - tick
            added = model.num_timesteps-first_step
            model.save(str(folder/f'checkpoint_{added}'))
            evaluation = evaluate(model, config, development)
            report['checkpoints'].append({'additional_steps': added, **evaluation})
            summary = evaluation['summary']
            key = (summary['survival_rate'], summary['ordinary_return'])
            if callback.completed and key > best_key:
                best_key = key
                model.save(str(folder/'comfort_v3_selected'))
                report['selected_additional_steps'] = added
                report['selected_development_summary'] = summary
            report.update({'actual_additional_steps': added, 'training_seconds': training_seconds,
                           'elapsed_training_and_validation_seconds': time.perf_counter()-started})
            write_json(folder/'training_report.json', report)
            print(json.dumps({'additional_steps': added, **summary}), flush=True)
    finally:
        env.close()
    if not callback.completed:
        raise RuntimeError('No verified revision bonus update; do not claim a paid revision.')
    selected_path = folder/'comfort_v3_selected.zip'
    selected = PPO.load(str(selected_path), device='cpu')
    # Freeze the selected policy before opening this new held-out test set.
    holdout = cases(13000, 20, 8891)
    test = {'protocol': '60 fresh paired conditions; frozen policy and reward; no further tuning',
            'reward': config.to_dict(), 'conditions': holdout, 'models': {}}
    for name, candidate, path in (('previous', previous, source), ('final', selected, selected_path)):
        result = evaluate(candidate, config, holdout)
        test['models'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **result}
        print('HOLDOUT', name, json.dumps(result['summary']), flush=True)
    write_json(folder/'holdout_audit.json', test)
    scenarios = [
        {'key': 'gentle', 'label': 'Umeren vetar · poređenje', 'seed': 10001, 'amplitude': .1, 'period': 4., 'phase': 0.},
        {'key': 'fast', 'label': 'Brže promene vetra · poređenje', 'seed': 10007, 'amplitude': .15, 'period': 2., 'phase': 0.},
        {'key': 'failure', 'label': 'Prethodni neuspeh · provera popravke', 'seed': 10002, 'amplitude': .1, 'period': 8., 'phase': 3.7531335705119595},
    ]
    improved = next((new for old, new in zip(test['models']['previous']['episodes'], test['models']['final']['episodes'])
                     if not old['survived'] and new['survived']), None)
    if improved:
        scenarios.insert(0, {'key': 'improved', 'label': 'Popravljen primer · novi test',
                             **{k: improved[k] for k in ('seed', 'amplitude', 'period', 'phase')}})
    failed = next((r for r in test['models']['final']['episodes'] if not r['survived']), None)
    if failed:
        scenarios.append({'key': 'newfailure', 'label': 'Neuspeh novog modela · novi test',
                          **{k: failed[k] for k in ('seed', 'amplitude', 'period', 'phase')}})
    recordings = {'dt': .02, 'reward_version': 3, 'scenarios': []}
    for scenario in scenarios:
        record = dict(scenario)
        record['models'] = {}
        for name, candidate in (('v1', previous), ('final', selected)):
            row = simulate(candidate, config, **{k: scenario[k] for k in ('seed', 'amplitude', 'period', 'phase')}, record=True)
            row.pop('metrics')
            record['models'][name] = row
        recordings['scenarios'].append(record)
    (folder/'animation_recordings.json').write_text(json.dumps(recordings, separators=(',', ':'))+'\n')
    change_path = ROOT/'comfort_runs'/'reward_v3_change.json'
    change = json.loads(change_path.read_text())
    change.update({'training_status': 'completed', 'revision_bonus_credited': True,
                   'revision_bonus': BONUS, 'model_discovered_defect': False,
                   'selected_checkpoint': str(selected_path), 'ledger': str(folder/'revision_ledger.json'),
                   'note': 'V3 continuation complete. Bonus credited once for a live on-policy reproduction of the known failure class; no claim of autonomous discovery.'})
    write_json(change_path, change)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    report['total_seconds_including_holdout_and_recording'] = time.perf_counter()-started
    write_json(folder/'training_report.json', report)
    print('COMPLETE', json.dumps({k: report[k] for k in ('training_seconds', 'total_seconds_including_holdout_and_recording', 'selected_additional_steps')}), flush=True)


if __name__ == '__main__':
    main()
