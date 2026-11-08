"""Read-only retrospective evaluation for the five-slide bonus presentation."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from stable_baselines3 import PPO
from cartpole_comfort import RewardConfig
from comfort_experiment import simulate, aggregate


def read(path):
    return json.loads((ROOT / path).read_text())


def evaluate(path, config, conditions):
    path = ROOT / path
    model = PPO.load(str(path), device='cpu')
    rows = [simulate(model, config, **condition) for condition in conditions]
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'num_timesteps': model.num_timesteps, 'n_updates_counter': model._n_updates,
            'summary': aggregate(rows), 'episodes': rows}


def event_summary(path):
    ledger = read(path)
    event = ledger['events'][0]
    return {'ledger_path': str(ROOT / path), 'event': event,
            'episode_transition_duration_s': .02,
            'full_ppo_update_completed': ledger['full_ppo_update_completed'],
            'training_scale': ledger['training_reward_scale']}


def main():
    torch.set_num_threads(1)
    v2_config = RewardConfig(**read('comfort_runs/reward_v2.json'))
    v3_config = RewardConfig(**read('comfort_runs/reward_v3.json'))
    v2_conditions = [{'seed': seed, 'amplitude': amplitude, 'period': 4., 'phase': 0.}
                     for amplitude in (0., .1, .15) for seed in range(9000, 9010)]
    v3_training = read('comfort_runs/v3/training_report.json')
    v3_conditions = v3_training['development_conditions']
    result = {
        'protocol': 'Retrospective deterministic evaluation only. No training, policy selection or held-out test use. All comparison rows within a stage use the same reward without bonus.',
        'causality_warning': 'No matched no-bonus continuation. V2 also changed reward. V3 source-to-after-bonus spans four earlier PPO updates without bonus plus the update containing the bonus. These observations do not isolate the bonus effect.',
        'counter_note': 'SB3 _n_updates increments by n_epochs; counter +10 here is one completed PPO rollout optimization with 10 epochs, not ten separate rollout batches.',
        'v2': {'conditions': v2_conditions, 'reward': v2_config.to_dict(),
               'bonus': event_summary('comfort_runs/revision_ledger.json'), 'policies': {}},
        'v3': {'conditions': v3_conditions, 'reward': v3_config.to_dict(),
               'bonus': event_summary('comfort_runs/v3/revision_ledger.json'), 'policies': {}},
    }
    for name, path in (
        ('source_before_revision', 'comfort_runs/comfort_v1.zip'),
        ('after_first_bonus_update', 'comfort_runs/after_bonus_update.zip'),
        ('after_300k_continuation', 'comfort_runs/comfort_v2_final.zip'),
    ):
        row = evaluate(path, v2_config, v2_conditions)
        result['v2']['policies'][name] = row
        print('V2', name, json.dumps(row['summary']), flush=True)
    for name, path in (
        ('source_before_continuation', 'comfort_runs/comfort_v2_refined.zip'),
        ('after_bonus_update', 'comfort_runs/v3/after_bonus_update.zip'),
        ('selected_after_153600_steps', 'comfort_runs/v3/comfort_v3_selected.zip'),
    ):
        row = evaluate(path, v3_config, v3_conditions)
        result['v3']['policies'][name] = row
        print('V3', name, json.dumps(row['summary']), flush=True)
    out = ROOT / 'presentation_assets/bonus_checkpoint_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(str(out), flush=True)


if __name__ == '__main__':
    main()
