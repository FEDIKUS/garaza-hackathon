"""Comfort proxy training, physical audits, and recorded reward-revision experiments."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from cartpole_comfort import RewardConfig, make_comfort_env


class ScaleReward(gym.RewardWrapper):
    """Scale ALL task rewards and bonuses equally, preserving the objective."""
    def __init__(self, env, factor=0.01):
        super().__init__(env)
        self.factor=factor

    def reward(self, reward):
        return reward * self.factor


def make_training_env(config, hook=None, scale=.01):
    return ScaleReward(make_comfort_env(config=config, reward_hook=hook, amplitude=.15, randomize=True),scale)


def simulate(model, config, seed=9000, amplitude=.1, period=4., smoothing=1., record=False, hook=None, phase=0.):
    env = make_comfort_env(config=config, reward_hook=hook, amplitude=amplitude, period=period, phase=phase)
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    previous_action = 0.
    ordinary_return = 0.
    actual_return = 0.
    frames, metrics, infos = [], [], []
    try:
        while not (terminated or truncated):
            predicted = float(model.predict(obs, deterministic=True)[0][0])
            action = smoothing * predicted + (1 - smoothing) * previous_action
            previous_action = action
            old_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
            ordinary = float(info.get("base_reward", reward))
            metric = dict(info["metrics"])
            metrics.append(metric)
            infos.append({"raw_cost":float(info["raw_cost"]), "clipped_cost":float(info["clipped_cost"])})
            if record:
                # State before action; cumulative rewards before this transition.
                frames.append([*[round(float(v), 7) for v in old_obs], round(10*action, 7), round(ordinary_return, 5), round(metric["head_acc"], 6), round(metric["head_jerk"], 6)])
            ordinary_return += ordinary
            actual_return += float(reward)
        if record:
            frames.append([*[round(float(v), 7) for v in obs], None, round(ordinary_return, 5), None, None])
        summaries = {}
        for name in ["head_acc", "head_jerk", "band_acc", "omega", "theta"]:
            values = np.array([m[name] for m in metrics])
            summaries[name+"_rms"] = float(np.sqrt(np.mean(values**2)))
            summaries[name+"_max"] = float(np.max(np.abs(values)))
        raw = np.array([i["raw_cost"] for i in infos])
        result = {
            "seed":seed,"amplitude":amplitude,"period":period,"phase":phase,"smoothing":smoothing,
            "steps":len(metrics),"survived":bool(truncated and not terminated),
            "ordinary_return":ordinary_return,"actual_return":actual_return,
            "raw_cost_sum":float(raw.sum()),"clipped_excess_sum":float(np.maximum(raw-2.,0).sum()),
            "clipped_fraction":float(np.mean(raw>2.)), **summaries,
        }
        if record:
            result["frames"]=frames
            result["metrics"]=metrics
        return result
    finally:
        env.close()


def audit(model, config, seeds, amplitudes=(0.,.1,.15), smoothings=(1.,)):
    return [simulate(model,config,seed,amplitude,smoothing=smoothing)
            for amplitude in amplitudes for seed in seeds for smoothing in smoothings]


def aggregate(rows):
    return {
        "episodes":len(rows),"survival_rate":float(np.mean([r["survived"] for r in rows])),
        **{key:float(np.mean([r[key] for r in rows])) for key in [
            "steps","ordinary_return","raw_cost_sum","clipped_excess_sum",
            "head_acc_rms","head_acc_max","head_jerk_rms","head_jerk_max",
            "band_acc_rms","omega_rms","omega_max","theta_rms"]},
    }


def train(args):
    config=RewardConfig()
    folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'reward_v1.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
    env=make_vec_env(lambda:make_training_env(config),n_envs=8,seed=args.seed)
    evaluation=make_vec_env(lambda:make_training_env(config),n_envs=1,seed=17000)
    callback=EvalCallback(evaluation,best_model_save_path=str(folder/'best_v1'),
        log_path=str(folder/'validation'),eval_freq=6250,n_eval_episodes=8,deterministic=True,verbose=0)
    model=PPO('MlpPolicy',env,device='cpu',seed=args.seed,
        n_steps=256,batch_size=256,n_epochs=10,gamma=.995,gae_lambda=.95,
        learning_rate=3e-4,clip_range=.2,ent_coef=0.,vf_coef=.5,
        policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]},'log_std_init':-1.5},verbose=0)
    print(f'Training continuous controller: {args.steps:,} steps...',flush=True)
    started=time.perf_counter()
    try:
        model.learn(args.steps,callback=callback)
        seconds=time.perf_counter()-started
        model.save(str(folder/'comfort_v1_last'))
        model=PPO.load(str(folder/'best_v1'/'best_model'),device='cpu')
        model.save(str(folder/'comfort_v1'))
    finally:
        env.close();evaluation.close()
    print(f'Training finished in {seconds:.1f}s; auditing selected checkpoint.',flush=True)
    rows=audit(model,config,range(9000,9010))
    report={'training_seconds':seconds,'requested_steps':args.steps,'reward':asdict(config),
            'selection':'best validation mean return, 8 random-wind episodes per check',
            'summary':aggregate(rows),'episodes':rows}
    (folder/'v1_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


def compare(args):
    folder=Path(args.output)
    config=RewardConfig(**json.loads((folder/'reward_v1.json').read_text()))
    model=PPO.load(str(folder/'comfort_v1'),device='cpu')
    rows=audit(model,config,range(9000,9010),smoothings=(1.,.8,.5,.25))
    (folder/'v1_smoothing_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
    for smoothing in (1.,.8,.5,.25):
        print(smoothing,json.dumps(aggregate([r for r in rows if r['smoothing']==smoothing])))
    inversions=[]
    for row in rows:
        if row['smoothing']!=1. or not row['survived']:
            continue
        alternatives=[r for r in rows if r['seed']==row['seed'] and r['amplitude']==row['amplitude'] and r['smoothing']!=1. and r['survived']]
        for other in alternatives:
            if row['ordinary_return']>other['ordinary_return'] and row['raw_cost_sum']>other['raw_cost_sum']:
                inversions.append({'policy':row,'smoother':other})
    (folder/'v1_ranking_inversions.json').write_text(json.dumps(inversions,indent=2)+'\n')
    print('Observed full-horizon reward/raw-cost ranking inversions:',len(inversions))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','compare'])
    parser.add_argument('--steps',type=int,default=400_000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',default='comfort_runs')
    args=parser.parse_args()
    torch.set_num_threads(1)
    train(args) if args.command=='train' else compare(args)


if __name__=='__main__':
    main()
