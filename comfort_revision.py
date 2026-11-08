"""One evidence-backed reward revision, credited inside real PPO rollouts."""

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from cartpole_comfort import RewardConfig
from comfort_experiment import make_training_env, audit, aggregate


BONUS=100_000.0
REWARD_SCALE=.001


class RevisionBook:
    """The same confirmed class earns exactly one global revision bonus."""
    def __init__(self, current, replacement, evidence):
        self.current=current
        self.replacement=replacement
        self.evidence=evidence
        self.event=None
        self.transitions=0

    def __call__(self, env, reward, info):
        self.transitions+=1
        info['revision_bonus']=0.
        # The audit independently demonstrated that clipping a startup shock
        # can invert the intended weighted-cost ranking. Credit a live
        # reproduction of this already accepted local defect, not a clock.
        reproduced=(env.episode_step==1 and info['raw_cost']>info['clipped_cost']+1e-6
                    and info['metrics']['head_jerk']>40 and not info['physical_failure'])
        if self.event is None and reproduced:
            old=asdict(self.current)
            new=asdict(self.replacement)
            self.event={
                'id':'clipped_startup_shock','transition':self.transitions,
                'trigger':'startup jerk >40 m/s^3 and positive cost discarded by flat cap; no physical failure',
                'interpretation':'reproduction of an audited reward-ranking defect; not proof of human nausea or intent',
                'evidence':deepcopy(self.evidence),
                'reward_before_bonus':float(reward),'revision_bonus':BONUS,
                'returned_reward':float(reward+BONUS),'training_scale':REWARD_SCALE,
                'metrics':dict(info['metrics']),'raw_cost':float(info['raw_cost']),
                'discarded_cost':float(info['raw_cost']-info['clipped_cost']),
                'wind_profile':{'amplitude':env.amplitude,'period':env.period,'phase':env.phase},
                'old_reward':old,'new_reward':deepcopy(new),
            }
            for name,value in new.items():
                setattr(self.current,name,value)
            info['revision_bonus']=BONUS
            info['revision_id']=self.event['id']
            info['next_reward_version']=self.current.version
            return reward+BONUS
        return reward


class VerifyBonusLearning(BaseCallback):
    def __init__(self, book, folder):
        super().__init__()
        self.book=book;self.folder=folder
        self.pending=[]
        self.completed=False
        self.updates_at_credit=None

    def write_ledger(self):
        data={
            'bonus':BONUS,'training_reward_scale':REWARD_SCALE,
            'discount':self.model.gamma,
            'ordinary_per_step_bounds':[-1,1],
            'worst_case_ordinary_episode_bounds':[-3500,500],
            'minimum_discounted_bonus_at_last_step':BONUS*self.model.gamma**499,
            'novelty':'one payout globally per accepted defect class; no repeats after repair',
            'events':[] if self.book.event is None else [self.book.event],
            'full_ppo_update_completed':self.completed,
        }
        (self.folder/'revision_ledger.json').write_text(json.dumps(data,indent=2)+'\n')

    def _on_step(self):
        for index,info in enumerate(self.locals['infos']):
            if info.get('revision_bonus',0)>0:
                position=self.model.rollout_buffer.pos
                scaled=float(self.locals['rewards'][index])
                expected=REWARD_SCALE*float(info['returned_reward'])
                if not np.isclose(scaled,expected,rtol=1e-6):
                    raise AssertionError('Bonus missing from PPO environment reward.')
                self.pending.append((position,index,scaled))
                self.updates_at_credit=self.model._n_updates
                self.book.event.update({'ppo_timestep':int(self.num_timesteps),
                    'rollout_position':int(position),'environment_index':index,
                    'ppo_scaled_reward':scaled,'ppo_updates_before_credit':self.updates_at_credit})
                print(f"Accepted reward revision; actual PPO reward includes +{BONUS:,.0f} bonus.",flush=True)
        return True

    def _on_rollout_end(self):
        for position,index,expected in self.pending:
            actual=float(self.model.rollout_buffer.rewards[position,index])
            if not np.isclose(actual,expected,rtol=1e-6):
                raise AssertionError('Credited reward was not stored in RolloutBuffer.')
            self.book.event['verified_in_rollout_buffer']=True
            self.book.event['raw_advantage_before_normalization']=float(self.model.rollout_buffer.advantages[position,index])
            self.book.event['return_target']=float(self.model.rollout_buffer.returns[position,index])
        self.pending=[]
        self.write_ledger()

    def _on_rollout_start(self):
        if self.updates_at_credit is not None and not self.completed and self.model._n_updates>self.updates_at_credit:
            self.completed=True
            self.book.event['ppo_updates_after_credit']=int(self.model._n_updates)
            self.model.save(str(self.folder/'after_bonus_update'))
            self.write_ledger()

    def _on_training_end(self):
        self.write_ledger()


def main():
    torch.set_num_threads(1)
    folder=Path('comfort_runs')
    config=RewardConfig(**json.loads((folder/'reward_v1.json').read_text()))
    replacement=RewardConfig(**json.loads((folder/'reward_v2_candidate.json').read_text()))
    inversions=json.loads((folder/'v1_ranking_inversions.json').read_text())
    if not inversions:
        raise RuntimeError('No accepted ranking defect: do not manufacture a revision or bonus.')
    book=RevisionBook(config,replacement,inversions[0])
    env=make_vec_env(lambda:make_training_env(config,book,REWARD_SCALE),n_envs=8,seed=25000)
    model=PPO.load(str(folder/'comfort_v1'),env=env,device='cpu')
    # New uniform reward scaling preserves the objective. Rescale the value
    # output to its corresponding units and reset optimizer moments.
    with torch.no_grad():
        model.policy.value_net.weight.mul_(REWARD_SCALE/.01)
        model.policy.value_net.bias.mul_(REWARD_SCALE/.01)
    model.learning_rate=1e-4
    model.lr_schedule=lambda remaining:1e-4
    model.policy.optimizer=model.policy.optimizer_class(model.policy.parameters(),lr=1e-4,**model.policy.optimizer_kwargs)
    callback=VerifyBonusLearning(book,folder)
    started=time.perf_counter()
    print('Training revision-seeking policy, with a globally novel +100,000 bonus...',flush=True)
    try:
        model.learn(total_timesteps=300_000,callback=callback,reset_num_timesteps=True)
        model.save(str(folder/'comfort_v2_final'))
    finally:
        env.close()
    if book.event is None or not callback.completed:
        raise RuntimeError('Revision did not receive a verified complete PPO update.')
    (folder/'reward_v2.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
    rows=audit(model,config,range(9000,9010))
    report={'training_seconds':time.perf_counter()-started,'reward':asdict(config),
            'summary':aggregate(rows),'episodes':rows,
            'selection':'last policy after 300k additional steps; not cherry-picked for comfort'}
    (folder/'v2_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
