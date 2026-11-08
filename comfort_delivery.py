"""Frozen-policy holdout evaluation and actual trajectories for the animation."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from cartpole_comfort import RewardConfig
from comfort_experiment import simulate, aggregate


def main():
    torch.set_num_threads(1)
    folder=Path(__file__).resolve().parent/'comfort_runs'
    config=RewardConfig(**json.loads((folder/'reward_v2.json').read_text()))
    paths={'v1':folder/'comfort_v1.zip', 'final':folder/'comfort_v2_refined.zip'}
    models={name:PPO.load(str(path),device='cpu') for name,path in paths.items()}
    rng=np.random.default_rng(7729)
    cases=[{'seed':seed,'amplitude':amplitude,'period':float((2,4,8)[i%3]),
            'phase':float(rng.uniform(0,2*np.pi))}
           for amplitude in (.05,.1,.15) for i,seed in enumerate(range(10000,10020))]
    report={'protocol':'60 held-out paired conditions; no model or reward changes after this evaluation',
        'reward':config.to_dict(),'cases':cases,'models':{}}
    for name,model in models.items():
        rows=[simulate(model,config,**case) for case in cases]
        report['models'][name]={'path':str(paths[name]),
            'sha256':hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            'summary':aggregate(rows),'episodes':rows}
        print(name,json.dumps(report['models'][name]['summary']),flush=True)
    (folder/'holdout_audit.json').write_text(json.dumps(report,indent=2)+'\n')

    scenarios=[
        {'key':'gentle','label':'Umeren vetar · novi primer','seed':10001,'amplitude':.1,'period':4.},
        {'key':'fast','label':'Brže promene vetra · novi primer','seed':10007,'amplitude':.15,'period':2.},
        {'key':'startup','label':'Početni trzaj · razvojni primer','seed':9007,'amplitude':0.,'period':4.},
    ]
    first_failure=next((r for r in report['models']['final']['episodes'] if not r['survived']),None)
    if first_failure:
        scenarios.append({'key':'failure','label':'Neuspeh završnog modela · novi primer',
            **{k:first_failure[k] for k in ('seed','amplitude','period','phase')}})
    recordings={'dt':.02,'reward_version':2,'checkpoint_hashes':{k:v['sha256'] for k,v in report['models'].items()},'scenarios':[]}
    for scenario in scenarios:
        entry=dict(scenario)
        entry['models']={}
        for name,model in models.items():
            row=simulate(model,config,seed=scenario['seed'],amplitude=scenario['amplitude'],period=scenario['period'],phase=scenario.get('phase',0.),record=True)
            row.pop('metrics')
            entry['models'][name]=row
        recordings['scenarios'].append(entry)
    (folder/'animation_recordings.json').write_text(json.dumps(recordings,separators=(',',':'))+'\n')
    print(f'Recorded {len(scenarios)} paired animation cases.',flush=True)


if __name__=='__main__':
    main()
