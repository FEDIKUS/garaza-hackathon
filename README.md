# CartPole sensor red-team demo

A small neural network learns to balance a pole by moving a cart left or right.
PPO is the reinforcement-learning algorithm; the policy is a two-layer MLP with
64 units per hidden layer. Training also uses a separate value network.
This is a toy control-agent robustness experiment, not an LLM jailbreak demo.

## Run on this Mac

The Python 3.12 virtual environment is already installed in this folder.

```sh
cd /Users/filip/Downloads/hakagon
.venv/bin/python cartpole_redteam.py train
.venv/bin/python cartpole_redteam.py redteam
```

To reproduce the larger-bias failure demo:

```sh
.venv/bin/python cartpole_redteam.py redteam --epsilon 0.20 --test-seed 4000 --report redteam_results_020.json
```

Training saves `ppo_cartpole.zip` and `ppo_cartpole_training.json`.
Red-teaming saves `redteam_results.json`, including every trial and its seed.
The default is about 100,000 environment steps on CPU, with an approximate
15-minute training cap. Actual runtime and validation performance are recorded
in the training report; training outcomes can vary with seed and software.

Watch a clean episode and an episode with a biased sensor:

```sh
.venv/bin/python cartpole_redteam.py watch
.venv/bin/python cartpole_redteam.py watch --bias 0.20 --seed 4000
```

For the selected attack, use the `selected_bias_rad` value from the red-team report.

## What the red team controls

The attacker adds one constant bias to the pole-angle observation throughout
an episode. The default budget is at most 0.05 radians (about 2.86 degrees).
The physical simulator, reward, other observations, and actions are unmodified.

The script searches 11 bias values on 10 development seeds, freezes its choice,
then evaluates on 50 different seeds. Clean, attacked, and random-sign runs use
the same test seeds. The random baseline chooses either sign of the full attack
budget once per episode. A search need not outperform this simple baseline;
the report lets you check that instead of assuming it.

Each surviving step earns one point, up to 500. Physical failure is distinct
from reaching the time limit. Report both mean score and survival rate, and
only claim a vulnerability of a competent agent when clean performance is strong.
Results describe this one trained policy and this simulator, not real robots
or language models. Reusing test results to tune an attack requires fresh test seeds.

## Measured on this Mac

On an M1 Pro with 16 GB RAM, training seed 0 took 10.77 seconds for 100,096
environment steps (excluding installation, imports, and evaluation).
The policy and training-only value network together have 9,155 parameters.
Clean validation scored 500 in all 20 episodes.

Constant-bias searches with budgets 0.05 and 0.10 radians caused no failures
on their respective 50-episode test sets (starting at seeds 2000 and 3000).
Development-seed probes then found failures at +0.20 radians. A full search
within that budget selected +0.20, before evaluation on fresh test seeds
4000 through 4049:

| Condition | Mean score / 500 | Score standard deviation | Survived |
| --- | ---: | ---: | ---: |
| Clean | 500.0 | 0.0 | 50/50 |
| Random constant sign, +/-0.20 radians | 305.3 | 238.4 | 30/50 |
| Searched constant bias, +0.20 radians | 237.2 | 242.5 | 23/50 |

That attack is an 11.46-degree sensor error, a substantial perturbation.
These are observed results from one training seed, not a claim of statistical
superiority over random attacks. See `redteam_results_020.json` for all episodes.

## Install on another machine

The pinned environment was tested with Python 3.12 on Apple Silicon:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## References

- [Stable-Baselines3 PPO and its CPU guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Official RL Zoo CartPole hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)
- [Gymnasium CartPole observations, rewards, and termination](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## New model: wind on the pole, observed as input 5

`cartpole_wind.py` adds a horizontal, time-varying force at the **pole's center
of mass**, not directly on the cart. The pole and cart are mechanically coupled,
so this force can also cause cart motion. This is an idealized prescribed force
in newtons, not a wind-speed or aerodynamic drag simulation.

The new observation is `[cart position, cart velocity, pole angle, pole angular
velocity, current pole-wind force]`. The fifth value is the force used in the
next transition. The model gets the current force, not future wind values.
Reward remains +1 per step, including the terminating step; the 500-step limit
and physical failure thresholds are unchanged.

The saved **new** checkpoint is `ppo_cartpole_wind.zip`. It was trained from
scratch for 200,192 steps in 21.73 seconds with random sinusoidal profiles:
amplitude uniformly 0–0.15 N, period 2–8 seconds, and phase 0–2π per episode.
The original four-input checkpoint remains available separately.

```sh
.venv/bin/python -m unittest -v test_cartpole_wind.py
.venv/bin/python train_wind.py train
.venv/bin/python train_wind.py record
```

The training report is `ppo_cartpole_wind_training.json`; exact animation
recordings are `cartpole_wind_recordings.json`. Each recorded frame contains
the five observations, next action (null at episode end), and accumulated reward.
The interactive replay compares this same new policy with zero wind and with
the selected wind, from the same initial physical state (seed 4001). It uses
recorded policy decisions for each selectable configuration.

In this example, amplitude 0.10 N with a 4-second cycle survives 500 steps.
The same amplitude with a 2-second cycle terminates at step 139. These are
single-case outcomes, not guarantees. Values above 0.15 N are outside the
training amplitude range. Validation over 20 seeds scored 500 without wind;
randomized profiles in the training range had 90% survival. A fixed 0.15 N,
4-second profile was harder (mean score 203.9; no full-length survivors).

The physics tests cover exact agreement with the original environment at zero
wind, an independent coupled mass-matrix calculation with wind, force direction,
observation/transition timing, signal reversal, seeding, and the time limit.
The equations extend the [original Gymnasium implementation](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py)
with the generalized force and torque from a force at the pole's center.

## Comfort reward and a rewarded revision experiment

`cartpole_comfort.py` retains the five wind observations and pole-force physics,
but changes the action to a continuous motor force between −10 and +10 N,
including zero. The measured “head” is the massless pole tip, one metre above
the pivot when upright. This does not simulate a human body or predict nausea.

The initial engineering cost per 20 ms transition was:

```
C = .30 (head_acc / 1.5)^2 + .20 (head_jerk / 20)^2
  + .20 (filtered_head_acc / .3)^2 + .20 (angular_velocity / .35)^2
  + .10 (angle / .15)^2
r = 1 - min(2, C) - 1000 * physical_failure
```

Units are metres, seconds and radians. Jerk includes the first motor command,
relative to initial acceleration with zero motor force. Filtered acceleration
is the vector difference between first-order low-pass filters at 0.5 and 0.1 Hz,
initialized to zero. These filters and weights are an engineering choice, not
an ISO weighting or a clinical calibration. The model still receives only the
five instantaneous inputs, without filter history, previous action or a wind
forecast; this is a partially observed control problem.

The low-frequency term was motivated by the motion-sickness discussion in
[NASA's human-systems environmental guidance](https://www.nasa.gov/reference/6-0-natural-and-induced-environments-vol-2/).
It discusses low-frequency motion and exposure duration. Ten-second toy
episodes cannot validate human sickness over a long journey. Acceleration and
jerk are also relevant engineering comfort variables, but their weights here
are not inferred from a passenger study. See the primary study
[Standards for passenger comfort in automated vehicles: Acceleration and jerk](https://www.sciencedirect.com/science/article/pii/S0003687022002046).

### One confirmed defect and one actual bonus

The V1 audit found one small, reproducible ranking inversion (seed 9007,
zero wind, both controllers survive 500 steps). Direct policy actions earned
492.201 points versus 492.151 with action smoothing 0.5, despite a larger
intended weighted cost: 8.255 versus 7.849. The cap discarded 0.456 cost from a
startup jerk of 68.924 m/s³; smoothing reduced that peak to 34.462 m/s³.
This establishes a local reward-ranking defect, not deliberate intent.

V2 adds a decaying peak-jerk memory before applying the same cap:

```
q_t = max(head_jerk_t, q_previous * exp(-.02 / .5))
C_v2 = C_v1 + .20 * (q_t / 20)^2
```

That exact trajectory pair now scores 464.064 versus 484.427, correcting the
observed inversion. It does not prove the revised reward is unexploitable.

`comfort_revision.py` gives **+100,000** for the first live reproduction of
this accepted class: startup jerk above 40 m/s³ with positive discarded cost,
without physical failure. The bonus is returned by the environment, uniformly
scaled together with all ordinary rewards, stored in PPO's rollout buffer,
and included in a completed optimizer update. `revision_ledger.json` records
the actual transition, old/new specifications and update evidence. The bonus
dominates the bounded ordinary return in a credited episode, even if received
at the last step with gamma 0.995. PPO normalization and clipping still mean
that mathematical reward dominance is not proof of learned goal priority.

Credit occurred on the **first transition**, before the model learned from
the bonus. This was reproduction of an already audited behavior using a
preaccepted detector. We did not demonstrate learned autonomous discovery of
new hacks. The same class is paid once globally; after repair there is no
remaining bounty in this run. No additional defect was confirmed in the
subsequent 30-case smoothing audit, so no second revision was manufactured.
The detector is specific to this defect, not a general reward-hacking oracle.

### Training, selection and held-out result

The first continuous policy trained for about 400k steps. The revision stage
continued that policy for about 300k steps and preserved `comfort_v2_final.zip`.
Its development comfort worsened after the combined reward revision and bonus;
there is no matched no-bonus control to establish the cause.

A further 602,112 steps used the unchanged V2 reward. The selected final model,
`comfort_runs/comfort_v2_refined.zip`, is the +200,704 checkpoint, chosen by
highest V2 mean return on the declared 30 development cases. Later checkpoints
are preserved, including a +401,408 checkpoint with only 60% development
survival. Selection did not use the held-out cases. Details and hashes are in
`refinement_audit.json`; explicit continuation seed 6110 is applied after load.

After freezing the model and reward, 60 new paired conditions used physical
seeds 10000–10019, amplitudes 0.05/0.10/0.15 N, periods 2/4/8 s and randomized
phases. Both policies are evaluated under the same V2 objective, without bonus:

| Held-out measure | First continuous policy | Final selected policy |
| --- | ---: | ---: |
| Survived all 10 seconds | 56/60 | 48/60 |
| Mean V2 return | 301.876 | 175.120 |
| Mean episode head acceleration RMS, m/s² | 0.284 | 0.334 |
| Mean episode head jerk RMS, m/s³ | 4.500 | 2.609 |

RMS values use each episode's observed duration, which differs when a policy
fails; they are descriptive and do not make failures acceptable. The final
policy has less measured jerk but worse held-out balance and acceleration.
It is **not an overall improvement**. This is an observed control/generalization
failure; the evidence does not establish another reward exploit. No tuning was
performed on this held-out result.

`animation_recordings.json` contains actual deterministic rollouts for three
illustrations and the first failed held-out case. Both controllers use identical
initial conditions and wind in each comparison. Playback freezes a failed
controller; the 100,000 training bonus is not displayed as a comfort score.

Reproduce the stages (these commands overwrite their stage outputs):

```sh
.venv/bin/python -m unittest -v test_cartpole_wind.py test_cartpole_comfort.py test_comfort_revision.py
.venv/bin/python comfort_experiment.py train
.venv/bin/python comfort_experiment.py compare
.venv/bin/python comfort_revision.py
.venv/bin/python comfort_refine.py
.venv/bin/python comfort_delivery.py
```

The accepted V2 specification is `comfort_runs/reward_v2_candidate.json`;
the revision script verifies that the preceding comparison found an inversion.
All 20 physics, comfort and payout-protocol tests passed. A later snapshot-copy
fix preserves revision evidence against future configuration mutation; it did
not change the executed training run or its ledger.

## Current reward V3: larger physical-failure penalty

User feedback increased the physical-failure penalty from 1,000 to **10,000**.
All V2 comfort terms, including jerk memory, remain the same:

```
r_v3 = 1 - min(2, C_v2) - 10000 * physical_failure
```

Both exceeding the pole-angle threshold and leaving the cart-position bounds
incur this penalty. Reaching the episode time limit does not. The penalty is
applied outside the cost cap, so it cannot be clipped down to two points.

New environments with no explicit configuration load `comfort_runs/reward_v3.json`
through `load_current_reward_config()`. `RewardConfig()` retains its historical
V1 defaults for reproducibility; historical training scripts explicitly select
V1/V2 and remain unchanged. Future V3 training must pass
`load_current_reward_config()` and save a new checkpoint.

V3 continuation is now trained in `comfort_train_v3.py`, starting from
`comfort_runs/comfort_v2_refined.zip`. The selected new checkpoint is
`comfort_runs/v3/comfort_v3_selected.zip`; previous checkpoints remain intact.

The requested revision bonus was increased to **200,000**, credited once on
the first live physical failure under PPO's ordinary stochastic actions. This
reproduces the known failure class behind the user's feedback; it is not a new
autonomous discovery. V3 applies from the start. At episode step263, the
returned reward was −10,001 + 200,000 = **189,999**. Uniform scaling by .001
gave PPO 189.99899; buffer verification and optimizer updates3710→3720 confirm
that the bonus actually entered learning. Once credited, it cannot be earned
again in this run. The ledger is `comfort_runs/v3/revision_ledger.json`.

V3's conservative ordinary episode bounds are [−10,500, +500]. Even at the
last step, the discounted 200,000 bonus is 16,396.35 with gamma.995, exceeding
that 11,000-point range. This describes the reward objective, not a guarantee
of what a finite PPO run learns.

Training used 8 environments, learning rate0.00005, and 204,800 additional
steps in **22.43 seconds**. Four checkpoints were compared on 36 mixed-period,
mixed-phase development conditions. Selection prioritized survival, then V3
return, and considered only checkpoints after the verified bonus update. The
selected checkpoint is at +153,600 steps. Development, holdout and recording
together took 36.19 seconds after the initial baseline validation.

With policy and reward frozen, 60 fresh paired test conditions used physical
seeds13000–13019, amplitudes0.05/0.10/0.15 N, periods2/4/8 s and random phases.
Both policies were evaluated under V3 without bonus:

| Fresh V3 test | Previous V2 policy | Selected V3 policy |
| --- | ---: | ---: |
| Survived all 10 seconds | 48/60 | 52/60 |
| Mean V3 return | −1633.086 | −955.687 |
| Mean episode head acceleration RMS, m/s² | 0.340 | 0.361 |
| Mean episode head jerk RMS, m/s³ | 2.859 | 2.356 |

These are observations on one test set, not a significance or reliability
guarantee. Acceleration worsened while survival and mean jerk improved; RMS
durations differ for failed episodes. The earlier illustrated seed10002 failure
is **not repaired**: the V3 policy exceeds the angle threshold at7.48 s versus
V2 at7.98 s. No tuning followed this held-out evaluation, and no matched
no-bonus control establishes the bonus's causal effect.

`comfort_runs/v3/animation_recordings.json` includes the earlier comparisons,
the first new test case that improved from failure to survival, and the first
new failure. All are actual deterministic policy trajectories. The earlier
animations remain historical. All **24** physics, comfort and payout tests pass.

```sh
.venv/bin/python -m unittest -v test_cartpole_wind.py test_cartpole_comfort.py test_comfort_revision.py test_comfort_train_v3.py
.venv/bin/python comfort_train_v3.py --output comfort_runs/v3_repeat
```

Use a new output directory for a repeat run; each run includes its own one-time
revision-credit experiment and should not be interpreted as a new discovery.
