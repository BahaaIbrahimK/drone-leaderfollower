"""Learning script for multi-agent problems with off-policy algorithms.

Agents are based on `ray[rllib]`'s implementation of SAC, TD3, and DDPG with a custom centralized Q-critic.

Example
-------
To run the script, type in a terminal:

    $ python multiagent_offpolicy.py --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --num_workers <num_workers>

Notes
-----
Check Ray's status at:

    http://127.0.0.1:8265

"""
import os
import time
import argparse
from datetime import datetime
from sys import platform
import subprocess
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import sac, ddpg
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import ENV_STATE

from drones.utils.enums import DroneModel, Physics
from drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from drones.utils.Logger import Logger

import shared_constants

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# Centralized critic for DDPG: github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py

############################################################
class CustomTorchCentralizedQCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized Q-function.

    This is for off-policy algorithms (SAC, TD3, DDPG) which use Q(s,a) instead of V(s).

    Key difference from on-policy version:
    - Centralized critic takes BOTH agents' actions as input
    - Q-function: Q(own_obs, opponent_obs, own_action, opponent_action)
    - Actor still only uses own_obs (decentralized execution)

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A Q model that looks at everything (own_obs, opponent_obs, own_action, opponent_action)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Actor: only sees own observation (decentralized execution)
        self.action_model = FullyConnectedNetwork(
            Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE,)),
            action_space,
            num_outputs,
            model_config,
            name + "_action"
        )

        # Centralized Q-function: sees everything
        # Input: own_obs (12) + opponent_obs (12) + own_action (1 or 4) + opponent_action (1 or 4)
        # Total: 26 for 1D actions, 32 for 4D actions
        centralized_critic_obs_size = 2 * OWN_OBS_VEC_SIZE + 2 * ACTION_VEC_SIZE

        self.q_model = FullyConnectedNetwork(
            Box(low=-np.inf, high=np.inf, shape=(centralized_critic_obs_size,)),
            action_space,
            1,  # Q-value output
            model_config,
            name + "_q"
        )

        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        # Actor: only uses own observation
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        # Centralized Q-function uses full observation (with both agents' actions filled in)
        q_out, _ = self.q_model({"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
        return torch.reshape(q_out, [-1])

############################################################
class FillInBothActions(DefaultCallbacks):
    """Callback to fill in both agents' actions for centralized Q-critic.

    For off-policy algorithms, the Q-function needs both agents' actions:
    Q(own_obs, opponent_obs, own_action, opponent_action)
    """

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0

        action_encoder = ModelCatalog.get_preprocessor_for_space(
            Box(-1, 1, (ACTION_VEC_SIZE,), np.float32)
        )

        # Get opponent's actions
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array([action_encoder.transform(np.clip(a, -1, 1))
                                     for a in opponent_batch[SampleBatch.ACTIONS]])

        # Get own actions
        own_actions = np.array([action_encoder.transform(np.clip(a, -1, 1))
                               for a in postprocessed_batch[SampleBatch.ACTIONS]])

        # Fill in: [...own_obs(12), opponent_obs(12), own_action(1-4), opponent_action(1-4)]
        # Last 2*ACTION_VEC_SIZE values are for both actions
        to_update[:, -(2*ACTION_VEC_SIZE):-ACTION_VEC_SIZE] = own_actions
        to_update[:, -ACTION_VEC_SIZE:] = opponent_actions

############################################################
def central_qcritic_observer(agent_obs, **kw):
    """Observation function for centralized Q-critic.

    Returns dict with structure for each agent:
    - own_obs: agent's own observation
    - opponent_obs: other agent's observation
    - own_action: agent's own action (filled in by callback)
    - opponent_action: other agent's action (filled in by callback)
    """
    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "own_action": np.zeros(ACTION_VEC_SIZE),  # Filled in by FillInBothActions
            "opponent_action": np.zeros(ACTION_VEC_SIZE),  # Filled in by FillInBothActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "own_action": np.zeros(ACTION_VEC_SIZE),  # Filled in by FillInBothActions
            "opponent_action": np.zeros(ACTION_VEC_SIZE),  # Filled in by FillInBothActions
        },
    }
    return new_obs

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning with off-policy algorithms')
    parser.add_argument('--num_drones',  default=2,                 type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default='leaderfollower',  type=str,             choices=['leaderfollower', 'flock', 'meetup'],      help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs',         default='kin',             type=ObservationType,                                                     help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',         default='one_d_rpm',       type=ActionType,                                                          help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--algo',        default='sac',             type=str,             choices=['sac', 'td3', 'ddpg'],                     help='RL algorithm (default: sac)', metavar='')
    parser.add_argument('--workers',     default=0,                 type=int,                                                                 help='Number of RLlib workers (default: 0)', metavar='')
    ARGS = parser.parse_args()

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print out current git commit hash #####################
    if platform == "linux" or platform == "darwin":
        try:
            git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
        except subprocess.CalledProcessError:
            git_commit = b"unknown"
        with open(filename+'/git_commit.txt', 'w+') as f:
            f.write(str(git_commit))

    #### Constants, and errors #################################
    if ARGS.obs==ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 12
    elif ARGS.obs==ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit()
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ARGS.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VEC_SIZE = 1
    elif ARGS.act in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VEC_SIZE = 4
    elif ARGS.act == ActionType.PID:
        ACTION_VEC_SIZE = 3
    else:
        print("[ERROR] unknown ActionType")
        exit()

    #### Warnings ##############################################
    print(f"\n[INFO] Using off-policy algorithm: {ARGS.algo.upper()}")
    print(f"[INFO] Centralized Q-critic input size: {2*OWN_OBS_VEC_SIZE + 2*ACTION_VEC_SIZE}D")
    print(f"[INFO] (own_obs={OWN_OBS_VEC_SIZE} + opponent_obs={OWN_OBS_VEC_SIZE} + own_action={ACTION_VEC_SIZE} + opponent_action={ACTION_VEC_SIZE})\n")

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the custom centralized Q-critic model ########
    ModelCatalog.register_custom_model("cq_model", CustomTorchCentralizedQCriticModel)

    #### Register the environment ##############################
    temp_env_name = "this-aviary-v0"
    if ARGS.env == 'flock':
        register_env(temp_env_name, lambda _: FlockAviary(num_drones=ARGS.num_drones,
                                                          aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                          obs=ARGS.obs,
                                                          act=ARGS.act
                                                          )
                     )
    elif ARGS.env == 'leaderfollower':
        register_env(temp_env_name, lambda _: LeaderFollowerAviary(num_drones=ARGS.num_drones,
                                                                   aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                                   obs=ARGS.obs,
                                                                   act=ARGS.act
                                                                   )
                     )
    elif ARGS.env == 'meetup':
        register_env(temp_env_name, lambda _: MeetupAviary(num_drones=ARGS.num_drones,
                                                           aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                           obs=ARGS.obs,
                                                           act=ARGS.act
                                                           )
                     )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    #### Unused env to extract the act and obs spaces ##########
    if ARGS.env == 'flock':
        temp_env = FlockAviary(num_drones=ARGS.num_drones,
                               aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                               obs=ARGS.obs,
                               act=ARGS.act
                               )
    elif ARGS.env == 'leaderfollower':
        temp_env = LeaderFollowerAviary(num_drones=ARGS.num_drones,
                                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                        obs=ARGS.obs,
                                        act=ARGS.act
                                        )
    elif ARGS.env == 'meetup':
        temp_env = MeetupAviary(num_drones=ARGS.num_drones,
                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                obs=ARGS.obs,
                                act=ARGS.act
                                )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    # Observer space includes both actions for centralized Q-critic
    observer_space = Dict({
        "own_obs": temp_env.observation_space[0],
        "opponent_obs": temp_env.observation_space[0],
        "own_action": temp_env.action_space[0],
        "opponent_action": temp_env.action_space[0],
    })
    action_space = temp_env.action_space[0]

    #### Set up the trainer's config ###########################
    if ARGS.algo == 'sac':
        config = sac.DEFAULT_CONFIG.copy()
    elif ARGS.algo in ['td3', 'ddpg']:
        config = ddpg.DEFAULT_CONFIG.copy()
    else:
        print(f"[ERROR] Unknown algorithm: {ARGS.algo}")
        exit()

    config.update({
        "env": temp_env_name,
        "num_workers": 0 + ARGS.workers,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "callbacks": FillInBothActions,
        "framework": "torch",
    })

    # TD3 needs two critics, delayed policy updates, and smoothed target policies.
    # RLlib does not flip these switches just because we run the TD3 trainer name,
    # so wire them manually when the user selects TD3.
    if ARGS.algo == 'td3':
        config.update({
            "twin_q": True,
            "policy_delay": 2,
            "smooth_target_policy": True,
            "target_noise": 0.2,
            "target_noise_clip": 0.5,
        })

    #### Set up the model parameters of the trainer's config ###
    config["model"] = {
        "custom_model": "cq_model",
    }

    #### Set up the multiagent params of the trainer's config ##
    config["multiagent"] = {
        "policies": {
            "pol0": (None, observer_space, action_space, {"agent_id": 0,}),
            "pol1": (None, observer_space, action_space, {"agent_id": 1,}),
        },
        "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1",
        "observation_fn": central_qcritic_observer,
    }

    #### Ray Tune stopping conditions ##########################
    stop = {
        "timesteps_total": 120000,
    }

    #### Train #################################################
    print(f"\n[INFO] Starting training with {ARGS.algo.upper()}...")
    results = tune.run(
        ARGS.algo.upper(),
        stop=stop,
        config=config,
        verbose=True,
        checkpoint_at_end=True,
        local_dir=filename,
    )

    #### Save agent ############################################
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                   mode='max'
                                                                                   ),
                                                      metric='episode_reward_mean'
                                                      )
    with open(filename+'/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])

    print(f"\n[INFO] Training completed. Results saved to: {filename}")

    #### Shut down Ray #########################################
    ray.shutdown()
