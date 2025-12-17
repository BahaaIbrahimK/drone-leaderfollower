# Comprehensive Code Review: Multi-Agent Drone Leader-Follower Training

## Table of Contents
1. [Training Flow - On-Policy (PPO, A2C)](#training-flow---on-policy-ppo-a2c)
2. [Training Flow - Off-Policy (SAC, TD3, DDPG)](#training-flow---off-policy-sac-td3-ddpg)
3. [Testing Flow](#testing-flow)
4. [Core Classes and Functions](#core-classes-and-functions)
5. [Algorithm-Specific Parameters](#algorithm-specific-parameters)

---

## Training Flow - On-Policy (PPO, A2C)

### Command
```bash
python multiagent.py --num_drones 2 --env leaderfollower --obs kin --act one_d_rpm --algo ppo --workers 0
```

### Execution Flow

#### 1. **Argument Parsing** (Lines 143-150)
```python
parser.add_argument('--algo', default='ppo', type=str, choices=['ppo', 'a2c'])
ARGS = parser.parse_args()
```
**Parameters:**
- `num_drones`: 2 (leader + follower)
- `env`: 'leaderfollower'
- `obs`: ObservationType.KIN
- `act`: ActionType.ONE_D_RPM
- `algo`: 'ppo' or 'a2c'
- `workers`: 0 (number of parallel workers)

#### 2. **Constants Setup** (Lines 167-183)
```python
if ARGS.obs == ObservationType.KIN:
    OWN_OBS_VEC_SIZE = 12  # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
if ARGS.act == ActionType.ONE_D_RPM:
    ACTION_VEC_SIZE = 1    # Single RPM value for all motors
```

#### 3. **Ray Initialization** (Lines 189-190)
```python
ray.shutdown()
ray.init(ignore_reinit_error=True)
```
Starts Ray distributed computing framework.

#### 4. **Custom Model Registration** (Line 193)
```python
ModelCatalog.register_custom_model("cc_model", CustomTorchCentralizedCriticModel)
```

**CustomTorchCentralizedCriticModel Class** (Lines 69-107):
- **Purpose**: Implements centralized training with decentralized execution (CTDE)
- **Components**:
  - `action_model`: Actor network (uses only own_obs: 12D)
  - `value_model`: Centralized critic (uses full observation: 25D)

**Model Architecture**:
```python
def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    # Actor: decentralized (12D input)
    self.action_model = FullyConnectedNetwork(
        Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE,)),  # 12D
        action_space,
        num_outputs,
        model_config,
        name + "_action"
    )

    # Critic: centralized (25D input)
    # Input: own_obs (12) + opponent_obs (12) + opponent_action (1) = 25D
    self.value_model = FullyConnectedNetwork(
        obs_space,  # Full observation space
        action_space,
        1,  # Single value output
        model_config,
        name + "_vf"
    )
```

#### 5. **Environment Registration** (Lines 196-220)
```python
register_env("this-aviary-v0", lambda _: LeaderFollowerAviary(
    num_drones=2,
    aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
    obs=ObservationType.KIN,
    act=ActionType.ONE_D_RPM
))
```

**LeaderFollowerAviary** (in `drones/envs/multi_agent_rl/LeaderFollowerAviary.py`):
- **Parent**: BaseMultiagentAviary
- **Observation Space**: Dict with 12D vector per agent
- **Action Space**: Box([-1, 1], (1,)) for 1D RPM
- **Reward Function**:
  ```python
  # Leader (Agent 0): minimize distance to target [0, 0, 0.6]
  reward_0 = -np.linalg.norm(state_0[0:3] - [0, 0, 0.6])

  # Follower (Agent 1): follow leader's position
  reward_1 = -np.linalg.norm(state_1[0:2] - state_0[0:2])  # XY distance
  reward_1 -= np.linalg.norm(state_1[2] - state_0[2])      # Z distance
  ```

#### 6. **Observation Function Setup** (Lines 124-137)
```python
def central_critic_observer(agent_obs, **kw):
    """
    Transforms observations for centralized critic.
    Input: {0: obs_0, 1: obs_1}  (12D each)
    Output: Enhanced observations with opponent info
    """
    new_obs = {
        0: {
            "own_obs": agent_obs[0],              # 12D
            "opponent_obs": agent_obs[1],         # 12D
            "opponent_action": np.zeros(ACTION_VEC_SIZE),  # 1D (filled later)
        },
        1: {
            "own_obs": agent_obs[1],              # 12D
            "opponent_obs": agent_obs[0],         # 12D
            "opponent_action": np.zeros(ACTION_VEC_SIZE),  # 1D (filled later)
        },
    }
    return new_obs
```

#### 7. **FillInActions Callback** (Lines 110-121)
```python
class FillInActions(DefaultCallbacks):
    """
    Fills in opponent's action for centralized critic.
    Called AFTER episode is collected, BEFORE training.
    """
    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id,
                                   policies, postprocessed_batch, original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0

        # Get opponent's actions from their batch
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array([np.clip(a, -1, 1)
                                     for a in opponent_batch[SampleBatch.ACTIONS]])

        # Fill in last ACTION_VEC_SIZE values with opponent's actions
        to_update[:, -ACTION_VEC_SIZE:] = opponent_actions
```

**Why this is needed**: The critic needs to see opponent's actions to properly evaluate the value function, but these aren't available until after both agents act.

#### 8. **Trainer Configuration** (Lines 259-282)
```python
config = {
    "env": "this-aviary-v0",
    "num_workers": 0,  # Number of parallel rollout workers
    "num_gpus": 0,     # GPU count
    "batch_mode": "complete_episodes",  # Wait for full episodes
    "callbacks": FillInActions,  # Fill opponent actions
    "framework": "torch",
}

config["model"] = {
    "custom_model": "cc_model",
}

config["multiagent"] = {
    "policies": {
        "pol0": (None, observer_space, action_space, {"agent_id": 0}),
        "pol1": (None, observer_space, action_space, {"agent_id": 1}),
    },
    "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1",
    "observation_fn": central_critic_observer,
}
```

**Key Parameters**:
- `policies`: Two separate policies (one per drone)
- `policy_mapping_fn`: Maps agent ID to policy
- `observation_fn`: Transforms observations for centralized critic

#### 9. **Training Execution** (Lines 292-299)
```python
stop = {
    "timesteps_total": 120000,  # Stop after 120k timesteps
}

results = tune.run(
    ARGS.algo.upper(),  # "PPO" or "A2C"
    stop=stop,
    config=config,
    verbose=True,
    checkpoint_at_end=True,
    local_dir=filename,
)
```

**Training Loop** (Ray RLlib internal):
1. **Rollout**: Collect episodes using current policies
   - Environment steps: ~5 seconds per episode
   - Both agents act in environment
   - Observations stored with placeholder opponent actions

2. **Postprocess**: FillInActions callback runs
   - Fills in opponent actions for critic

3. **Train**: Update policies using collected data
   - **PPO**: Clipped surrogate objective
   - **A2C**: Advantage actor-critic with entropy regularization

4. **Repeat** until 120k timesteps

---

## Training Flow - Off-Policy (SAC, TD3, DDPG)

### Command
```bash
python multiagent_offpolicy.py --num_drones 2 --env leaderfollower --obs kin --act one_d_rpm --algo td3 --workers 0
```

### Key Differences from On-Policy

#### 1. **Custom Model: Centralized Q-Critic** (Lines 67-118)
```python
class CustomTorchCentralizedQCriticModel(TorchModelV2, nn.Module):
    """
    Off-policy algorithms use Q(s,a) instead of V(s).
    Q-function needs BOTH agents' actions as input.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # Actor: decentralized (12D input)
        self.action_model = FullyConnectedNetwork(
            Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE,)),  # 12D
            action_space,
            num_outputs,
            model_config,
            name + "_action"
        )

        # Centralized Q-function (26D input for 1D actions)
        # Input: own_obs (12) + opponent_obs (12) + own_action (1) + opponent_action (1) = 26D
        centralized_critic_obs_size = 2 * OWN_OBS_VEC_SIZE + 2 * ACTION_VEC_SIZE

        self.q_model = FullyConnectedNetwork(
            Box(low=-np.inf, high=np.inf, shape=(centralized_critic_obs_size,)),
            action_space,
            1,  # Q-value output
            model_config,
            name + "_q"
        )
```

**Critical Difference**:
- **On-policy (V)**: `V(own_obs, opponent_obs, opponent_action)` - 25D input
- **Off-policy (Q)**: `Q(own_obs, opponent_obs, own_action, opponent_action)` - 26D input

#### 2. **FillInBothActions Callback** (Lines 121-148)
```python
class FillInBothActions(DefaultCallbacks):
    """
    Fills in BOTH agents' actions for centralized Q-critic.
    """
    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id,
                                   policies, postprocessed_batch, original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0

        # Get opponent's actions
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array([np.clip(a, -1, 1)
                                     for a in opponent_batch[SampleBatch.ACTIONS]])

        # Get own actions
        own_actions = np.array([np.clip(a, -1, 1)
                               for a in postprocessed_batch[SampleBatch.ACTIONS]])

        # Fill in: [...own_obs(12), opponent_obs(12), own_action(1), opponent_action(1)]
        to_update[:, -(2*ACTION_VEC_SIZE):-ACTION_VEC_SIZE] = own_actions
        to_update[:, -ACTION_VEC_SIZE:] = opponent_actions
```

#### 3. **Observation Function** (Lines 151-174)
```python
def central_qcritic_observer(agent_obs, **kw):
    """
    Adds placeholders for BOTH actions.
    """
    new_obs = {
        0: {
            "own_obs": agent_obs[0],                       # 12D
            "opponent_obs": agent_obs[1],                  # 12D
            "own_action": np.zeros(ACTION_VEC_SIZE),       # 1D (filled later)
            "opponent_action": np.zeros(ACTION_VEC_SIZE),  # 1D (filled later)
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "own_action": np.zeros(ACTION_VEC_SIZE),
            "opponent_action": np.zeros(ACTION_VEC_SIZE),
        },
    }
    return new_obs
```

#### 4. **Algorithm-Specific Configuration** (Lines 294-320)
```python
if ARGS.algo == 'sac':
    config = sac.DEFAULT_CONFIG.copy()
elif ARGS.algo in ['td3', 'ddpg']:
    config = ddpg.DEFAULT_CONFIG.copy()

config.update({
    "env": temp_env_name,
    "num_workers": 0,
    "num_gpus": 0,
    "callbacks": FillInBothActions,  # Different callback!
    "framework": "torch",
})

# TD3-specific settings
if ARGS.algo == 'td3':
    config.update({
        "twin_q": True,              # Use two Q-networks
        "policy_delay": 2,           # Update policy every 2 critic updates
        "smooth_target_policy": True,  # Add noise to target policy
        "target_noise": 0.2,         # Target policy noise
        "target_noise_clip": 0.5,    # Clip target noise
    })
```

**TD3 Special Features**:
- **Twin Q-networks**: Uses Q1 and Q2, takes minimum to reduce overestimation
- **Delayed Policy Updates**: Updates actor less frequently than critic
- **Target Policy Smoothing**: Adds noise to target actions for robustness

---

## Testing Flow

### Command
```bash
python test_multiagent.py --exp ./results/save-leaderfollower-2-td3-kin-one_d_rpm-12.05.2025_13.56.47
```

### Execution Flow

#### 1. **Parse Experiment Path** (Lines 132-144)
```python
NUM_DRONES = int(ARGS.exp.split("-")[2])  # 2
ALGO = ARGS.exp.split("-")[3].lower()     # 'td3'
OBS = ObservationType.KIN                  # From split("-")[4]
ACT = ActionType.ONE_D_RPM                 # From split("-")[5]
```

**Path Format**: `save-{env}-{num_drones}-{algo}-{obs}-{act}-{timestamp}`

#### 2. **Setup Same as Training** (Lines 165-258)
- Register custom model
- Register environment
- Create observation function
- Configure trainer

#### 3. **Algorithm Detection & Agent Creation** (Lines 229-266)
```python
if ALGO == 'ppo' or ALGO == 'cc':
    base_config = ppo.DEFAULT_CONFIG.copy()
    agent = ppo.PPOTrainer(config=config)
elif ALGO == 'a2c':
    base_config = A2C_DEFAULT_CONFIG.copy()
    agent = A2CTrainer(config=config)
else:
    raise ValueError(f"Unknown algorithm: {ALGO}")
```

**Note**: The script automatically detects which algorithm was used from the path!

#### 4. **Restore Checkpoint** (Lines 268-270)
```python
with open(ARGS.exp+'/checkpoint.txt', 'r+') as f:
    checkpoint = f.read()
agent.restore(checkpoint)
```

Loads trained weights into the agent.

#### 5. **Extract Policies** (Lines 273-278)
```python
policy0 = agent.get_policy("pol0")  # Leader policy
print("action model 0", policy0.model.action_model)
print("value model 0", policy0.model.value_model)

policy1 = agent.get_policy("pol1")  # Follower policy
print("action model 1", policy1.model.action_model)
print("value model 1", policy1.model.value_model)
```

#### 6. **Create Test Environment** (Lines 280-307)
```python
test_env = LeaderFollowerAviary(
    num_drones=NUM_DRONES,
    aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
    obs=OBS,
    act=ACT,
    gui=True,      # Render GUI
    record=False   # Don't record video
)
```

#### 7. **Evaluation Loop** (Lines 324-340)
```python
obs = test_env.reset()
for i in range(6*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)):  # 6 seconds
    # Deploy policies
    temp = {}
    # Note: Counterintuitive order to match training observation structure
    temp[0] = policy0.compute_single_action(np.hstack([action[1], obs[1], obs[0]]))
    temp[1] = policy1.compute_single_action(np.hstack([action[0], obs[0], obs[1]]))

    action = {0: temp[0][0], 1: temp[1][0]}
    obs, reward, done, info = test_env.step(action)
    test_env.render()

    # Log data
    logger.log(drone=j, timestamp=i/test_env.SIM_FREQ, state=..., control=...)
```

**Input to policy.compute_single_action**:
```python
# For agent 0:
[opponent_action(1), opponent_obs(12), own_obs(12)]  # Total: 25D

# Structure matches training:
# own_obs (12) + opponent_obs (12) + opponent_action (1)
```

---

## Core Classes and Functions

### 1. LeaderFollowerAviary

**Location**: `drones/envs/multi_agent_rl/LeaderFollowerAviary.py`

**Key Methods**:

```python
class LeaderFollowerAviary(BaseMultiagentAviary):
    def __init__(self, num_drones=2, obs=ObservationType.KIN, act=ActionType.RPM):
        """Initialize leader-follower environment"""

    def _computeReward(self):
        """
        Compute rewards for each agent.
        Returns: {0: reward_0, 1: reward_1}
        """
        # Leader reward: negative distance to target
        state_0 = self._getDroneStateVector(0)
        reward_0 = -np.linalg.norm(state_0[0:3] - np.array([0, 0, 0.6]))

        # Follower reward: negative distance to leader
        state_1 = self._getDroneStateVector(1)
        reward_1 = -np.linalg.norm(state_1[0:2] - state_0[0:2])
        reward_1 -= np.linalg.norm(state_1[2] - state_0[2])

        return {0: reward_0, 1: reward_1}

    def _computeDone(self):
        """Check if episode is done"""
        # Done after time limit or if drones crash

    def _computeInfo(self):
        """Additional info for logging"""
```

### 2. BaseSingleAgentAviary

**Location**: `drones/envs/single_agent_rl/BaseSingleAgentAviary.py`

**Action Types**:

```python
class ActionType(Enum):
    ONE_D_RPM = "one_d_rpm"  # Single value controls all motors
    RPM = "rpm"               # 4 values, one per motor
    PID = "pid"               # 3 values [x, y, z] target
    VEL = "vel"               # 4 values [vx, vy, vz, yaw_rate]
```

**Observation Types**:

```python
class ObservationType(Enum):
    KIN = "kin"  # 12D: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    RGB = "rgb"  # Image-based (not implemented for multi-agent)
```

### 3. Logger

**Location**: `drones/utils/Logger.py`

```python
class Logger:
    def __init__(self, logging_freq_hz, num_drones):
        """Initialize logger for recording flight data"""

    def log(self, drone, timestamp, state, control):
        """Log data for one drone at one timestep"""

    def save_as_csv(self, file_name):
        """Save logged data to CSV files"""

    def plot(self):
        """Generate plots of flight trajectories"""
```

---

## Algorithm-Specific Parameters

### PPO (Proximal Policy Optimization)

**Default Config** (from Ray RLlib):
```python
{
    # Rollout
    "num_sgd_iter": 30,           # Epochs per training iteration
    "sgd_minibatch_size": 128,    # Minibatch size
    "train_batch_size": 4000,     # Total batch size

    # Policy
    "lr": 5e-5,                   # Learning rate
    "clip_param": 0.3,            # PPO clipping parameter
    "vf_clip_param": 10.0,        # Value function clipping

    # Loss
    "vf_loss_coeff": 1.0,         # Value function loss coefficient
    "entropy_coeff": 0.0,         # Entropy bonus
    "kl_coeff": 0.2,              # KL divergence coefficient

    # GAE
    "lambda": 1.0,                # GAE lambda
    "gamma": 0.99,                # Discount factor
}
```

**Key Features**:
- **Clipped Objective**: Prevents large policy updates
- **Multiple Epochs**: Reuses data for multiple gradient updates
- **GAE**: Generalized Advantage Estimation for variance reduction

### A2C (Advantage Actor-Critic)

**Default Config**:
```python
{
    # Rollout
    "rollout_fragment_length": 20,  # Steps per rollout
    "train_batch_size": 512,        # Total batch size

    # Policy
    "lr": 1e-4,                     # Learning rate

    # Loss
    "vf_loss_coeff": 0.5,           # Value function loss coefficient
    "entropy_coeff": 0.01,          # Entropy bonus

    # Discount
    "gamma": 0.99,                  # Discount factor
}
```

**Key Features**:
- **Synchronous**: All workers step together
- **Single Update**: Uses data only once (on-policy)
- **Entropy Regularization**: Encourages exploration

### SAC (Soft Actor-Critic)

**Default Config**:
```python
{
    # Replay Buffer
    "buffer_size": 1000000,         # Replay buffer size
    "prioritized_replay": False,    # No prioritization

    # Training
    "train_batch_size": 256,        # Batch size
    "learning_starts": 1500,        # Start training after N steps
    "optimization_config": {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    },

    # SAC-specific
    "target_entropy": "auto",       # Auto-tune entropy
    "initial_alpha": 1.0,           # Entropy coefficient
    "target_network_update_freq": 1,  # Update target every step
    "tau": 0.005,                   # Soft update coefficient

    # Discount
    "gamma": 0.99,
}
```

**Key Features**:
- **Maximum Entropy**: Balances reward and entropy
- **Off-Policy**: Uses replay buffer
- **Auto-Tuning**: Automatically adjusts entropy coefficient

### TD3 (Twin Delayed DDPG)

**Default Config**:
```python
{
    # Replay Buffer
    "buffer_size": 1000000,
    "prioritized_replay": True,     # Use PER
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,

    # Training
    "train_batch_size": 256,
    "learning_starts": 1500,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,

    # TD3-specific
    "twin_q": True,                 # Use two Q-networks
    "policy_delay": 2,              # Update actor every 2 critic updates
    "smooth_target_policy": True,   # Add noise to target
    "target_noise": 0.2,            # Target noise std
    "target_noise_clip": 0.5,       # Clip target noise
    "target_network_update_freq": 0,  # Update every step (with tau)
    "tau": 0.002,                   # Soft update coefficient

    # Exploration
    "exploration_config": {
        "type": "OrnsteinUhlenbeckNoise",
        "ou_sigma": 0.2,
        "ou_theta": 0.15,
    },

    # Discount
    "gamma": 0.99,
}
```

**Key Features**:
- **Twin Q-networks**: Reduces overestimation bias
- **Delayed Policy Updates**: More stable learning
- **Target Policy Smoothing**: Adds robustness
- **Prioritized Replay**: Samples important transitions more often

### DDPG (Deep Deterministic Policy Gradient)

**Default Config**:
```python
{
    # Replay Buffer
    "buffer_size": 1000000,
    "prioritized_replay": False,

    # Training
    "train_batch_size": 256,
    "learning_starts": 1500,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,

    # DDPG-specific
    "twin_q": False,                # Single Q-network
    "target_network_update_freq": 0,
    "tau": 0.001,

    # Exploration
    "exploration_config": {
        "type": "OrnsteinUhlenbeckNoise",
        "ou_sigma": 0.2,
        "ou_theta": 0.15,
    },

    # Discount
    "gamma": 0.99,
}
```

**Key Features**:
- **Deterministic Policy**: No stochastic action selection
- **Off-Policy**: Uses replay buffer
- **Ornstein-Uhlenbeck Noise**: Temporally correlated exploration

---

## Data Flow Summary

### Training (On-Policy)
```
1. Environment → Raw observations {0: obs_0, 1: obs_1}
                 ↓
2. central_critic_observer() → Enhanced obs with placeholders
                 ↓
3. Policies act → Collect episode data
                 ↓
4. FillInActions callback → Fill opponent actions
                 ↓
5. Training → Update policies
   - Actor uses: own_obs (12D)
   - Critic uses: own_obs + opponent_obs + opponent_action (25D)
```

### Training (Off-Policy)
```
1. Environment → Raw observations
                 ↓
2. central_qcritic_observer() → Enhanced obs with placeholders for BOTH actions
                 ↓
3. Policies act → Store in replay buffer
                 ↓
4. FillInBothActions callback → Fill own + opponent actions
                 ↓
5. Sample from buffer → Training
   - Actor uses: own_obs (12D)
   - Q-critic uses: own_obs + opponent_obs + own_action + opponent_action (26D)
```

### Testing
```
1. Load checkpoint → Restore trained policies
                 ↓
2. Environment.reset() → Get initial observations
                 ↓
3. Loop:
   a. Construct input: [opponent_action, opponent_obs, own_obs]
   b. policy.compute_single_action() → Get action
   c. Environment.step() → Execute actions
   d. Render & log
                 ↓
4. Save logs → CSV files + plots
```

---

## Critical Implementation Details

### 1. Why the Counterintuitive Order in Testing?

```python
# In test_multiagent.py line 327:
temp[0] = policy0.compute_single_action(np.hstack([action[1], obs[1], obs[0]]))
```

**Reason**: The observation structure from `central_critic_observer()` is:
```python
{
    "own_obs": agent_obs[0],      # Goes LAST
    "opponent_obs": agent_obs[1], # Goes MIDDLE
    "opponent_action": np.zeros(1), # Goes FIRST
}
```

When flattened, it becomes: `[opponent_action, opponent_obs, own_obs]`

### 2. Why Two Callbacks?

- **On-policy (FillInActions)**: Only needs opponent's action (critic uses V(s))
- **Off-policy (FillInBothActions)**: Needs both actions (Q-function uses Q(s,a))

### 3. Why Centralized Critic?

**Problem**: Independent learners suffer from non-stationarity
- Each agent sees environment as non-stationary (other agents are changing)
- Hard to learn effective policies

**Solution**: Centralized critic
- Critic sees all agents' observations (and actions for Q-functions)
- More stable learning
- Actors still decentralized (can deploy without communication)

### 4. Episode Length

```python
# From shared_constants.py
AGGR_PHY_STEPS = 5  # Aggregate physics steps

# Simulation
SIM_FREQ = 240 Hz  # Physics frequency
CONTROL_FREQ = 240 / 5 = 48 Hz  # Control frequency

# Episode length: ~5 seconds = 5 * 48 = 240 control steps
```

---

## Conclusion

This implementation provides a complete framework for multi-agent reinforcement learning with:

1. **Flexibility**: 6 different algorithms, multiple action/observation types
2. **CTDE Architecture**: Centralized training for stability, decentralized execution for deployment
3. **Proper Off-Policy Support**: Centralized Q-critics for SAC, TD3, DDPG
4. **Comprehensive Testing**: Automatic algorithm detection and visualization
5. **Production-Ready**: Checkpointing, logging, and evaluation tools

The code successfully separates concerns between environment, algorithm, and evaluation, making it easy to experiment with different approaches while maintaining clean architecture.
