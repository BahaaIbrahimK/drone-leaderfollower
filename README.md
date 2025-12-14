# Multi-Agent Drone Leader-Follower Control

Multi-agent reinforcement learning for drone swarm control using the leader-follower paradigm. This project implements centralized training with decentralized execution (CTDE) with support for both on-policy (PPO, A2C) and off-policy (SAC, TD3, DDPG) algorithms.

## Overview

This project trains multiple drones in a leader-follower formation where:
- **Leader drone (Agent 0)**: Learns to reach and maintain a target altitude
- **Follower drones**: Learn to follow the leader's position while maintaining formation

## Key Features

- **Multiple RL Algorithms**: PPO, A2C (on-policy) and SAC, TD3, DDPG (off-policy)
- **Centralized Training with Decentralized Execution (CTDE)**:
  - **On-policy**: Centralized value function V(s) observing all agents' states
  - **Off-policy**: Centralized Q-function Q(s,a) observing all agents' states and actions
- **Decentralized Execution**: Each agent acts independently based on its own observations
- **Flexible Action Spaces**: 1D (simplified) or 4D (individual motor control)
- **PyBullet Physics**: Realistic drone dynamics simulation
- **Ray RLlib**: Distributed training framework

## Algorithms

### On-Policy Algorithms (Centralized Critic)
- **PPO**: Proximal Policy Optimization with centralized value function
- **A2C**: Advantage Actor-Critic with centralized value function
- **Script**: `multiagent.py`

### Off-Policy Algorithms (Centralized Q-Critic)
- **SAC**: Soft Actor-Critic with centralized Q-function
- **TD3**: Twin Delayed DDPG with centralized Q-function
- **DDPG**: Deep Deterministic Policy Gradient with centralized Q-function
- **Script**: `multiagent_offpolicy.py`

## Environment Details

### Observation Space (12D per agent)
- Position (x, y, z)
- Roll, Pitch, Yaw
- Linear velocity (vx, vy, vz)
- Angular velocity (wx, wy, wz)

### Action Space
- **1D (default)**: Single RPM adjustment value `[-1, 1]`
- **4D (optional)**: Individual RPM for each motor

### Reward Function
- **Leader**: Penalized by distance from target `[0, 0, 0.6]`
- **Followers**: Penalized by distance from leader's x,y position and z height

## Installation

```bash
# Create conda environment
conda create -n drone python=3.8
conda activate drone

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training with On-Policy Algorithms (PPO, A2C)

```bash
cd experiments/learning

# Train with PPO (default)
python multiagent.py --num_drones 2 --env leaderfollower --obs kin --act one_d_rpm --algo ppo

# Train with A2C
python multiagent.py --num_drones 2 --env leaderfollower --obs kin --act one_d_rpm --algo a2c
```

### Training with Off-Policy Algorithms (SAC, TD3, DDPG)

```bash
cd experiments/learning

# Train with TD3
python multiagent_offpolicy.py --num_drones 2 --env leaderfollower --obs kin --act one_d_rpm --algo td3

# Train with SAC
python multiagent_offpolicy.py --num_drones 2 --env leaderfollower --obs kin --act one_d_rpm --algo sac

# Train with DDPG
python multiagent_offpolicy.py --num_drones 2 --env leaderfollower --obs kin --act one_d_rpm --algo ddpg
```

**Training Arguments:**
- `--num_drones`: Number of drones (default: 2)
- `--env`: Environment type (leaderfollower, flock, meetup)
- `--obs`: Observation type (kin for kinematic)
- `--act`: Action type (one_d_rpm, rpm, pid)
- `--algo`: Algorithm (ppo, a2c for on-policy | sac, td3, ddpg for off-policy)
- `--workers`: Number of RLlib workers (default: 0)

### Testing Trained Models

```bash
cd experiments/learning

# Test a specific trained model (works for all algorithms)
python test_multiagent.py --exp ./results/save-leaderfollower-2-td3-kin-one_d_rpm-12.05.2025_13.56.47

# The script automatically detects the algorithm from the path
```

### Batch Evaluation

```bash
cd experiments/learning

# Evaluate all trained models and compute RMSE metrics
python testAll.py
```

This will output leader and follower RMSE for all flight trajectories.

## Project Structure

```
drone-leaderfollower/
├── drones/
│   ├── envs/
│   │   ├── multi_agent_rl/
│   │   │   ├── BaseMultiagentAviary.py   # Base multi-agent environment
│   │   │   └── LeaderFollowerAviary.py    # Leader-follower task
│   │   ├── single_agent_rl/
│   │   │   └── BaseSingleAgentAviary.py   # Action/observation types
│   │   └── BaseAviary.py                   # Core simulation
│   ├── control/                            # PID controllers
│   ├── utils/                              # Utility functions
│   └── assets/                             # Drone URDF models
├── experiments/
│   └── learning/
│       ├── multiagent.py                   # On-policy training (PPO, A2C)
│       ├── multiagent_offpolicy.py         # Off-policy training (SAC, TD3, DDPG)
│       ├── test_multiagent.py              # Testing script (all algorithms)
│       ├── testAll.py                      # Batch testing utility
│       └── shared_constants.py             # Configuration
├── IMPROVEMENTS.md                          # Training improvements documentation
├── requirements.txt                         # Python dependencies
└── README.md
```

## Architecture Details

### On-Policy: Centralized Critic (PPO, A2C)

**Training (Centralized):**
- Critic observes: `own_obs (12D) + opponent_obs (12D) + opponent_action (1D or 4D)`
- Total critic input: **25D** (for 1D actions) or **28D** (for 4D actions)
- Value function: `V(own_obs, opponent_obs, opponent_action)`

**Execution (Decentralized):**
- Actor uses only: `own_obs (12D)`
- Each drone acts independently with local observations

### Off-Policy: Centralized Q-Critic (SAC, TD3, DDPG)

**Training (Centralized):**
- Q-function observes: `own_obs (12D) + opponent_obs (12D) + own_action (1D) + opponent_action (1D)`
- Total Q-critic input: **26D** (for 1D actions) or **32D** (for 4D actions)
- Q-function: `Q(own_obs, opponent_obs, own_action, opponent_action)`

**Execution (Decentralized):**
- Actor uses only: `own_obs (12D)`
- Each drone acts independently with local observations

**Key Difference:**
- On-policy critic needs only opponent's action
- Off-policy Q-critic needs BOTH agents' actions (because Q-functions take actions as input)

## Training Details

- **Total Timesteps**: 120,000
- **Episode Length**: 5 seconds (configurable)
- **Physics Frequency**: 240 Hz
- **Aggregate Steps**: Configurable in `shared_constants.py`
- **Framework**: Ray RLlib 1.13.0
- **Backend**: PyTorch

## Results & Monitoring

Training results are saved in `experiments/learning/results/` with:
- TensorBoard logs
- Checkpoint files
- Evaluation metrics
- Git commit hash for reproducibility

View training progress:
```bash
tensorboard --logdir experiments/learning/results/
```

## Performance Comparison

Based on training experiments, different algorithms show varying performance:

- **PPO**: Stable, good baseline performance
- **A2C**: Faster training but potentially less stable
- **TD3**: Best performance among off-policy methods for continuous control
- **SAC**: Good exploration, robust to hyperparameters
- **DDPG**: Baseline off-policy, can be sensitive to hyperparameters

See `IMPROVEMENTS.md` for detailed training improvements and results analysis.

## Troubleshooting

### Common Issues

1. **Ray initialization errors**: Ensure no other Ray instances are running
   ```bash
   ray stop
   ```

2. **CUDA/GPU errors**: If you don't have a GPU, the code will automatically use CPU

3. **Import errors**: Make sure you're in the conda environment
   ```bash
   conda activate drone
   ```

4. **Training crashes**: Check that you have enough disk space for checkpoints and logs

## Dependencies

- Python 3.8+
- PyBullet >= 3.2.0
- Ray[rllib] == 1.13.0
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Gym >= 0.21.0
- Matplotlib >= 3.3.0

## Authors

**Eman Mohamed** (g202426580@kfupm.edu.sa)
**Bahaa Ibrahim** (g202510630@kfupm.edu.sa)

King Fahd University of Petroleum and Minerals (KFUPM)

### Contributions

This project represents original research and implementation work including:

- **Multi-Agent RL Algorithms**: Implementation of 6 different RL algorithms (PPO, A2C, SAC, TD3, DDPG) with centralized training and decentralized execution (CTDE)
- **Centralized Critic Architectures**: Novel implementations of centralized value functions (V) for on-policy methods and centralized Q-functions for off-policy methods
- **Leader-Follower Environment**: Custom multi-agent environment with specialized reward shaping for leader-follower coordination
- **Training Framework**: Comprehensive training scripts supporting both on-policy and off-policy methods with Ray RLlib
- **Evaluation Tools**: Testing and batch evaluation utilities with performance metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{drone-leaderfollower-2025,
  author = {Mohamed, Eman and Ibrahim, Bahaa},
  title = {Multi-Agent Drone Leader-Follower Control with Centralized Training},
  year = {2025},
  publisher = {GitHub},
  institution = {King Fahd University of Petroleum and Minerals},
  url = {https://github.com/BahaaIbrahimK/drone-leaderfollower}
}
```

This project builds upon the gym-pybullet-drones framework. Please also cite:

```bibtex
@misc{gym-pybullet-drones,
  author = {Panerati, Jacopo and others},
  title = {gym-pybullet-drones},
  year = {2021},
  publisher = {GitHub},
  url = {https://github.com/utiasDSL/gym-pybullet-drones}
}
```

## License

This project builds upon gym-pybullet-drones. Please refer to the original project's license.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
