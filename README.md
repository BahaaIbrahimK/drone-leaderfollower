# Multi-Agent Drone Leader-Follower Control

Multi-agent reinforcement learning for drone swarm control using the leader-follower paradigm. This project implements centralized training with decentralized execution (CTDE) using PPO with a centralized critic.

## Overview

This project trains multiple drones in a leader-follower formation where:
- **Leader drone (Agent 0)**: Learns to reach and maintain a target altitude
- **Follower drones**: Learn to follow the leader's position while maintaining formation

## Key Features

- **Centralized Critic Architecture**: Shared value function that observes all agents' states
- **Decentralized Execution**: Each agent acts independently based on its own observations
- **1D Action Space**: Simplified control using single RPM value for all motors
- **PyBullet Physics**: Realistic drone dynamics simulation
- **Ray RLlib**: Distributed training framework

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

### Training

```bash
cd experiments/learning
python multiagent.py --num_drones 2 --env leaderfollower --obs kin --act one_d_rpm --workers 0
```

**Arguments:**
- `--num_drones`: Number of drones (default: 2)
- `--env`: Environment type (leaderfollower)
- `--obs`: Observation type (kin for kinematic)
- `--act`: Action type (one_d_rpm, rpm, pid)
- `--workers`: Number of RLlib workers (default: 0)

### Testing

```bash
cd experiments/learning
python test_multiagent.py --exp /path/to/checkpoint
```

### Test All Trained Models

```bash
cd experiments/learning
python testAll.py
```

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
│       ├── multiagent.py                   # Training script
│       ├── test_multiagent.py              # Testing script
│       ├── testAll.py                      # Batch testing
│       └── shared_constants.py             # Configuration
└── README.md
```

## Algorithm: Centralized Critic

The implementation uses a **centralized critic** architecture:

### Training (Centralized)
- Critic observes: own_obs (12D) + opponent_obs (12D) + opponent_action (1D or 4D)
- Total critic input: 25D (for 1D actions) or 28D (for 4D actions)
- Enables better credit assignment and coordination learning

### Execution (Decentralized)
- Actor uses only: own_obs (12D)
- Each drone acts independently with local observations
- No communication required during deployment

## Training Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Ray RLlib 1.x
- **Total Timesteps**: 120,000
- **Episode Length**: 5 seconds
- **Physics Frequency**: 240 Hz
- **Aggregate Steps**: Configurable

## Results

Training results are saved in `experiments/learning/results/` with:
- TensorBoard logs
- Checkpoint files
- Evaluation metrics
- Git commit hash for reproducibility

View training progress:
```bash
tensorboard --logdir experiments/learning/results/
```

## Dependencies

- Python 3.8+
- PyBullet
- Ray[rllib] 1.x
- PyTorch
- NumPy
- Gym

## Citation

If you use this code, please cite the original gym-pybullet-drones framework:

```
@misc{gym-pybullet-drones,
  author = {Jacopo Panerati and others},
  title = {gym-pybullet-drones},
  year = {2021},
  publisher = {GitHub},
  url = {https://github.com/utiasDSL/gym-pybullet-drones}
}
```

## License

This project builds upon gym-pybullet-drones. Please refer to the original project's license.

## Author

**Bahaa Ibrahim** (bahaaibrahim117@gmail.com)

Multi-agent leader-follower implementation and training experiments.
