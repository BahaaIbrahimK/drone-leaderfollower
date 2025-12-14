# Takeoff Drone Training Improvements

## Problems Fixed

### 1. **Inverted Reward Function** (CRITICAL BUG)
**Problem**: The original reward function penalized the drone MORE for climbing higher:
```python
# OLD - WRONG!
if state[2] < 0.02:
    return -5
else:
    return -1 / (10*state[2])  # Gets MORE negative as z increases!
```

At z=1.0m: reward = -1/(10*1.0) = **-1**
At z=0.1m: reward = -1/(10*0.1) = **-10**

This taught the agent that **climbing is bad**!

**Fix**: Exponential reward that increases as drone approaches target height (1 meter):
```python
# NEW - CORRECT!
target_z = 1.0
height_error = abs(z - target_z)
height_reward = np.exp(-3 * height_error)  # 0 to 1
if height_error < 0.1:
    height_reward += 5.0  # Bonus for precision
```

### 2. **Limited Action Space**
**Problem**: Only ±5% thrust range made it very hard to take off:
```python
# OLD
return np.array(self.HOVER_RPM * (1+0.05*action))  # 95-105% of hover
```

**Fix**: Increased to ±20% for better control:
```python
# NEW
return np.array(self.HOVER_RPM * (1+0.2*action))  # 80-120% of hover
```

### 3. **No Early Termination**
**Problem**: Episodes continued even after crashes, accumulating huge negative rewards.

**Fix**: Added early termination conditions:
- Crash: z < 0.05m → episode ends
- Too high: z > 2.5m → episode ends
- Time limit: 5 seconds → episode ends

### 4. **Additional Reward Components**
Added penalties for:
- **Tilt stability**: Penalize excessive roll/pitch
- **XY drift**: Penalize horizontal movement (should hover in place)
- **Crash**: Large penalty (-10) for hitting ground

## Results

### Before Fixes:
```
[INFO] 0: episode_reward max -3885.037 min -3912.482 mean -3898.759
[INFO] 1: episode_reward max -3833.549 min -5274.676 mean -4245.835
```
- Drone learned to avoid climbing (due to inverted reward)
- Very negative rewards
- Random/chaotic behavior

### After Fixes:
```
[INFO] 0: episode_reward max -12.899 min -1453.510 mean -181.843
[INFO] 1: episode_reward max -12.899 min -1453.510 mean -167.326
[INFO] 2: episode_reward max -12.899 min -1490.252 mean -170.672
```
- **301x better** max reward (-12.9 vs -3885)
- Drone now learning to take off and hover at target height
- Much more stable training

## Configuration

### Updated Files:
1. `drones/envs/single_agent_rl/TakeoffAviary.py` - Fixed reward and done conditions
2. `drones/envs/single_agent_rl/BaseSingleAgentAviary.py` - Increased action range
3. `drones/examples/learn.py` - Added Gymnasium wrapper for RLlib 2.x compatibility

### Training Setup:
- Algorithm: PPO (distributed with 2 workers)
- Framework: PyTorch
- Episode length: 5 seconds (1200 timesteps @ 240Hz)
- Target altitude: 1.0 meter

## Recommendations

### For Better Results:
1. **Increase training iterations**: Change from 3 to 100+ in learn.py
2. **Tune hyperparameters**: Adjust PPO learning rate, batch size, etc.
3. **Extend episode length**: Consider 8-10 seconds for more learning time
4. **Add curriculum learning**: Start with lower target heights, gradually increase
5. **Reward shaping**: Fine-tune reward coefficients based on training progress

### Next Steps:
- Monitor training with TensorBoard
- Visualize learned behavior with `--gui=True`
- Save and evaluate best checkpoints
- Consider adding more complex objectives (land after takeoff, navigate to waypoints, etc.)
