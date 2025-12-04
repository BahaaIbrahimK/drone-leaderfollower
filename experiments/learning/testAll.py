import glob, os, numpy as np, pandas as pd

TARGET_LEADER_Z = 0.1   #Normalized

def rmse(arr, target):
    return np.sqrt(np.mean((arr - target)**2))

for flight_dir in glob.glob("results/save-flight-ma-*"):
    z0 = pd.read_csv(os.path.join(flight_dir, "z0.csv"),names=["time", "z"])
    z1 = pd.read_csv(os.path.join(flight_dir, "z1.csv"),names=["time", "z"])
    z0 = z0["z"].to_numpy()
    z1 = z1["z"].to_numpy()
    leader_rmse = rmse(z0, TARGET_LEADER_Z)
    follower_rmse = rmse(z1, z0)  # follower should match leader’s z trace
    print(f"{flight_dir}: leader RMSE={leader_rmse:.3f}, follower RMSE={follower_rmse:.3f}")
