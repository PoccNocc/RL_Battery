import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from battery_env import BatteryEnv

def train():
    # 1. Config
    models_dir = "models/PPO_Relative_Forecast"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    NUM_CORES = 12 
    print(f"--- STARTING SWEETSPOT TRAINING (Gamma 0.995) ---")
    print(f"🚀 Spawning {NUM_CORES} parallel environments...")

    env_kwargs = {
        'capacity_mwh': 10.0,
        'duration_hours': 1.0,
        'min_soc': 0.2,
        'initial_soc': 0.5,
        'annual_degradation_pct': 0.02
    }

    env = make_vec_env(
        BatteryEnv, 
        n_envs=NUM_CORES, 
        seed=42, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs=env_kwargs
    )

    # 2. Hyperparameters (THE UPGRADE)
    # Define a Custom Neural Network Architecture
    # Default is [64, 64]. We upgrade to [256, 256] to handle the 24h forecast.
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        device="cpu",
        
        # --- CRITICAL CHANGES ---
        learning_rate=0.0003,   # Slower, more precise
        n_steps=2048,           # Longer rollout per env to capture daily cycles
        batch_size=256,         
        gamma=0.99,            # Long-term horizon (1 week view)
        ent_coef=0.01,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs
    )

    # 3. Training Loop
    TIMESTEPS = 8_000_000 
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // NUM_CORES, 
        save_path=models_dir, 
        name_prefix="ppo_relative_forecast"
    )
    
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    final_path = os.path.join(models_dir, "ppo_relative_forecast_final")
    model.save(final_path)
    print(f"✅ Training Complete. Model saved to {final_path}")
    
    env.close()

if __name__ == "__main__":
    train()