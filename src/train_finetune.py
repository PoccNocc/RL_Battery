import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from battery_env import BatteryEnv

# --- CUSTOM WRAPPER TO FORCE 2022 DATA ---
class CrisisBatteryEnv(BatteryEnv):
    def reset(self, seed=None, options=None):
        # 1. Standard Gym Seeding (Corrected Line)
        gym.Env.reset(self, seed=seed) 
        
        # 2. FIND 2022 INDICES
        if not hasattr(self, 'crisis_start_idx'):
            # Convert date column to datetime if not already
            date_col = self.df.columns[0]
            self.df[date_col] = pd.to_datetime(self.df[date_col], utc=True)
            
            # Define Crisis Period (Late 2021 to End of 2022)
            mask = (self.df[date_col] >= "2021-09-01") & (self.df[date_col] <= "2022-12-31")
            self.valid_indices = self.df[mask].index.tolist()
            
        # 3. Pick a random start strictly within the Crisis period
        # We need a buffer (forecast_horizon) so we don't pick the very last index
        buffer = self.forecast_horizon + 24
        rand_idx = np.random.choice(self.valid_indices[:-buffer]) 
        self.current_step = rand_idx
        
        # 4. Simulate realistic aging (approximate)
        steps_passed = self.current_step
        self.capacity = self.initial_capacity - (steps_passed * self.degradation_per_step)
        
        self.soc = 0.5
        self.max_power_mw = self.capacity / self.duration
        
        return self._get_obs(), {}

def train_crisis():
    # 1. Config
    source_model = "models/PPO_Relative_Forecast/ppo_relative_forecast_final" # YOUR BEST MODEL
    new_model_dir = "models/PPO_Crisis_Tuned"
    os.makedirs(new_model_dir, exist_ok=True)

    print(f"--- STARTING CRISIS BOOTCAMP (Fine-Tuning on 2022) ---")

    # 2. Create the Crisis-Specific Environment
    NUM_CORES = 12
    # Important: We must allow the env to initialize first to avoid multiprocessing errors
    env = make_vec_env(
        CrisisBatteryEnv, 
        n_envs=NUM_CORES, 
        seed=42, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={'capacity_mwh': 10.0}
    )

    # 3. Load the Existing Brain
    # Check if source model exists first
    if not os.path.exists(source_model + ".zip"):
        print(f"❌ Error: Source model not found at {source_model}.zip")
        return

    print(f"Loading pre-trained brain from: {source_model}")
    model = PPO.load(source_model, env=env, device='cpu')

    # 4. Hyperparameters for Fine-Tuning
    model.learning_rate = 0.00005  # Slow learning rate to refine, not destroy
    model.ent_coef = 0.01          # Keep exploring

    # 5. Train for a short burst
    TIMESTEPS = 1_000_000 
    
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    
    final_path = os.path.join(new_model_dir, "ppo_crisis_final")
    model.save(final_path)
    print(f"✅ Crisis Tuning Complete. Model saved to {final_path}")
    
    env.close()

if __name__ == "__main__":
    train_crisis()