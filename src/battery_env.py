import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class BatteryEnv(gym.Env):
    def __init__(self, data_path=None, 
                 capacity_mwh=10.0,       
                 duration_hours=1.0,      
                 min_soc=0.2,             
                 efficiency=0.9, 
                 initial_soc=0.5,
                 annual_degradation_pct=0.02): 
        super(BatteryEnv, self).__init__()

        # 1. Load Data
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train_data.csv")
        
        # Fallback check
        if not os.path.exists(data_path):
             parent = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train_data.csv")
             if os.path.exists(parent): data_path = parent
             
        self.df = pd.read_csv(data_path)
        
        # 2. Physics
        self.initial_capacity = capacity_mwh
        self.capacity = capacity_mwh 
        self.min_soc = min_soc
        self.duration = duration_hours
        self.efficiency = efficiency
        
        hours_per_year = 365 * 24
        self.degradation_per_step = (self.initial_capacity * annual_degradation_pct) / hours_per_year
        
        # 3. Data Setup
        # We use the REAL prices to calculate dynamic stats
        self.price_col = 'SpotPriceDKK' 
        
        self.obs_features = [
            'WindPowerProg_norm', 'SolarPowerProg_norm', 
            'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
        ]
        
        self.data_matrix = self.df[self.obs_features].values.astype(np.float32)
        self.real_prices = self.df[self.price_col].values.astype(np.float32)
        
        # Calculate Global Max for absolute scaling context
        self.global_max_price = self.df[self.price_col].max()
        
        self.forecast_horizon = 24 
        
        # 4. Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: 
        # [SoC, Cap] (2) + [Features] (6) + [Relative Forecast] (24) + [Avg Price Context] (1)
        # Total = 33
        obs_dim = 2 + len(self.obs_features) + self.forecast_horizon + 1
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.soc = initial_soc
        self.max_steps = len(self.df) - self.forecast_horizon - 1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and options.get('eval'):
            self.current_step = 0
            self.capacity = self.initial_capacity
        else:
            self.current_step = np.random.randint(0, self.max_steps)
            self.capacity = self.initial_capacity - (self.current_step * self.degradation_per_step)
            
        self.soc = 0.5
        self.max_power_mw = self.capacity / self.duration
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Physics
        cap_norm = self.capacity / self.initial_capacity
        phys_obs = np.array([self.soc, cap_norm], dtype=np.float32)
        
        # 2. Features
        current_feats = self.data_matrix[self.current_step]
        
        # 3. DYNAMIC FORECAST SCALING (The Fix)
        # Get raw future prices
        start = self.current_step
        end = self.current_step + self.forecast_horizon
        
        if end <= len(self.real_prices):
            raw_prices = self.real_prices[start:end]
        else:
            # Pad if at end
            available = self.real_prices[start:]
            raw_prices = np.pad(available, (0, self.forecast_horizon - len(available)), 'edge')
            
        # Calculate statistics for this specific window
        window_mean = np.mean(raw_prices)
        window_std = np.std(raw_prices) + 1e-6
        
        # RELATIVE PRICES: How high is this hour compared to the daily average
        relative_forecast = (raw_prices - window_mean) / window_std
        
        # CONTEXT: How expensive is today globally
        price_context = np.array([window_mean / self.global_max_price], dtype=np.float32)
        
        return np.concatenate([phys_obs, current_feats, relative_forecast, price_context]).astype(np.float32)

    def step(self, action):
        action_val = np.clip(float(action[0]), -1.0, 1.0)
        self.max_power_mw = self.capacity / self.duration
        
        power_mw = action_val * self.max_power_mw
        current_price = self.real_prices[self.current_step]
        
        reward = 0.0
        actual_added = 0.0
        actual_removed = 0.0
        
        # Physics Logic
        if power_mw > 0: # Charge
            energy_in = power_mw
            added = energy_in * self.efficiency
            space = (1.0 - self.soc) * self.capacity
            actual_added = min(added, space)
            self.soc += actual_added / self.capacity
            reward = - (actual_added / self.efficiency) * current_price
            
        elif power_mw < 0: # Discharge
            needed = abs(power_mw)
            available = max(0.0, self.soc - self.min_soc) * self.capacity
            actual_removed = min(needed, available)
            self.soc -= actual_removed / self.capacity
            reward = (actual_removed * self.efficiency) * current_price

        # Degradation & Penalty
        self.capacity = max(0.1, self.capacity - self.degradation_per_step)
        
        # LOWER PENALTY: The 1.0 penalty was scaring the agent in low-price years.
        # Reduced to 0.1 to encourage trading.
        reward -= 0.1 * (actual_added + actual_removed)

        # Scale Reward for Neural Net Stability
        scaled_reward = reward / 100.0 

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'step_profit': reward,
            'soc': self.soc,
            'capacity': self.capacity,
            'price': current_price
        }
        
        return self._get_obs(), scaled_reward, terminated, False, info