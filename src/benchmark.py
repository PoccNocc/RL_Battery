import gymnasium as gym
from stable_baselines3 import PPO
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
import sys
import os
from tqdm import tqdm
from matplotlib.lines import Line2D

# Ensure we can find battery_env.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from battery_env import BatteryEnv

# --- CONFIGURATION ---
MODEL_PATH = "models/PPO_Relative_Forecast/ppo_relative_forecast_final" # Insert your best model here
DATA_PATH = "data/train_data.csv"
FORECAST_HORIZON = 24 

def run_rl_full_dataset(env, model):
    print("🤖 Running RL Agent on Full Dataset...")
    obs, info = env.reset(options={'eval': True})
    profits = []
    
    pbar = tqdm(total=env.max_steps)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture raw step profit
        profits.append(info['step_profit'])
        pbar.update(1)
        
        if terminated or truncated:
            break
            
    pbar.close()
    return profits

def solve_gurobi_step(prices, current_soc, current_capacity, efficiency=0.9):
    T = len(prices)
    m = gp.Model("MPC")
    m.setParam('OutputFlag', 0)
    
    P_max = current_capacity 
    c = m.addVars(T, lb=0, ub=P_max, name="c")
    d = m.addVars(T, lb=0, ub=P_max, name="d")
    e = m.addVars(T, lb=0.0, ub=current_capacity, name="e")
    
    start_energy = current_soc * current_capacity
    
    for t in range(T):
        prev_e = e[t-1] if t > 0 else start_energy
        m.addConstr(e[t] == prev_e + c[t]*efficiency - d[t]/efficiency)
        
    avg_price = sum(prices) / T
    terminal_value = e[T-1] * avg_price * efficiency
    
    obj = gp.quicksum((d[t] * prices[t] - c[t] * prices[t]) - 0.001*(c[t]+d[t]) for t in range(T))
    m.setObjective(obj + terminal_value, GRB.MAXIMIZE)
    
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        return c[0].X - d[0].X 
    else:
        return 0.0

def run_gurobi_rolling_full(df, capacity_mwh=10.0, annual_degradation=0.02):
    print("\n🧮 Running Gurobi Rolling Horizon...")
    prices = df['SpotPriceDKK'].values
    steps = len(prices) - FORECAST_HORIZON
    
    soc = 0.5
    capacity = capacity_mwh
    efficiency = 0.9
    
    hours_per_year = 8760
    deg_per_step = (capacity_mwh * annual_degradation) / hours_per_year
    
    step_profits = []
    
    pbar = tqdm(total=steps)
    
    for t in range(steps):
        window_prices = prices[t : t+FORECAST_HORIZON]
        net_flow = solve_gurobi_step(window_prices, soc, capacity, efficiency)
        
        current_price = prices[t]
        step_profit = 0.0
        current_energy = soc * capacity
        
        if net_flow > 0: # Charge
            space_left = capacity - current_energy
            actual_flow = min(net_flow, space_left / efficiency)
            energy_added = actual_flow * efficiency
            cost = actual_flow * current_price
            step_profit = -cost
            soc = (current_energy + energy_added) / capacity
            
        else: # Discharge
            energy_to_remove = abs(net_flow) / efficiency
            if energy_to_remove > current_energy: energy_to_remove = current_energy
            energy_delivered = energy_to_remove * efficiency
            revenue = energy_delivered * current_price
            step_profit = revenue
            soc = (current_energy - energy_to_remove) / capacity
        
        soc = np.clip(soc, 0.0, 1.0)
        capacity = max(0.1, capacity - deg_per_step)
        
        step_profits.append(step_profit)
        pbar.update(1)
        
    pbar.close()
    return step_profits

def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH): return
    df = pd.read_csv(DATA_PATH)
    
    # Ensure Date parsing
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    
    # 2. Setup & Run RL
    env = BatteryEnv(capacity_mwh=10.0, duration_hours=1.0, initial_soc=0.5)
    
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"❌ Model not found at {MODEL_PATH}")
        return
        
    model = PPO.load(MODEL_PATH, device='cpu')
    
    rl_profits = run_rl_full_dataset(env, model)
    gurobi_profits = run_gurobi_rolling_full(df)
    
    # 3. Align DataFrames
    # Truncate to the shortest length (usually determined by forecast horizon)
    min_len = min(len(rl_profits), len(gurobi_profits))
    
    # Create a Results DataFrame
    results_df = df.iloc[:min_len].copy()
    results_df['RL_Profit'] = rl_profits[:min_len]
    results_df['Gurobi_Profit'] = gurobi_profits[:min_len]
    results_df['Year'] = results_df[date_col].dt.year
    
    # 4. Yearly Aggregation
    yearly_stats = results_df.groupby('Year').agg({
        'RL_Profit': 'sum',
        'Gurobi_Profit': 'sum',
        date_col: 'count' # Count hours -> convert to days
    }).rename(columns={date_col: 'Hours'})
    
    yearly_stats['Days'] = round(yearly_stats['Hours'] / 24, 1)
    yearly_stats['Efficiency %'] = (yearly_stats['RL_Profit'] / yearly_stats['Gurobi_Profit']) * 100
    
    # --- PRINT TABLE ---
    print("\n" + "="*65)
    print(f"{'YEAR':<6} | {'DAYS':<6} | {'RL PROFIT (DKK)':<18} | {'GUROBI (DKK)':<18} | {'EFFICIENCY':<10}")
    print("-" * 65)
    
    for year, row in yearly_stats.iterrows():
        print(f"{year:<6} | {row['Days']:<6} | {row['RL_Profit']:,.0f}".ljust(33) + 
              f" | {row['Gurobi_Profit']:,.0f}".ljust(21) + 
              f" | {row['Efficiency %']:.1f}%")
    
    print("-" * 65)
    total_rl = yearly_stats['RL_Profit'].sum()
    total_gu = yearly_stats['Gurobi_Profit'].sum()
    total_eff = (total_rl / total_gu) * 100
    print(f"{'TOTAL':<6} | {'ALL':<6} | {total_rl:,.0f}".ljust(33) + 
          f" | {total_gu:,.0f}".ljust(21) + 
          f" | {total_eff:.1f}%")
    print("="*65 + "\n")

    # --- PLOTTING HISTOGRAM ---
    DRACULA_BG = '#282a36'
    DRACULA_FG = '#f8f8f2'
    DRACULA_CYAN = '#8be9fd'
    DRACULA_ORANGE = '#ffb86c'
    DRACULA_PINK = '#ff79c6'
    
    plt.rcParams.update({
        "figure.facecolor": DRACULA_BG,
        "axes.facecolor": DRACULA_BG, 
        "text.color": DRACULA_FG,
        "xtick.color": DRACULA_FG, "ytick.color": DRACULA_FG
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = yearly_stats.index
    x = np.arange(len(years))
    width = 0.35
    
    # Bars
    rects1 = ax.bar(x - width/2, yearly_stats['Gurobi_Profit'], width, label='Gurobi (Optimal)', color=DRACULA_PINK, alpha=0.9)
    rects2 = ax.bar(x + width/2, yearly_stats['RL_Profit'], width, label='RL Agent', color=DRACULA_ORANGE, alpha=0.9)
    
    # Labels & Title
    ax.set_ylabel('Profit (DKK)', fontsize=12, fontweight='bold', color=DRACULA_FG)
    ax.set_title(f'Annual Performance: AI vs Optimal (Total Eff: {total_eff:.1f}%)', fontsize=16, fontweight='bold', pad=20, color=DRACULA_FG)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=12)
    ax.legend(facecolor=DRACULA_BG, edgecolor=DRACULA_FG, labelcolor=DRACULA_FG)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color=DRACULA_FG)
    
    # Helper to add labels on top of bars
    def autolabel(rects, is_efficiency=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            label = f'{height/1e6:.1f}M' # Convert to Millions
            
            # If it's the RL bar, add the efficiency percentage
            if is_efficiency:
                eff = yearly_stats['Efficiency %'].iloc[i]
                ax.annotate(f'{eff:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 15), # shift up specifically for percentage
                            textcoords="offset points",
                            ha='center', va='bottom', color=DRACULA_ORANGE, fontweight='bold', fontsize=11)
                
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', color=DRACULA_FG)

    autolabel(rects1)
    autolabel(rects2, is_efficiency=True)
    
    plt.tight_layout()
    plt.savefig("images/yearly_performance_v2.png", dpi=300)
    print("✅ Yearly Histogram saved to images/yearly_performance_v2.png")
    plt.show()

if __name__ == "__main__":
    main()