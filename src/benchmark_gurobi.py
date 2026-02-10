import gymnasium as gym
from stable_baselines3 import PPO
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
from matplotlib.lines import Line2D
import sys
import os

# Ensure we can find battery_env.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from battery_env import BatteryEnv

# --- CONFIGURATION ---
MODEL_PATH = "models/PPO_SweetSpot_Gamma995_24HDA/ppo_sweetspot_final" # Insert your best model here
DATA_PATH = "data/train_data.csv"
START_DATE = "2025-01-01" 
DURATION_DAYS = 14 

def run_rl_agent(env, model, steps_to_run):
    print("🤖 Running RL Agent...")
    rl_profit = 0.0
    socs = []
    
    obs = env._get_obs()
    
    for _ in range(steps_to_run):
        if env.current_step >= len(env.df) - 1:
            break
            
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        rl_profit += info['step_profit']
        socs.append(info['soc'])
        
    return rl_profit, socs

def run_gurobi_solver(prices, current_capacity_mwh):
    print("🧮 Running Gurobi Solver...")
    
    T = len(prices)
    E_now = current_capacity_mwh
    P_max = E_now / 1.0 
    
    m = gp.Model("Battery_Opt")
    m.setParam('OutputFlag', 0)
    
    c = m.addVars(T, lb=0, ub=P_max, name="c")
    d = m.addVars(T, lb=0, ub=P_max, name="d")
    e = m.addVars(T, lb=0.2*E_now, ub=E_now, name="e") 
    
    current_stored = 0.5 * E_now 
    
    for t in range(T):
        prev_e = e[t-1] if t > 0 else current_stored
        m.addConstr(e[t] == prev_e + c[t]*0.9 - d[t]/0.9)
        
    obj = gp.quicksum((d[t] * prices[t] - c[t] * prices[t]) - 0.01*(c[t]+d[t]) for t in range(T))
    m.setObjective(obj, GRB.MAXIMIZE)
    
    try:
        m.optimize()
        if m.status == GRB.OPTIMAL:
            gurobi_socs = [e[t].X / E_now for t in range(T)]
            return m.ObjVal, gurobi_socs
        else:
            return 0, []
    except gp.GurobiError:
        return 0, []

def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print("❌ Data not found.")
        return
        
    df = pd.read_csv(DATA_PATH)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    
    try:
        target_timestamp = pd.to_datetime(START_DATE).tz_localize('UTC')
        start_index = df[df[date_col] >= target_timestamp].index[0]
        print(f"📅 Benchmark Date: {START_DATE}")
    except:
        start_index = 0
        print(f"⚠️ Date not found. Starting at index 0.")

    # 2. Slice Data
    test_len = DURATION_DAYS * 24
    if start_index + test_len > len(df):
        test_len = len(df) - start_index
    
    prices = df['SpotPriceDKK'].iloc[start_index : start_index + test_len].values
    plot_dates = df[date_col].iloc[start_index : start_index + test_len]

    # 3. Run Agents
    env = BatteryEnv(capacity_mwh=10.0, duration_hours=1.0, initial_soc=0.5)
    env.reset()
    env.current_step = start_index
    
    steps_passed = start_index
    env.capacity = max(0, env.initial_capacity - (steps_passed * env.degradation_per_step))
    
    model = PPO.load(MODEL_PATH, device='cpu')
    rl_score, rl_socs = run_rl_agent(env, model, test_len)
    gurobi_score, gurobi_socs = run_gurobi_solver(prices, env.capacity)

    efficiency = (rl_score / gurobi_score) * 100 if gurobi_score > 0 else 0
    print(f"\n📊 RESULTS: Efficiency {efficiency:.2f}%")

    # --- DRACULA PLOT SETUP ---
    DRACULA_BG = '#282a36'
    DRACULA_FG = '#f8f8f2'
    DRACULA_CYAN = '#8be9fd'   
    DRACULA_ORANGE = '#ffb86c' 
    DRACULA_PINK = '#ff79c6'   

    plt.rcParams.update({
        "figure.facecolor": DRACULA_BG,
        "axes.facecolor": DRACULA_BG,
        "axes.edgecolor": DRACULA_FG,
        "text.color": DRACULA_FG,
        "xtick.color": DRACULA_FG,
        "ytick.color": DRACULA_FG,
        "grid.color": DRACULA_FG,
        "axes.labelcolor": DRACULA_FG
    })

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Plot Price
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spot Price (DKK)', color=DRACULA_CYAN, fontsize=12, fontweight='bold')
    ax1.plot(plot_dates, prices, color=DRACULA_CYAN, alpha=0.3, linewidth=1)
    ax1.fill_between(plot_dates, prices, color=DRACULA_CYAN, alpha=0.1)
    ax1.tick_params(axis='y', labelcolor=DRACULA_CYAN)
    ax1.grid(visible=True, which='major', color=DRACULA_FG, linestyle='--', linewidth=0.5, alpha=0.1)

    # Plot SoC
    ax2 = ax1.twinx()
    ax2.set_ylabel('State of Charge (SoC)', color=DRACULA_FG, fontsize=12, fontweight='bold')
    
    # Gurobi (Pink Dashed)
    if len(gurobi_socs) > 0:
        ax2.plot(plot_dates, gurobi_socs, color=DRACULA_PINK, linewidth=2.5, 
                 linestyle='--') # Label removed here to avoid auto-legend issues

    # RL Agent (Orange Solid)
    ax2.plot(plot_dates, rl_socs, color=DRACULA_ORANGE, linewidth=2.5)
    
    try:
        mplcyberpunk.make_lines_glow(ax2, n_glow_lines=5, diff_linewidth=1.05, alpha_line=0.2)
    except: pass

    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor=DRACULA_FG)

    # Limits
    ax1.set_xlim(plot_dates.iloc[0], plot_dates.iloc[-1])
    ax1.set_ylim(bottom=0, top=max(prices)*1.1)
    ax1.margins(x=0)

    # Title
    plt.title(f"AI vs Optimal Strategy ({START_DATE})\nEfficiency: {efficiency:.1f}% | Profit: {rl_score:,.0f} vs {gurobi_score:,.0f} DKK", 
              fontsize=16, fontweight='bold', pad=20, color=DRACULA_FG)

    # --- CUSTOM LEGEND (THE FIX) ---
    # We manually create the legend handles to guarantee they look correct
    custom_lines = [
        Line2D([0], [0], color=DRACULA_CYAN, lw=2, label='Spot Price'),
        Line2D([0], [0], color=DRACULA_PINK, lw=2, linestyle='--', label='Gurobi (Optimal)'),
        Line2D([0], [0], color=DRACULA_ORANGE, lw=2, label='RL Agent (AI)')
    ]

    legend = fig.legend(handles=custom_lines, loc="upper right", 
                        bbox_to_anchor=(0.9, 0.88), 
                        facecolor=DRACULA_BG, edgecolor=DRACULA_FG)
    
    # Ensure legend text is white
    plt.setp(legend.get_texts(), color=DRACULA_FG)

    plt.tight_layout()
    output_filename = 'images/benchmark_comparison.png'
    os.makedirs('images', exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ Plot saved to '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    main()