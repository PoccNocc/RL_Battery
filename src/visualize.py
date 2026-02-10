import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
import sys
import os

# Ensure we can find battery_env.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from battery_env import BatteryEnv

def evaluate():
    # --- CONFIGURATION ---
    model_path = "models/PPO_Relative_Forecast/ppo_relative_forecast_final" # Insert your best model here
    
    # 📅 SET START DATE HERE
    START_DATE = "2025-01-01" 
    DURATION_DAYS = 14
    
    print(f"--- LOADING MODEL: {model_path} ---")
    
    # 1. Load Data explicitly to find the start index
    data_path = "data/train_data.csv"
    if not os.path.exists(data_path):
        print("❌ Error: data/train_data.csv not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # Try to parse the date column (assuming it's the first column or named 'HourDK')
    # Adjust 'HourDK' if your column has a different name
    date_col = df.columns[0] 
    df[date_col] = pd.to_datetime(df[date_col], utc=True) # Ensure UTC for safety
    
    # Find the index for the target date
    try:
        target_timestamp = pd.to_datetime(START_DATE).tz_localize('UTC')
        start_index = df[df[date_col] >= target_timestamp].index[0]
        print(f"📅 Start Date: {START_DATE} found at Index: {start_index}")
    except IndexError:
        print(f"❌ Error: Date {START_DATE} not found in dataset. Starting from 0.")
        start_index = 0
    except Exception as e:
        print(f"⚠️ Warning: Could not parse dates ({e}). Using Index 0.")
        start_index = 0

    # 2. Initialize Environment
    env = BatteryEnv(
        capacity_mwh=10.0, 
        duration_hours=1.0, 
        min_soc=0.2,
        initial_soc=0.5,
        annual_degradation_pct=0.02
    )
    
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: Model not found at {model_path}.zip")
        return

    model = PPO.load(model_path, device='cpu')

    print(f"--- STARTING SIMULATION ({DURATION_DAYS} Days) ---")
    
    # 3. Manually Reset to Specific Date
    obs, info = env.reset(options={'eval': True})
    
    # FORCE THE TIME JUMP
    env.current_step = start_index
    
    # FORCE BATTERY AGING (Crucial for realism)
    # The battery is not new in 2025! It has degraded for 6 years.
    steps_passed = start_index
    degraded_capacity = env.initial_capacity - (steps_passed * env.degradation_per_step)
    env.capacity = max(0, degraded_capacity) # Ensure not negative
    
    print(f"🔋 Battery Age Adjusted: {env.capacity:.2f} MWh capacity (started at 10.0)")

    prices = []
    socs = []
    profits = []
    
    steps_to_plot = DURATION_DAYS * 24
    total_profit = 0.0
    
    for i in range(steps_to_plot):
        # Stop if we reach end of data
        if env.current_step >= len(env.df) - 1:
            break
            
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        prices.append(info['price'])
        socs.append(info['soc'])
        profits.append(info['step_profit'])
        total_profit += info['step_profit']
            
    print(f"--- RESULTS ({START_DATE} + {DURATION_DAYS} days) ---")
    print(f"Total Profit: {total_profit:,.2f} DKK")
    print(f"Average Price: {np.mean(prices):.2f} DKK")
    
    # --- PROFESSIONAL PLOTTING (DRACULA THEME) ---
    DRACULA_BG = '#282a36'
    DRACULA_FG = '#f8f8f2'
    DRACULA_CYAN = '#8be9fd'   
    DRACULA_ORANGE = '#ffb86c' 
    
    plt.rcParams.update({
        "figure.facecolor": DRACULA_BG,
        "axes.facecolor": DRACULA_BG,
        "axes.edgecolor": DRACULA_FG,
        "text.color": DRACULA_FG,
        "axes.labelcolor": DRACULA_FG,
        "xtick.color": DRACULA_FG,
        "ytick.color": DRACULA_FG,
        "grid.color": DRACULA_FG,
    })
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # X-Axis Labels (Dates instead of hours)
    # We create a date range for the plot x-axis
    plot_dates = df[date_col].iloc[start_index : start_index + len(prices)]
    
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spot Price (DKK/MWh)', color=DRACULA_CYAN, fontsize=12, fontweight='bold')
    
    # Plot Price
    ax1.plot(plot_dates, prices, color=DRACULA_CYAN, alpha=0.9, linewidth=1.5, label='Spot Price')
    ax1.fill_between(plot_dates, prices, color=DRACULA_CYAN, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=DRACULA_CYAN)
    ax1.grid(visible=True, which='major', color=DRACULA_FG, linestyle='--', linewidth=0.5, alpha=0.2)

    # Plot SoC
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Battery State of Charge (SoC)', color=DRACULA_ORANGE, fontsize=12, fontweight='bold')
    ax2.plot(plot_dates, socs, color=DRACULA_ORANGE, linewidth=3, label='Battery SoC')
    
    try:
        mplcyberpunk.make_lines_glow(ax2, n_glow_lines=5, diff_linewidth=1.05, alpha_line=0.3)
    except:
        pass # Fallback if library has issues
    
    ax2.tick_params(axis='y', labelcolor=DRACULA_ORANGE)
    ax2.set_ylim(0, 1.1) 

    # Axis Formatting
    ax1.set_xlim(plot_dates.iloc[0], plot_dates.iloc[-1])
    ax1.set_ylim(bottom=0, top=max(prices) * 1.1) 
    ax1.margins(x=0) 

    plt.title(f'AI Strategy: {START_DATE} (Profit: {total_profit:,.0f} DKK)\nBattery Health: {env.capacity:.2f} MWh / 10.0 MWh', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Clean Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels1 + labels2, handles1 + handles2))
    
    legend = fig.legend(by_label.values(), by_label.keys(), loc="upper right", 
                        bbox_to_anchor=(0.9, 0.88), 
                        facecolor=DRACULA_BG, edgecolor=DRACULA_FG)
    plt.setp(legend.get_texts(), color=DRACULA_FG)

    plt.tight_layout()
    
    output_filename = 'images/agent_strategy_2025.png'
    os.makedirs('images', exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ Plot saved to '{output_filename}'.")
    plt.show()

if __name__ == "__main__":
    evaluate()