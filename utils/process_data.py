import pandas as pd
import numpy as np
import os

def process_data():
    print("--- STARTING DATA TRANSFORMATION ---")
    
    # 1. Load Data
    print("Loading raw CSVs...")
    try:
        df_prices = pd.read_csv(os.path.join("data", "energinet_prices.csv"))
        df_forecast = pd.read_csv(os.path.join("data", "energinet_forecast.csv"))
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find data files. Did you run the fetch scripts? {e}")
        return

    # 2. Preprocessing & Merging
    # Convert string timestamps to datetime objects
    df_prices['HourUTC'] = pd.to_datetime(df_prices['HourUTC'])
    df_forecast['HourUTC'] = pd.to_datetime(df_forecast['HourUTC'])
    
    # Set Index
    df_prices.set_index('HourUTC', inplace=True)
    df_forecast.set_index('HourUTC', inplace=True)
    
    # Merge on Index (Inner join ensures we only keep rows where we have BOTH price and forecast)
    print("Merging datasets...")
    df = df_prices.join(df_forecast, how='inner')
    
    # Drop unnecessary columns if they exist
    cols_to_drop = ['HourDK', 'PriceArea', 'ForecastType'] # Clean up
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # Handle Missing Values (Forward fill, then 0)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    print(f"Merged Data Shape: {df.shape}")

    # 3. Feature Engineering: Time Cyclical Encoding
    # We want the agent to know 'Daily Cycle' (0-23) and 'Yearly Cycle' (Seasonality)
    
    # Extract integer features
    hours = df.index.hour
    day_of_year = df.index.dayofyear
    
    # Encode Hour (Period = 24)
    df['sin_hour'] = np.sin(2 * np.pi * hours / 24)
    df['cos_hour'] = np.cos(2 * np.pi * hours / 24)
    
    # Encode Day of Year (Period = 365)
    df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365)
    df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365)
    
    # 4. Normalization (Crucial for RL Stability)
    # We save the Mean/Std to 'un-normalize' later if we want to see real DKK profit
    
    norm_params = {}
    
    # List of columns to normalize
    cols_to_norm = ['SpotPriceDKK', 'SpotPriceEUR', 'WindPowerProg', 'SolarPowerProg']
    
    for col in cols_to_norm:
        if col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std()
            
            # Save params
            norm_params[col] = {'mean': mu, 'std': sigma}
            
            # Create new normalized column (Z-score)
            df[f'{col}_norm'] = (df[col] - mu) / sigma
            
    print("Normalization complete.")
    
    # 5. Save Final Training Data
    output_path = os.path.join("data", "train_data.csv")
    df.to_csv(output_path)
    
    # Save normalization parameters (simple text or json)
    # We will just print them for now, the Env can recalculate them or load them
    print(f"✅ SUCCESS: Processed data saved to {output_path}")
    print("Sample:")
    print(df[['SpotPriceDKK', 'SpotPriceDKK_norm', 'sin_hour', 'WindPowerProg_norm']].head())
    
    return df

if __name__ == "__main__":
    process_data()