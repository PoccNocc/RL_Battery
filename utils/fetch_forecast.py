import requests
import pandas as pd
from datetime import datetime, timedelta
import os

def get_forecast_data(start_date: str, end_date: str, area: str = "DK1"):
    """
    Fetches Wind and Solar Forecasts from Energinet.
    Dataset: 'Forecasts_Hour'
    
    Structure based on user sample:
    - Column: 'ForecastDayAhead' (The value we need)
    - Types: 'Offshore Wind', 'Onshore Wind', 'Solar'
    """
    print(f"--- FETCHING GRID FORECASTS ---")
    print(f"Target: {area}")
    print(f"Dataset: Forecasts_Hour (Day Ahead)")
    
    url = "https://api.energidataservice.dk/dataset/Forecasts_Hour"
    
    # We explicitly ask for ForecastDayAhead
    params = {
        'start': start_date,
        'end': end_date,
        'filter': f'{{"PriceArea":["{area}"]}}',
        'columns': 'HourUTC,ForecastType,ForecastDayAhead', 
        'sort': 'HourUTC ASC',
        'timezone': 'UTC',
        'limit': 0 
    }
    
    try:
        print("Requesting data... (This covers 10 years, please wait)")
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return None
            
        data = response.json().get('records', [])
        
        if not data:
            print(f"⚠️ WARNING: No forecast data found.")
            return None
            
        df = pd.DataFrame(data)
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        
        # --- PIVOTING ---
        # We transform the 'Long' format into columns for each type
        print("Pivoting data structure...")
        
        # Pivot parameters:
        # Index = Time
        # Columns = Type (Offshore Wind, Onshore Wind, Solar)
        # Values = The Forecast Number
        df_pivot = df.pivot_table(
            index='HourUTC', 
            columns='ForecastType', 
            values='ForecastDayAhead', 
            aggfunc='sum'
        )
        
        # Fill NaNs with 0 (e.g., if Solar is missing at night)
        df_pivot.fillna(0, inplace=True)
        
        # --- FEATURE ENGINEERING ---
        # Create standard columns for the RL agent
        # Note: We use .get() or check columns to be safe against missing keys
        
        # 1. Solar
        if 'Solar' in df_pivot.columns:
            df_pivot['SolarPowerProg'] = df_pivot['Solar']
        else:
            print("⚠️ Warning: 'Solar' column missing, filling with 0")
            df_pivot['SolarPowerProg'] = 0

        # 2. Wind (Sum of Onshore + Offshore)
        wind_sum = 0
        if 'Offshore Wind' in df_pivot.columns:
            wind_sum += df_pivot['Offshore Wind']
        if 'Onshore Wind' in df_pivot.columns:
            wind_sum += df_pivot['Onshore Wind']
            
        df_pivot['WindPowerProg'] = wind_sum
        
        # Keep only the final features
        df_final = df_pivot[['WindPowerProg', 'SolarPowerProg']]
        
        print(f"✅ Success! Fetched {len(df_final)} rows.")
        return df_final
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

if __name__ == "__main__":
    # 10 Years
    years_to_fetch = 10
    days_to_fetch = int(365.25 * years_to_fetch)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_fetch)
    
    str_start = start_date.strftime('%Y-%m-%d')
    str_end = end_date.strftime('%Y-%m-%d')
    
    # Execution
    df = get_forecast_data(str_start, str_end, area="DK1")
    
    if df is not None:
        output_dir = "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, "energinet_forecast.csv")
        df.to_csv(output_path)
        print(f"💾 Forecast data saved to {output_path}")
        print(df.head())