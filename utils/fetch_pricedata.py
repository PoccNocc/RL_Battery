import requests
import pandas as pd
from datetime import datetime, timedelta
import os

def get_energinet_data(start_date: str, end_date: str, area: str = "DK1"):
    """
    Fetches spot prices from Energinet DataService.
    
    Documentation Compliance:
    - Dataset: 'Elspotprices' (Day-ahead spot prices)
    - Filter: PriceArea (DK1/DK2)
    - Sort: HourUTC ASC (Critical for time-series alignment)
    - Limit: 0 (Disable pagination limit to get full history)
    """
    print(f"--- FETCHING DATA ---")
    print(f"Target: {area}")
    print(f"Period: {start_date} to {end_date}")
    
    # Official API Endpoint
    url = "https://api.energidataservice.dk/dataset/Elspotprices"
    
    # Parameters according to Energinet API guide
    params = {
        'start': start_date,
        'end': end_date,
        'filter': f'{{"PriceArea":["{area}"]}}',
        'columns': 'HourUTC,HourDK,SpotPriceDKK,SpotPriceEUR,PriceArea',
        'sort': 'HourUTC ASC',
        'timezone': 'UTC', # Explicitly request UTC handling
        'limit': 0         # 0 = Unlimited rows (Default is often 100)
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # Raise error for bad status codes (404, 500)
        
        data = response.json().get('records', [])
        
        if not data:
            print(f"⚠️ WARNING: API returned 0 rows. Check your date range.")
            return None
            
        df = pd.DataFrame(data)
        
        # --- TYPE SAFETY & INDEXING ---
        # Convert HourUTC to datetime objects
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        
        # Set the Index to UTC time (Critical for RL Agents)
        df.set_index('HourUTC', inplace=True)
        
        # Check for data freshness
        last_date = df.index[-1]
        print(f"Fetched {len(df)} rows.")
        print(f"Oldest data: {df.index[0]}")
        print(f"Newest data: {last_date}")
        
        # Warn if data is stale (older than 2 days)
        if (datetime.utcnow() - last_date).days > 2:
            print("⚠️ WARNING: The latest data point is over 48 hours old.")
            print("Note: 'Elspotprices' dataset sometimes lags. If crucial, we may switch to 'DayAheadPrices'.")
            
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API CONNECTION ERROR: {e}")
        return None

if __name__ == "__main__":
    # 1. Setup Dates (Last 365 Days)
    years_to_fetch = 11
    days_to_fetch = int(365.25 * years_to_fetch)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_fetch)
    
    # Format for API (YYYY-MM-DD)
    str_start = start_date.strftime('%Y-%m-%d')
    str_end = end_date.strftime('%Y-%m-%d')
    
    # 2. Fetch
    df = get_energinet_data(str_start, str_end, area="DK1")
    
    if df is not None:
        # 3. Save
        output_dir = "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, "energinet_prices.csv")
        df.to_csv(output_path)
        print(f"✅ SUCCESS: Data saved to {output_path}")
        print(df.head(3))