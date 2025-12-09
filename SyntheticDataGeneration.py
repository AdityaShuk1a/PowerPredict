import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_grid_data():
    print("Generating PJM Grid Data (Prices & Outages)...")
    
    regions = ['PJM RTO', 'Mid Atlantic - Dominion', 'Western']
    
    start_date = datetime(2015, 1, 1)
    end_date = datetime.now()
    dates = pd.date_range(start_date, end_date, freq='D')
    
    data = []
    
    for r in regions:
        for d in dates:
            month = d.month
            
            is_summer = month in [6, 7, 8]
            is_winter = month in [12, 1, 2]
            
            base_outage = 3000 if r == 'PJM RTO' else 1500
            seasonal_adder = 1000 if (is_summer or is_winter) else 0
            random_spike = np.random.exponential(500) 
            total_outage = base_outage + seasonal_adder + random_spike
            
            solar_price = np.random.normal(30, 5) 
            if is_winter: solar_price += 10
            
            wind_price = np.random.normal(25, 8)
            
            gas_price = np.random.normal(45, 10) 
            if is_winter: gas_price += 25 
            if is_summer: gas_price += 15 

            coal_price = np.random.normal(50, 5)
            if is_winter: coal_price += 5
            
            nuclear_price = np.abs(np.random.normal(8, 1))
            
            oil_price = np.random.normal(90, 15)
            if is_winter: oil_price += 20 
            
            hydro_price = np.random.normal(10, 3)
            if month in [3, 4, 5]: hydro_price -= 5 
            
            data.append({
                'interval_start_local': d.strftime('%Y-%m-%d'),
                'region': r,
                'total_outage_mw': round(total_outage, 2),
                'solar_price': round(solar_price, 2),
                'wind_price': round(wind_price, 2),
                'natural_gas_price': round(gas_price, 2),
                'coal_price': round(coal_price, 2),
                'nuclear_price': round(nuclear_price, 2),
                'oil_price': round(oil_price, 2),
                'hydro_price': round(hydro_price, 2)
            })
            
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = "PowerGridPriceData.csv"
    df.to_csv(filename, index=False)
    print(f"Success! Saved {len(df)} rows to {filename}")
    print("Columns:", df.columns.tolist())

if __name__ == "__main__":
    generate_grid_data()