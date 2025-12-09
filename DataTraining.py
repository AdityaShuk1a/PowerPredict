import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

REGIONS = ['PJM RTO', 'Mid Atlantic - Dominion', 'Western']

def train_and_forecast(data, region, target_col, feature_cols):
    print(f"\nTarget: {target_col} | Using: {feature_cols} + Previous Day Value")
    
    region_data = data[data['region'] == region].copy()
    if region_data.empty:
        print(f"No data found for {region}")
        return []

    # Add previous day's value as a feature
    region_data['previous'] = region_data[target_col].shift(1)
    region_data = region_data.dropna()
    
    training_features = feature_cols + ['previous']
    
    X = region_data[training_features]
    Y = region_data[target_col]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    totalRows = len(region_data)

    models = {
        'Linear Regression': LinearRegression(),
        'KNN': make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=min(int(np.sqrt(totalRows)), 20))), 
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'Bagging': BaggingRegressor(random_state=42),
        'SVR': make_pipeline(StandardScaler(), SVR()),
        'MLP Neural Net': make_pipeline(StandardScaler(), MLPRegressor(random_state=42, max_iter=1000))
    }

    best_name, best_model, best_score, best_r2 = None, None, float('inf'), None

    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            if mae < best_score:
                best_score, best_name, best_model, best_r2 = mae, name, model, r2
        except Exception:
            continue

    print(f"Winner: {best_name} (MAE: {best_score:.2f}, R2: {best_r2:.2f})")

    last_date = pd.to_datetime(region_data.iloc[-1]['interval_start_local'])
    current_lag_value = region_data.iloc[-1][target_col]
    
    future_predictions = []

    for i in range(1, 6):
        next_date = last_date + timedelta(days=i)
        
        # Use data from 7 days ago as proxy for future features
        index = (i - 7)
        
        row_data = {}
        planned_proxy_val = 0
        
        for col in feature_cols:
            if col in ['dayOfWeek', 'month', 'isWeekend']:
                if col == 'dayOfWeek': row_data[col] = next_date.dayofweek
                elif col == 'month': row_data[col] = next_date.month
                elif col == 'isWeekend': row_data[col] = 1 if next_date.dayofweek >= 5 else 0
            else:
                val = region_data[col].iloc[index]
                row_data[col] = val
                
                if col == 'planned_outages_mw':
                    planned_proxy_val = val
        
        # Recursive step: use last known/predicted value
        row_data['previous'] = current_lag_value
        
        feat_df = pd.DataFrame([row_data])
        feat_df = feat_df[training_features]
        
        pred_val = best_model.predict(feat_df)[0]
        current_lag_value = pred_val
        
        future_predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "prediction": round(pred_val, 2),
            "planned_proxy": planned_proxy_val
        })

    print(f"   5-Day Forecast ({target_col}):")
    for p in future_predictions:
        print(f"      {p['date']}: {p['prediction']} MW")
        
    return future_predictions

if __name__ == "__main__":
    
    
    data = pd.read_csv("./Outage_Dataset.csv")
    print("Dataset loaded successfully.")
    

    try:
        data['interval_start_local'] = pd.to_datetime(data['interval_start_local'], utc=True).dt.tz_convert(None)
    except:
        data['interval_start_local'] = pd.to_datetime(data['interval_start_local'])

    data['dayOfWeek'] = data['interval_start_local'].dt.dayofweek
    data['month'] = data['interval_start_local'].dt.month
    data['isWeekend'] = data['dayOfWeek'].isin([5, 6]).astype(int)

    unique_regions = data['region'].unique()
    
    for region in unique_regions:
        print(f"\nProcessing Region: {region}")
        
        maint_preds = train_and_forecast(
            data, 
            region, 
            target_col='maintenance_outages_mw', 
            feature_cols=['forced_outages_mw', 'planned_outages_mw', 'dayOfWeek', 'month', 'isWeekend']
        )
        
        forced_preds = train_and_forecast(
            data, 
            region, 
            target_col='forced_outages_mw', 
            feature_cols=['maintenance_outages_mw', 'planned_outages_mw', 'dayOfWeek', 'month', 'isWeekend']
        )
        
        if maint_preds and forced_preds:
            print(f"\n   Total Outages Calculation (Predictions + Planned):")
            
            total_outage_forecast = []
            
            for m, f in zip(maint_preds, forced_preds):
                total_val = m['prediction'] + f['prediction'] + m['planned_proxy']
                
                total_outage_forecast.append({
                    "date": m['date'],
                    "total_mw": round(total_val, 2)
                })
                
                print(f"{m['date']}: {round(total_val, 2)} MW")