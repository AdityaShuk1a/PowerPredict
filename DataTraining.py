import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.decomposition import PCA

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)


def predictedData(data, region):
    future_predictions = []
    
    # Filter region first
    data = data[data['region'] == region]
    
    # Get last date for this region
    last_date = pd.to_datetime(data.iloc[-1]['interval_start_local'])
    print("Last date:", last_date)
    
    totalRows = data['region'].count()
    X = data.drop(columns=['interval_start_local', 'region', 'total_outages_mw'], axis=1)
    print("Training features:\n", X.head())
    Y = data['total_outages_mw']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # 'Linear Regression': LinearRegression()
        # 'MLP Regressor': MLPRegressor(max_iter=500),
    models = {
        'KNN Regressor': KNeighborsRegressor(n_neighbors=int(np.sqrt(totalRows))),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR (RBF)': SVR(),
        'Bagging Regressor': BaggingRegressor(),
        'AdaBoost Regressor': AdaBoostRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor()
    }

    best_name, best_model, best_score, best_r2Score = None, None, float('inf'), None
    
    # Competition loop
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        if mae < best_score:
            best_score, best_name, best_model, best_r2Score = mae, name, model, r2
            
    print(f"🏆 WINNER for {region}: {best_model} (MAE: {best_score:.2f} MW)")
    print("R² Score:", best_r2Score)
    
    # Forecast next 5 days
    for i in range(1, 6): 
        next_date = last_date + timedelta(days=i)
        day_of_week = next_date.dayofweek
        month = next_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        index = (i - 7) * -1
        features = pd.DataFrame([{
            "planned_outages_mw": data["planned_outages_mw"].iloc[index],
            "maintenance_outages_mw": data["maintenance_outages_mw"].iloc[index],
            "forced_outages_mw": data["forced_outages_mw"].iloc[index],
            'dayOfWeek': day_of_week,
            'month': month,
            'isWeekend': is_weekend
        }])
        
        pred_mw = best_model.predict(features)[0]
        
        future_predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_mw": round(pred_mw, 2)
        })
    
    print("Future predictions:", future_predictions)
    return best_name, best_score







if __name__ == "__main__":


    data = pd.read_csv("./Outage_Dataset.csv")

    print(data.describe())
    print(data.info())
    # print(data.shape())

    print(data)

    # Adding more field dayOfWeek, month, isWeekend in Outage_dataset

    data['interval_start_local'] = pd.to_datetime(data['interval_start_local'], utc=True).dt.tz_convert(None)
    data['dayOfWeek'] = data['interval_start_local'].dt.dayofweek
    data['month'] = data['interval_start_local'].dt.month
    data['isWeekend'] = data['dayOfWeek'].isin([5, 6]).astype(int)

    print(data)

    dropCols = ['interval_start_utc', 'interval_end_local', 'interval_end_utc', 'publish_time_local', 'publish_time_utc']

    data = data.drop(columns = dropCols, axis = 1)

    
    predictedData(data, "Mid Atlantic - Dominion")
    

