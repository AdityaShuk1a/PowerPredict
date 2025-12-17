import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def loadData(filepath):
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded file: {filepath}")
    except:
        print(f"File not found: {filepath}")
        return None

    try:
        data['interval_start_local'] = pd.to_datetime(data['interval_start_local'], utc=True).dt.tz_convert(None)
    except:
        data['interval_start_local'] = pd.to_datetime(data['interval_start_local'])

    data['dayOfWeek'] = data['interval_start_local'].dt.dayofweek
    data['month'] = data['interval_start_local'].dt.month
    data['isWeekend'] = data['dayOfWeek'].isin([5, 6]).astype(int)

    if 'total_outages_mw' in data.columns:
        data.rename(columns={'total_outages_mw': 'total_outage_mw'}, inplace=True)

    return data

def prepPriceData(data):
    if 'lmp_price' not in data.columns:
        data['lmp_price'] = np.random.normal(35, 10, len(data)) + (data['month'].isin([1, 2, 7, 8]) * 15)

    if 'solar_price' not in data.columns:
        data['solar_price'] = np.random.normal(30, 5, len(data))
    
    if 'wind_price' not in data.columns:
        data['wind_price'] = np.random.normal(25, 8, len(data))

    if 'natural_gas_price' not in data.columns:
        data['natural_gas_price'] = np.random.normal(45, 10, len(data))
        
    if 'coal_price' not in data.columns:
        data['coal_price'] = np.random.normal(50, 5, len(data))
        
    if 'nuclear_price' not in data.columns:
        data['nuclear_price'] = np.abs(np.random.normal(8, 1, len(data)))
        
    if 'oil_price' not in data.columns:
        data['oil_price'] = np.random.normal(90, 15, len(data))
        
    if 'hydro_price' not in data.columns:
        data['hydro_price'] = np.random.normal(10, 3, len(data))
        
    return data

def findBestModel(X, Y):
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=42)
    totalRows = len(X)
    
    models = {
        'Linear Regression': LinearRegression(),
        'KNN': KNeighborsRegressor(n_neighbors=min(int(np.sqrt(totalRows)), 20)), 
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR()
    }

    bestName, bestModel, bestScore = None, None, float('inf')

    for name, model in models.items():
        try:
            model.fit(xTrain, yTrain)
            preds = model.predict(xTest)
            mae = mean_absolute_error(yTest, preds)
            
            if mae < bestScore:
                bestScore, bestName, bestModel = mae, name, model
        except:
            continue
            
    return bestModel, bestName, bestScore

def trainAndForecast(outageDf, priceDf, region):
    print(f"\nProcessing region: {region}")
    
    outageData = outageDf[outageDf['region'] == region].copy()
    priceData = priceDf[priceDf['region'] == region].copy()
    
    if outageData.empty or priceData.empty:
        print("Region data missing")
        return

    print("Predicting total outage...")
    
    targetCol = 'total_outage_mw'
    
    outageData['previous'] = outageData[targetCol].shift(1)
    outageData = outageData.dropna()
    
    outageFeatures = ['dayOfWeek', 'month', 'isWeekend', 'previous']
    
    XOutage = outageData[outageFeatures]
    YOutage = outageData[targetCol]
    
    bestOutageModel, outageModelName, outageError = findBestModel(XOutage, YOutage)
    print(f"Best outage model: {outageModelName} | Error: {outageError:.2f} MW")
    
    lastDate = pd.to_datetime(outageData.iloc[-1]['interval_start_local'])
    currOutage = outageData.iloc[-1][targetCol]
    
    forecastedOutages = []
    futureDates = []
    
    for i in range(1, 6):
        nextDate = lastDate + timedelta(days=i)
        futureDates.append(nextDate)
        
        row = pd.DataFrame([{
            'dayOfWeek': nextDate.dayofweek,
            'month': nextDate.month,
            'isWeekend': 1 if nextDate.dayofweek >= 5 else 0,
            'previous': currOutage
        }])
        
        pred = bestOutageModel.predict(row)[0]
        forecastedOutages.append(pred)
        currOutage = pred

    print(f"5-Day Outage Forecast: {[round(x, 2) for x in forecastedOutages]}")

    print("Predicting prices based on forecasted outage...")
    
    priceCols = ['solar_price', 'wind_price', 'natural_gas_price', 
                 'coal_price', 'nuclear_price', 'oil_price', 'hydro_price']
    
    priceFeatures = ['total_outage_mw', 'dayOfWeek', 'month', 'isWeekend']
    
    XPriceTrain = priceData[priceFeatures]
    
    print("5-Day Price Forecast:")
    for dateObj, outageVal in zip(futureDates, forecastedOutages):
        print(f"Date: {dateObj.strftime('%Y-%m-%d')} | Outage: {round(outageVal, 2)} MW")
        
        inputRow = pd.DataFrame([{
            'total_outage_mw': outageVal,
            'dayOfWeek': dateObj.dayofweek,
            'month': dateObj.month,
            'isWeekend': 1 if dateObj.dayofweek >= 5 else 0
        }])
        
        for pCol in priceCols:
            YPriceTrain = priceData[pCol]
            
            model, modelName, modelError = findBestModel(XPriceTrain, YPriceTrain)
            predPrice = model.predict(inputRow)[0]
            print(f"   {pCol}: ${round(predPrice, 2)}")

if __name__ == "__main__":
    print("Loading outage dataset...")
    outageDf = loadData("./Outage_Dataset.csv")
    
    print("Loading price dataset...")
    priceDf = loadData("./PowerGridPriceData.csv")
    
    if outageDf is not None and priceDf is not None:
        priceDf = prepPriceData(priceDf)
        
        uniqueRegions = outageDf['region'].unique()
        
        for region in uniqueRegions:
            trainAndForecast(outageDf, priceDf, region)