import pandas as pd
import numpy as np
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "https://power-predict.vercel.app"}}
)

def loadData(filepath):
    try:
        data = pd.read_csv(filepath)
    except:
        return None

    try:
        data['interval_start_local'] = pd.to_datetime(
            data['interval_start_local'], utc=True
        ).dt.tz_convert(None)
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
        data['lmp_price'] = np.random.normal(35, 10, len(data)) + (
            data['month'].isin([1, 2, 7, 8]) * 15
        )
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
    xTrain, xTest, yTrain, yTest = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    totalRows = len(X)

    models = {
        'Linear Regression': LinearRegression(),
        'KNN': KNeighborsRegressor(n_neighbors=min(int(np.sqrt(totalRows)), 20)),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR()
    }

    bestModel = None
    bestScore = float('inf')

    for model in models.values():
        try:
            model.fit(xTrain, yTrain)
            preds = model.predict(xTest)
            mae = mean_absolute_error(yTest, preds)
            if mae < bestScore:
                bestScore = mae
                bestModel = model
        except:
            continue

    return bestModel

def get_forecast_logic(region, include_prices):
    outageDf = loadData("./Outage_Dataset.csv")
    if outageDf is None:
        return "Error: Outage_Dataset.csv not found"

    outageData = outageDf[outageDf['region'] == region].copy()
    if outageData.empty:
        return "Region Not Found"

    history_slice = outageData.tail(10).copy()
    history_results = [
        {
            "date": row['interval_start_local'].strftime('%Y-%m-%d'),
            "outage_mw": round(row['total_outage_mw'], 2),
            "type": "History"
        }
        for _, row in history_slice.iterrows()
    ]

    targetCol = 'total_outage_mw'
    outageData['previous'] = outageData[targetCol].shift(1)
    outageData = outageData.dropna()

    features = ['dayOfWeek', 'month', 'isWeekend', 'previous']
    XOutage = outageData[features]
    YOutage = outageData[targetCol]

    bestOutageModel = findBestModel(XOutage, YOutage)

    lastDate = outageData.iloc[-1]['interval_start_local']
    currOutage = outageData.iloc[-1][targetCol]

    forecast_results = []

    priceData = None
    priceCols = []

    if include_prices:
        priceDf = loadData("./PowerGridPriceData.csv")
        if priceDf is not None:
            priceDf = prepPriceData(priceDf)
            priceData = priceDf[priceDf['region'] == region].copy()
            if not priceData.empty:
                priceCols = [
                    'solar_price', 'wind_price', 'natural_gas_price',
                    'coal_price', 'nuclear_price', 'oil_price', 'hydro_price'
                ]

    for i in range(1, 6):
        nextDate = lastDate + timedelta(days=i)

        inputRow = pd.DataFrame([{
            'dayOfWeek': nextDate.dayofweek,
            'month': nextDate.month,
            'isWeekend': int(nextDate.dayofweek >= 5),
            'previous': currOutage
        }])

        pred_outage = bestOutageModel.predict(inputRow)[0]
        currOutage = pred_outage

        day_result = {
            "date": nextDate.strftime('%Y-%m-%d'),
            "outage_mw": round(pred_outage, 2),
            "type": "Forecast"
        }

        if include_prices and priceData is not None:
            day_result["predicted_prices"] = {}

            priceInput = pd.DataFrame([{
                'total_outage_mw': pred_outage,
                'dayOfWeek': nextDate.dayofweek,
                'month': nextDate.month,
                'isWeekend': int(nextDate.dayofweek >= 5)
            }])

            priceFeatures = ['total_outage_mw', 'dayOfWeek', 'month', 'isWeekend']
            XPriceTrain = priceData[priceFeatures]

            for col in priceCols:
                model = findBestModel(XPriceTrain, priceData[col])
                day_result["predicted_prices"][col] = round(
                    model.predict(priceInput)[0], 2
                )

        forecast_results.append(day_result)

    return {
        "history": history_results,
        "forecast": forecast_results
    }

@app.route('/')
def home():
    return jsonify({
        "message": "Backend is running!",
        "usage": "GET /predict?region=Western&include_prices=true"
    })

@app.route('/predict', methods=['GET'])
def predict():
    region = request.args.get('region')
    include_prices = request.args.get('include_prices', 'false').lower() == 'true'

    if not region:
        return jsonify({"error": "Please provide a 'region' parameter"}), 400

    try:
        results = get_forecast_logic(region, include_prices)

        if results is None:
            return jsonify({"error": "Server Error"}), 500
        if isinstance(results, str) and results.startswith("Error"):
            return jsonify({"error": results}), 500
        if results == "Region Not Found":
            return jsonify({"error": f"Region '{region}' not found"}), 404

        return jsonify({
            "region": region,
            "data": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
