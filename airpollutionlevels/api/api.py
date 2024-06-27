from fastapi import FastAPI

from airpollutionlevels.ml_logic.model import predict, predict_rf

app = FastAPI()

@app.get('/')
def root():
    return {'Welcome to the Air Pollution Levels API!': "Use the /predict or /predict_rf endpoint to make predictions."}

# Endpoint for predicting pollution level using Decision Tree model
@app.post("/predict")
def predict_pollution_level(city_year: dict):
        city = city_year['city']
        year = int(city_year['year'])
        prediction = predict(city, year)
        return {"prediction": prediction}


# Endpoint for predicting PM2.5 concentration using Random Forest model
@app.post("/predict_rf")
def predict_pm25_concentration(city_year: dict):
        city = city_year['city']
        year = int(city_year['year'])
        prediction = predict_rf(city, year)
        return {"predicted_pm25_concentration": prediction}
