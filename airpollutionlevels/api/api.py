from fastapi import FastAPI

from airpollutionlevels.ml_logic.model import predict, predict_rf

app = FastAPI()


# Endpoint for predicting pollution level using Decision Tree model
@app.get("/predict")
def predict_pollution_level(city:str, year:int):
        prediction = predict(city, year)
        return {"prediction": int(prediction)}

# Endpoint for predicting PM2.5 concentration using Random Forest model
@app.get("/predict_rf")
def predict_pm25_concentration(city:str, year:int):
        prediction = predict_rf(city, year)
        return {"predicted_pm25_concentration": float(prediction)}

@app.get('/')
def root():
    return {'Welcome to the Air Pollution Levels API!': "Use the /predict or /predict_rf endpoint to make predictions."}
