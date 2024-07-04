from fastapi import FastAPI, HTTPException
from airpollutionlevels.ml_logic.model import forecast_pm25
from airpollutionlevels.ml_logic.map_graphics import display_gif

app = FastAPI()

@app.get('/')
def root():
    return {'Welcome to the Air Pollution Levels API!': "Use the /forecast_pm25 endpoint to make predictions and /display_gif to show GIF."}

# Endpoint for forecasting PM2.5 levels
@app.post("/forecast_pm25")
def forecast_pm25_endpoint(city_country_periods: dict):
    """
    Endpoint to forecast PM2.5 levels for a city.

    Parameters:
        city_country_periods (dict): Dictionary containing 'city', 'country', and 'future_periods'.
    """
    city_name = city_country_periods['city']
    country_name = city_country_periods['country']
    future_periods = int(city_country_periods['future_periods'])

    try:
        forecast = forecast_pm25(city_name, country_name, future_periods)
        return {"forecast": forecast.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Endpoint for displaying a GIF
@app.get("/display_gif")
def display_gif_endpoint():
    """
    Endpoint to display a GIF file.

    """
    try:
        display_gif()  # Assuming display_gif function already handles displaying the GIF
        return {"message": "GIF displayed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
