from fastapi import FastAPI, HTTPException
from airpollutionlevels.ml_logic.model import *
from airpollutionlevels.ml_logic.map_graphics import display_gif
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import io

app = FastAPI()

@app.get('/')
def root():
    return {'Welcome to the Air Pollution Levels API!': "Use the /forecast_pm25 endpoint to make predictions and /display_gif to show GIF."}

# Endpoint for forecasting PM2.5 levels
@app.get("/forecast_pm25")
def get_forecast_pm25(city_name: str, country_name: str, future_periods: int):
    try:
        plot_image, summary_text = forecast_pm25(city_name, country_name, future_periods)
        if plot_image:
            return {
                "plot_image": plot_image,
                "summary_text": summary_text
            }
        else:
            raise HTTPException(status_code=404, detail="Plot image could not be generated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/forecast_pm25_data")
def get_forecast_pm25_data(city_name: str, country_name: str, future_periods: int):
        city_data, forecast, trend, yhat_lower_mean, yhat_upper_mean, yhat_mean = forecast_pm25_data (city_name, country_name, future_periods)
        city_data = city_data.to_dict(orient='records')
        forecast = forecast.to_dict(orient='records')
        trend = trend.to_dict(orient='records')

        data = {
            "city_data": city_data,
            "forecast": forecast,
            "trend": trend,
            "yhat_lower_mean": float(yhat_lower_mean),
            "yhat_upper_mean": float(yhat_upper_mean),
            "yhat_mean": float(yhat_mean)
            }

        return data






# Endpoint to display a GIF file
@app.get("/display_gif")
def display_gif_endpoint():
    """
    Endpoint to display a GIF file.
    """
    gif_file_path = os.path.join(os.path.dirname(__file__), 'animation.gif')
    try:
        # Open the GIF file using PIL
        with open(gif_file_path, "rb") as f:
            gif_bytes = f.read()

        # Check if the file is indeed a GIF
        gif = Image.open(BytesIO(gif_bytes))
        if gif.format == 'GIF':
            print(f"Displaying GIF: {gif_file_path}")
            return StreamingResponse(BytesIO(gif_bytes), media_type="image/gif")
        else:
            raise HTTPException(status_code=400, detail="The file is not a GIF format.")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File {gif_file_path} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

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
