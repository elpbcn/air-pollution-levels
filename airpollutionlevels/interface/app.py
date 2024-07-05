import streamlit as st
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import base64

BASE_URL = 'http://localhost:8000'  # Update with your FastAPI container IP if necessary
#BASE_URL = 'https://airpollutionlevels-image-qgw4wjcfua-ew.a.run.app'

def fetch_forecast_pm25(city_name, country_name, future_periods):
    try:
        url = f"{BASE_URL}/forecast_pm25"
        params = {
            'city_name': city_name,
            'country_name': country_name,
            'future_periods': future_periods
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('plot_image'), data.get('summary_text')
        else:
            st.error(f"Failed to fetch forecast. Status code: {response.status_code}")
            return None, None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

def fetch_gif_local(file_path):
    try:
        with open(file_path, "rb") as f:
            gif_content = f.read()
        return gif_content
    except Exception as e:
        st.error(f"An error occurred while fetching local GIF: {str(e)}")
        return None

def fetch_forecast_pm25_data(city_name, country_name, future_periods):
    url = f"{BASE_URL}/forecast_pm25_data"
    params = {
            'city_name': city_name,
            'country_name': country_name,
            'future_periods': future_periods
        }
    response = requests.get(url, params=params)
    data = response.json()
    return data


def main():
    st.title('Air Pollution Forecast')

    # Sidebar menu
    st.sidebar.title('Menu')
    menu_option = st.sidebar.radio('Select Option', ['Forecast PM2.5 Levels', 'Europe Forecasts'])

    if menu_option == 'Forecast PM2.5 Levels':
        st.subheader('Forecast PM2.5 Levels')



        # Input fields
        city_name = st.text_input('Enter City Name', 'City')
        country_name = st.text_input('Enter Country Name', 'Country')
        future_periods = st.number_input('Enter Future Periods (months)', min_value=1, max_value=120, value=6)

        if st.button('Show Forecast'):
            data = fetch_forecast_pm25_data(city_name, country_name, future_periods)

            city_data = pd.json_normalize(data['city_data'])
            forecast = pd.json_normalize(data['forecast'])
            trend = pd.json_normalize(data['trend'])

            yhat_lower_mean = data['yhat_lower_mean']
            yhat_upper_mean = data['yhat_upper_mean']
            yhat_mean = data['yhat_mean']

            # Plot the forecast and actual values
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

            # Plot 1: Actual vs. Forecasted PM2.5 Levels
            ax1.plot(city_data['ds'], city_data['y'], label='Actual', color='blue')
            ax1.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--', color='red')
            ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], label='Uncertainty Range', color='lightgreen', alpha=0.3)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('PM2.5 (µg/m³)')
            ax1.set_title(f'Actual vs. Forecasted PM2.5 Levels for {city_name}, {country_name}')
            ax1.legend()
            ax1.grid(True)

            # Plot 2: Trend of PM2.5 Levels
            ax2.plot(trend['ds'], trend['trend'], label='Trend', color='lightblue')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Trend Value')
            ax2.set_title(f'Trend of PM2.5 Levels for {city_name}, {country_name}')
            ax2.legend()
            ax2.grid(True)

            figures = st.pyplot(fig)

            # Generate a summary text
            summary_text = (f"For the next {future_periods} months, the forecasted PM2.5 levels range from "
                            f"{yhat_lower_mean:.2f} to {yhat_upper_mean:.2f} µg/m³ with an average prediction of {yhat_mean:.2f} µg/m³.")

            st.markdown(summary_text)




    # if menu_option == 'Forecast PM2.5 Levels':
    #     st.subheader('Forecast PM2.5 Levels')

    #     # Input fields
    #     city_name = st.text_input('Enter City Name', 'City')
    #     country_name = st.text_input('Enter Country Name', 'Country')
    #     future_periods = st.number_input('Enter Future Periods (months)', min_value=1, max_value=120, value=6)

    #     if st.button('Show Forecast'):
    #         plot_image, summary_text = fetch_forecast_pm25(city_name, country_name, future_periods)
    #         if plot_image and summary_text:
    #             # Display text description
    #             st.markdown(summary_text)

    #             # Display plot image with increased size
    #             image_bytes = plot_image.encode('utf-8')
    #             image = Image.open(BytesIO(base64.b64decode(image_bytes)))
    #             st.image(image, caption=f'Forecast PM2.5 Levels for {city_name}, {country_name}',width=1000, use_column_width=True)

    #         else:
    #             st.error("Failed to fetch or display forecast.")

    elif menu_option == 'Europe Forecasts':
        st.subheader('Europe Forecasts')

        # Specify the path to your local GIF file
        #local_gif_path = resolve_path('airpollutionlevels/raw_data/animation.gif')

        # gif_content_local = fetch_gif_local(local_gif_path)
        # if gif_content_local:
        #     try:
        #         st.markdown(
        #             f'<img src="data:image/gif;base64,{base64.b64encode(gif_content_local).decode("utf-8")}" alt="Local GIF">',
        #             unsafe_allow_html=True
        #         )
        #     except Exception as e:
        #         st.error(f"Error displaying local GIF: {str(e)}")
        # else:
        #     st.error("Failed to fetch or display local GIF.")

        url = f"{BASE_URL}/display_gif"

        response = requests.get(url)
        data = response.json()
        st.markdown(
            f'<img src="data:image/gif;base64,{base64.b64encode(response).decode("utf-8")}" alt="Deployed GIF">',
           unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()
