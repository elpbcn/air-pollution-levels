import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
from airpollutionlevels.config import resolve_path

BASE_URL = 'http://localhost:8000'  # Update with your FastAPI container IP if necessary

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
            plot_image, summary_text = fetch_forecast_pm25(city_name, country_name, future_periods)
            if plot_image and summary_text:
                # Display text description
                st.markdown(summary_text)

                # Display plot image with increased size
                image_bytes = plot_image.encode('utf-8')
                image = Image.open(BytesIO(base64.b64decode(image_bytes)))
                st.image(image, caption=f'Forecast PM2.5 Levels for {city_name}, {country_name}',width=1000, use_column_width=True)

            else:
                st.error("Failed to fetch or display forecast.")

    elif menu_option == 'Europe Forecasts':
        st.subheader('Europe Forecasts')
        st.markdown("![Alt Text](https://airpollutionlevels-image-qgw4wjcfua-ew.a.run.app/display_gif)")
        # # Specify the path to your local GIF file
        # local_gif_path = resolve_path('airpollutionlevels/raw_data/animation.gif')
        # cloud_gif_path = 'https://airpollutionlevels-image-qgw4wjcfua-ew.a.run.app/display_gif'
        # gif_content_local = fetch_gif_local(cloud_gif_path)
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

if __name__ == '__main__':
    main()
