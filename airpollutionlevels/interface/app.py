import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import folium
from folium.plugins import HeatMapWithTime
from streamlit_folium import folium_static
from scipy.spatial import KDTree
import pandas as pd


BASE_URL = 'https://airpollutionlevels-image-qgw4wjcfua-ew.a.run.app'  # Update with your FastAPI container IP if necessary


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

# 2. DEFINE FUNCTIONS TO PROCESS CITY ENTRERED BY USER INTO COORDINATES TO LOOKUP IN OUR DF IN STEP 1 ABOVE

# 2(a) get user city coordinates from opensoft api
def get_coordinates_opendatasoft1(city_name, country_name):
    base_url = "https://public.opendatasoft.com/api/records/1.0/search/"
    params = {
        'dataset': 'geonames-all-cities-with-a-population-1000',
        'q': f'{city_name}',
        'refine.country': country_name,
        'rows': 1,
        'sort': 'population',
        'format': 'json'
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['nhits'] > 0:
            record = data['records'][0]['fields']
            lat = float(record['coordinates'][0])
            lon = float(record['coordinates'][1])
            return round(lat, 2), round(lon, 2)
        else:
            print(f"No results found for {city_name}, {country_name}")
            return None, None
    else:
        print(f"Failed to get coordinates for {city_name}, {country_name}. Response code: {response.status_code}")
        return None, None

# 2(b) find nearest coordinates in df
def find_nearest_coordinates1(target_lat, target_lon, dataset):
    tree = KDTree(dataset[['latitude', 'longitude']])
    dist, idx = tree.query([(target_lat, target_lon)])
    nearest_lat = dataset.iloc[idx[0]]['latitude']
    nearest_lon = dataset.iloc[idx[0]]['longitude']
    return nearest_lat, nearest_lon

# 2(c) combined 2(a) and 2(b) above to get final result from df. result used to feed into maps
def user_city_latlon1(user_city, user_country, df):
    targetlat, targetlon = get_coordinates_opendatasoft1(user_city, user_country)
    nearest_lat, nearest_lon = find_nearest_coordinates1(targetlat, targetlon, df)
    return nearest_lat, nearest_lon

def main():
    st.title('Air Pollution Forecast')

    # Sidebar menu
    st.sidebar.title('Menu')
    menu_option = st.sidebar.radio('Select Option', ['Forecast PM2.5 Levels', 'Europe Forecasts', 'PM2.5 Heatmap'])

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

    elif menu_option == 'PM2.5 Heatmap':
        st.subheader('PM2.5 Heatmap')

        df = pd.read_csv("https://raw.githubusercontent.com/elpbcn/air-pollution-levels/master/airpollutionlevels/interface/predicts.csv") #replace with file path to raw data folder where predicts is saved

        # drop irrelevant columns for heatmap
        df.drop(columns =['ds', 'yhat', 'yhat_lower','yhat_upper'], inplace=True)

        # convert month_year to date time
        df['month_year'] = pd.to_datetime(df['month_year'])

        # calculate a weight
        df['weight'] = df['weight']/6

        # Input fields
        city_name = st.text_input('Enter City Name', 'City')
        country_name = st.text_input('Enter Country Name', 'Country')

        if st.button('Show HeatMap'):

            # Get latitude and longitude for user's city
            user_city = city_name
            user_country = country_name
            userlat, userlon = user_city_latlon1(user_city, user_country, df)

            # Create a base map centered on the user's city
            m = folium.Map(location=[userlat, userlon], zoom_start=9.5)  # Adjust the zoom level as needed
            #folium.Marker([userlat, userlon], popup=city).add_to(m)

            # Filter heat_data for the user's city coordinates
            heat_map_df = df[(df['latitude'] == userlat) & (df['longitude'] == userlon)]

            # Create heat data in the format required by HeatMapWithTime
            heat_data = []
            time_index = sorted(heat_map_df['month_year'].unique())
            for time in time_index:
                data = heat_map_df[heat_map_df['month_year'] == time]
                heat_data.append(data[['latitude', 'longitude', 'weight']].values.tolist())


            # Add HeatMapWithTime layer for the filtered data
            HeatMapWithTime(
                data=heat_data,
                index=[time.strftime("%Y-%m-%d") for time in time_index],
                gradient={0.167: '#36ac56', 0.333: '#9bd445', 0.5: '#f1d208', 0.667: '#ffbb02', 0.833: '#ff8b00', 1.0: '#ed0e06'},
                radius=10,
                scale_radius=True,
            ).add_to(m)

            # Display the map in Streamlit
            folium_static(m)

            # Add legend below the map
            st.markdown("""
            <style>
            .legend {
                position: relative;
                bottom: 0;
                width: 100%;
                display: flex;
                justify-content: space-around;
                padding: 10px;
                background: white;
                border: 2px solid grey;
                border-radius: 5px;
            }
            .legend div {
                display: flex;
                align-items: center;
            }
            .legend div span {
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 5px;
            }
            </style>
            <div class="legend">
                <div><span style="background: #36ac56"></span>1</div>
                <div><span style="background: #9bd445"></span>2</div>
                <div><span style="background: #f1d208"></span>3</div>
                <div><span style="background: #ffbb02"></span>4</div>
                <div><span style="background: #ff8b00"></span>5</div>
                <div><span style="background: #ed0e06"></span>6</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
