import streamlit as st
import requests

# Define the new API endpoint URL
API_URL = 'https://airpollutionlevels-image-qgw4wjcfua-ew.a.run.app'

st.title("AirPulse - your pm2.5 concentration predictor")

st.markdown("""## Breathe Easy: Discover your air quality with AirPulse.
Just enter a City and Year.
Let us do the heavy lifting!""")

with st.form("city_year_form"):
    city = st.text_input("City Name")
    year = st.number_input("Year", min_value=2020, max_value=2040, step=1)
    submitted = st.form_submit_button("Submit")

if submitted:
    if city and year:
        # Construct the API query parameters
        params = {
            "city": city,
            "year": year
        }

        # Send GET request to the API endpoint
        response = requests.get(f"{API_URL}/predict_rf", params=params)

        if response.status_code == 200:
            data = response.json()
            predicted_pm25 = data.get('predicted_pm25_concentration')
            st.write(f"The predicted PM2.5 concentration for {city} in the year {year} is {predicted_pm25} µg/m³.")
        else:
            st.write("Error: Could not fetch the predicted concentration.")
    else:
        st.write("Please enter both city and year!")
