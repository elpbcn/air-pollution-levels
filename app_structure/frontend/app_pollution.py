import streamlit as st
import requests

# Define the new API endpoint URL
API_URL = 'https://airpollutionlevels-image-qgw4wjcfua-ew.a.run.app'

st.title("AirPulse - Predicted PM2.5 Concentration Finder")

st.markdown("""## Predicted PM2.5 Concentration for a City and Year
Based on Machine Learning Predictions""")

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
            st.write(f"The predicted PM2.5 concentration for {city} in the year {year} is {predicted_pm25}.")
        else:
            st.write("Error: Could not fetch the predicted concentration.")
    else:
        st.write("Please enter both city and year!")
