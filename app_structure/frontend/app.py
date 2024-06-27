import streamlit as st
import requests

# Define the FastAPI backend endpoint URL
API_URL = 'http://localhost:8000'

st.markdown("""## Breathe Easy: Discover Your Air Quality with AirPulse
All for free!""")

with st.form("city_year_form"):
    city = st.text_input("City Name")
    year = st.number_input("Year", min_value=2020, max_value=2040, step=1)
    submitted = st.form_submit_button("Submit")

if submitted:
    if city and year:
        response = requests.post(f"{API_URL}/get_index/", json={"city": city, "year": year})
        if response.status_code == 200:
            data = response.json()
            index = data['index']
            st.write(f"The Air Quality Index for {city} in the year {year} is {index}.")
        else:
                st.write("Error: Could not get the index.")
    else:
        st.write("Please enter both city and year!")
