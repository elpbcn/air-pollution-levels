import streamlit as st
import pandas as pd
import pickle
import folium
import os
from folium.plugins import HeatMapWithTime
from streamlit_folium import folium_static
from branca.element import Template, MacroElement

# Streamlit app
st.title("Air Quality index")


# loading from file
with open("/home/elpbcn/code/elpbcn/air-pollution-levels/airpollutionlevels/interface/heat_data.txt", 'rb') as f:
    heat_data = pickle.load(f)

with open("/home/elpbcn/code/elpbcn/air-pollution-levels/airpollutionlevels/interface/time_index.txt", 'rb') as f:
    time_index = pickle.load(f)


# Create a base map
m = folium.Map(location=[54.5260, 15.2551], zoom_start=5)

# Add HeatMapWithTime layer
HeatMapWithTime(
    data=heat_data,
    index=time_index,
    gradient={0.167: '#36ac56', 0.333: '#9bd445', 0.5: '#f1d208', 0.667: '#ffbb02', 0.833: '#ff8b00', 1.0: '#ed0e06'},  # Adjust gradient colors as needed: gradient={0.167: '#36ac56', 0.333: '#9bd445', 0.5: '#f1d208', 0.667: '#ffbb02', 0.833: '#ff8b00', 1.0: '#ed0e06'}
    radius=0.75,
    # min_opacity=0.1,
    # max_opacity=0.3,
    scale_radius=True,
    # auto_play=False,
    # max_speed=10,
    # speed_step=0.1,
    # position='bottomleft'
).add_to(m)

# Display map in Streamlit
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
