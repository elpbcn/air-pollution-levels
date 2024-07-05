import streamlit as st
import pandas as pd
import folium
import os
from folium.plugins import HeatMapWithTime
from streamlit_folium import folium_static
from branca.element import Template, MacroElement
import pickle

chunk_size = 20_000  # Adjust this value based on your system's capacity
chunks = []
for chunk in pd.read_csv("/home/elpbcn/code/elpbcn/air-pollution-levels/airpollutionlevels/raw_data/predicts.csv", chunksize=chunk_size):
    chunks.append(chunk)

df = pd.concat(chunks, axis=0)

#df = pd.read_csv("/home/elpbcn/code/elpbcn/air-pollution-levels/airpollutionlevels/raw_data/predicts.csv")
df.drop(columns =['month_year', 'yhat', 'yhat_lower','yhat_upper'], inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

df['weight'] = df['weight']/6

# Prepare data for HeatMapWithTime
heat_data = []
time_index = []

# Generate heatmap data for each unique date
for date in df['ds'].dt.date.unique():
    day_data = df[df['ds'].dt.date == date]
    heat_data.append([[row['latitude'], row['longitude'], row['weight']] for _, row in day_data.iterrows()])
    time_index.append(date.strftime('%Y-%m-%d'))

# storing to file
with open("heat_data.txt", 'wb') as f:
    pickle.dump((heat_data), f)

with open("time_index.txt", 'wb') as f:
    pickle.dump((time_index), f)
