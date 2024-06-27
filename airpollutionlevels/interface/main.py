import pandas as pd
import numpy as np

from airpollutionlevels.ml_logic.data import *

def preprocess_classification():
    '''
    A function to preprocess the data for classification.
    '''
    data = load_data()
    data = clean_data(data)
    data = impute_stations(data)
    data = classify_concentrations(data)
    data = encode_scale_data(data)
    print("Classification preprocessing completed.")

def preprocess_regression():
    '''
    A function to preprocess the data for regression.
    '''
    data = load_data()
    data = clean_data(data)
    data = impute_stations_rf(data)
    data = encode_scale_data_rf(data)
    print("Regression preprocessing completed.")

