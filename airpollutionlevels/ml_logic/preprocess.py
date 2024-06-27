import pandas as pd

def classify_concentrations(df):
    '''
    A function that classifies the concentrations of NO2, PM10, and PM2.5 into categories based on the European Air Quality Index (AQI) classification.
    '''
    # Define classification limits
    no2_limits = [0, 40, 90, 120, 230, 340, 1000]
    pm10_limits = [0, 10, 20, 25, 50, 75, 800]
    pm25_limits = [0, 20, 40, 50, 100, 150, 1200]

    # Classify PM10 concentrations
    df['pm10_class'] = pd.cut(df['pm10_concentration'], bins=pm10_limits, labels=[1, 2, 3, 4, 5, 6])

    # Classify PM2.5 concentrations
    df['pm25_class'] = pd.cut(df['pm25_concentration'], bins=pm25_limits, labels=[1, 2, 3, 4, 5, 6])

    # Classify NO2 concentrations
    df['no2_class'] = pd.cut(df['no2_concentration'], bins=no2_limits, labels=[1, 2, 3, 4, 5, 6])

    # Drop the original concentration columns
    df = df.drop(columns=['no2_concentration', 'pm10_concentration', 'pm25_concentration'])

    # Determine the target class as the maximum of the three pollutant classes
    df['target_class'] = df[['no2_class', 'pm10_class', 'pm25_class']].max(axis=1)
    return df
