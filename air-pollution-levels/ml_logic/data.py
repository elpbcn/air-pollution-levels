import pandas as pd

def load_data():
    '''
    A function for loading csv data into dataframe df.
    '''

    #Location of csv file
    csv_file = '../raw_data/air_pollution_data.csv'

    #Loading csv file into df dataframe
    df = pd.read_csv(csv_file)

    return df

def clean_data(df):
    '''
    A function to clean raw data:
    - Dropping unuseful columns
    - Dropping rows with year = NA
    - Dropping rows where pm10_concentration AND pm25_concentration AND no2_concentration are NA
    '''

    #Dropping columns: web_link, reference, iso3, who_ms, population_source, version, pm10_tempcov, pm25_tempcov, no2_tempcov
    df.drop(columns=['web_link',
                     'reference',
                     'iso3',
                     'who_ms',
                     'population_source',
                     'version',
                     'pm10_tempcov',
                     'pm25_tempcov',
                     'no2_tempcov'],
            inplace=True)

    #Dropping rows where year is NA (3 rows for India)
    df.dropna(subset=['year'], inplace=True)

    #Dropping rows where pm10_concentration AND pm25_concentration AND no2_concentration are NA
    df.dropna(how='all', subset=['pm10_concentration', 'pm25_concentration', 'no2_concentration'], inplace=True)

    return df


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

    # Determine the target class as the maximum of the three pollutant classes
    df['target_class'] = df[['no2_class', 'pm10_class', 'pm25_class']].max(axis=1)

    # Drop the class concentration columns
    df = df.drop(columns=['pm10_class', 'no2_class', 'pm25_class'])
    return df
