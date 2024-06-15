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
