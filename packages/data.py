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

    return df
