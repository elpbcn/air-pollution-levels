import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def load_data():
    '''
    A function for loading csv data into dataframe df.
    '''

    #Location of csv file
    csv_file = '../air-pollution-levels/raw_data/air_pollution_data.csv'

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

def simplify_stations(station_type):
    '''
    Simplifies the station type string by removing duplicates and sorting.

    Args:
    - station_type (str): A string containing station types separated by ', ' e.g. Urban, urban, urban.

    Returns:
    - str: Simplified station types joined into a single string e.g "Urban, urban, urban" returns "Urban"

    If station_type is NaN (missing), returns 'unknown'.'''

    if pd.isna(station_type):
        return "unknown"
    unique_types = sorted(set(station_type.split(', ')))
    return ', '.join(unique_types)

def simplified_station_type(df):
    '''
    Adds a new column 'simplified_station_type' to the DataFrame 'df' based on simplifying 'type_of_stations'.

    Args:
    - df (pandas.DataFrame): The DataFrame containing the column 'type_of_stations' to be simplified.

    Returns:
    - pandas.DataFrame: The input DataFrame 'df' with an additional column 'simplified_station_type'.

    This function applies the 'simplify_stations' function to each value in the 'type_of_stations' column
    and stores the simplified result in a new column 'simplified_station_type'
    '''

    df['type_of_stations'] = df['type_of_stations'].astype('string') #converts type_of_stations column into a string in order to apply simplify_stations function
    df['simplified_station_type'] = df['type_of_stations'].apply(simplify_stations)
    return df

def impute_stations(df):
    '''
    Imputes the values of missing type_of_stations based on similar pollution metrics of know types of stations using KNN imputer'''

    #first simplify station names using simplified_station_type function
    simplified_station_type(df)

    # Manually map known types of stations to numerical labels from stations3 df
    type_mapping = {
        'Unknown': np.nan, #will need this to be nan for imputer to work
        'Urban': 1,
        'Rural': 2,
        'Suburban': 3,
        'Suburban, Urban': 4,
        'Rural, Urban': 5,
        'Rural, Suburban, Urban': 6,
        'Rural, Suburban': 7,
        'Background': 8,
        'Residential And Commercial Area': 9,
        'Traffic': 10,
        'Residential And Commercial Area, Urban Traffic': 11,
        'Background, Traffic': 12,
        'Industrial': 13,
        'Residential And Commercial Area, Urban Traffic': 14,
        'Industrial, Urban': 15,
        'Industrial, Rural, Urban': 16,
        'Residential': 17,
        'Fond Urbain, Traffic': 18,
        'Residential - industrial': 19
    }

    df['encoded_station_type'] = df['simplified_station_type'].map(type_mapping) # encode simpified_station_type column to feed into KNN imputer

    # Select features for imputation
    features = ['pm10_concentration', 'pm25_concentration', 'no2_concentration', 'encoded_station_type'] #features to be learned by imputer

    # Perform KNN imputation
    imputer = KNNImputer(n_neighbors=5) #instantiate imputer
    df_imputed = imputer.fit_transform(df[features]) #returns array with learned features

    # Assign imputed values back to DataFrame
    df['encoded_station_type_imputed'] = df_imputed[:, -1]  # Assuming encoded_station_type is the last column after imputation

    # Revert encoded_station_type back to original categorical values
    reverse_mapping = {v: k for k, v in type_mapping.items() if pd.notna(v)}  # Reverse mapping excluding NaNs. source >> https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping

    df['final_station_type'] = df['encoded_station_type_imputed'].round().astype(int).map(reverse_mapping).fillna(np.nan)

    return df
