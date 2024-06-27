import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from airpollutionlevels.config import resolve_path
import joblib


def load_data():
    '''
    A function for loading csv data into dataframe df.
    '''

    #Location of csv file
    csv_file = resolve_path('airpollutionlevels/raw_data/air_pollution_data_upd.csv')

    #Loading csv file into df dataframe
    df = pd.read_csv(csv_file)

    #adding unique_id column
    df['unique_id'] = range(len(df))
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
    Classifies the concentrations of NO2, PM10, and PM2.5 into categories based on the European Air Quality Index (AQI) classification.
    Sets the target class as the maximum of the three classified pollutant concentrations.
    '''
    # Define classification limits
    no2_limits = [0, 40, 90, 120, 230, 340, 1000]
    pm25_limits = [0, 10, 20, 25, 50, 75, 800]
    pm10_limits = [0, 20, 40, 50, 100, 150, 1200]

    # Classify PM10 concentrations
    df['pm10_class'] = pd.cut(df['pm10_concentration'], bins=pm10_limits, labels=[1, 2, 3, 4, 5, 6])

    # Classify PM2.5 concentrations
    df['pm25_class'] = pd.cut(df['pm25_concentration'], bins=pm25_limits, labels=[1, 2, 3, 4, 5, 6])

    # Classify NO2 concentrations
    df['no2_class'] = pd.cut(df['no2_concentration'], bins=no2_limits, labels=[1, 2, 3, 4, 5, 6])

    # Determine the target class as the maximum of the three pollutant classes
    df['target_class'] = df[['pm10_class', 'pm25_class', 'no2_class']].apply(lambda row: row.max(), axis=1)

    # Drop the intermediate class columns
    df = df.drop(columns=['pm10_class', 'no2_class', 'pm25_class'])
    # Saving in a csv file as we need that to fetch information for Predictions
    csv_file = resolve_path('airpollutionlevels/raw_data/data_lib.csv')
    df.to_csv(csv_file, index=False)
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

def encode_scale_data(df):
    """
    Encode and scale the data for prediction.
    Add a temporary unique_id for tracking purposes.

    Parameters:
    df (DataFrame): Input DataFrame containing raw data.

    Returns:
    DataFrame: Transformed DataFrame with encoded and scaled features.
    """
    df = df.copy()
    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['country_name', 'year', 'population', 'latitude', 'longitude'])

    # Convert 'year' to integer
    df['year'] = df['year'].astype(int)

    # Columns to drop if they exist in the DataFrame
    columns_to_drop = ['pm10_concentration', 'pm25_concentration', 'no2_concentration', 'type_of_stations',
                       'simplified_station_type', 'encoded_station_type', 'encoded_station_type_imputed']
    df = df.drop(columns=columns_to_drop, axis=1)

    # Drop 'city' column if it exists
    if 'city' in df.columns:
        df = df.drop('city', axis=1)

    # Reset index to ensure it's sequential and clean
    df = df.reset_index(drop=True)

    # Define the columns for encoding and scaling
    categorical_cols = ['who_region', 'country_name', 'final_station_type']
    numeric_cols = ['population', 'latitude', 'longitude']

    # Instantiate encoders and scalers
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()

    # Pipeline for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', onehot_encoder, categorical_cols),
            ('scaler', scaler, numeric_cols)
        ],
        remainder='passthrough'  # Keep the 'unique_id' column unchanged
    )

    # Apply transformations (excluding 'pm25_concentration' if it exists)
    transformed_data = preprocessor.fit_transform(df.drop(columns=['target_class'], errors='ignore'))

    # Get the feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_cols)

    # Construct the final DataFrame columns
    final_columns = list(ohe_feature_names) + numeric_cols + ['year', 'unique_id']


    # Create the final DataFrame
    df_transformed = pd.DataFrame(transformed_data, columns=final_columns)
    df_transformed['target_class'] = df['target_class'].values

    # Save the transformed data to a CSV file
    csv_file = resolve_path('airpollutionlevels/raw_data/air_pollution_data_encoded_class.csv')
    df_transformed.to_csv(csv_file, index=False)

    return df_transformed


def impute_stations_rf(df):
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
    features = ['population', 'pm25_concentration', 'encoded_station_type'] #features to be learned by imputer

    # Perform KNN imputation
    imputer = KNNImputer(n_neighbors=5) #instantiate imputer
    df_imputed = imputer.fit_transform(df[features]) #returns array with learned features

    # Assign imputed values back to DataFrame
    df['encoded_station_type_imputed'] = df_imputed[:, -1]  # Assuming encoded_station_type is the last column after imputation

    # Revert encoded_station_type back to original categorical values
    reverse_mapping = {v: k for k, v in type_mapping.items() if pd.notna(v)}  # Reverse mapping excluding NaNs. source >> https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping

    df['final_station_type'] = df['encoded_station_type_imputed'].round().astype(int).map(reverse_mapping).fillna(np.nan)

    csv_file = resolve_path('airpollutionlevels/raw_data/data_lib_rf.csv')
    df.to_csv(csv_file, index=False) #save the transformed data to a csv file to be used in the model for city prediction
    return df

def encode_scale_data_rf(df):
    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['pm25_concentration', 'country_name', 'year', 'population', 'latitude', 'longitude'])

    # Convert 'year' to integer
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame slice
    df['year'] = df['year'].astype(int)

    # Columns to drop if they exist in the DataFrame
    columns_to_drop = ['pm10_concentration', 'no2_concentration', 'type_of_stations',
                       'simplified_station_type', 'encoded_station_type', 'encoded_station_type_imputed']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

    # Drop 'city' column if it exists
    if 'city' in df.columns:
        df = df.drop(columns='city', axis=1)

    # Reset index to ensure it's sequential and clean
    df = df.reset_index(drop=True)


    # Define the columns for encoding and scaling
    categorical_cols = ['who_region', 'country_name', 'final_station_type']
    numeric_cols = ['population', 'latitude', 'longitude']

    # Instantiate encoders and scalers
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()
    pm25_scaler = StandardScaler()

    # Fit the pm25_concentration scaler separately and transform
    df['pm25_concentration'] = pm25_scaler.fit_transform(df[['pm25_concentration']])

    # Save the pm25_scaler for future inverse transformation
    pm25_scaler_filename = resolve_path('airpollutionlevels/models/pm25_scaler.pkl')
    joblib.dump(pm25_scaler, pm25_scaler_filename)

    # Pipeline for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', onehot_encoder, categorical_cols),
            ('scaler', scaler, numeric_cols)
        ],
        remainder='passthrough'  # Keep the year and unique_id unchanged
    )

    # Apply transformations (excluding the already scaled 'pm25_concentration')
    transformed_data = preprocessor.fit_transform(df.drop(columns=['pm25_concentration']))

    # Get the feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_cols)

    # Construct the final DataFrame columns
    final_columns = list(ohe_feature_names) + numeric_cols + ['year', 'unique_id']

    # Create the final DataFrame without the pm25_concentration column
    df_transformed = pd.DataFrame(transformed_data, columns=final_columns)

    # Add the scaled pm25_concentration back to the DataFrame
    df_transformed['pm25_concentration'] = df['pm25_concentration'].values

    # Save the transformed data to a CSV file
    csv_file = resolve_path('airpollutionlevels/raw_data/air_pollution_data_encoded_rf.csv')
    df_transformed.to_csv(csv_file, index=False)

    return df_transformed
