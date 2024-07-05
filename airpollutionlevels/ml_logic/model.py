import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from airpollutionlevels.ml_logic.data import encode_scale_data, encode_scale_data_rf

from scipy.stats import randint

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score , mean_absolute_error, mean_squared_error, r2_score , root_mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree
import requests
from prophet import Prophet

from io import BytesIO
import base64

import joblib
import os
from airpollutionlevels.config import resolve_path

def train_and_save_model():
    """
    Train a Decision Tree model on the provided dataset and save it.

    Parameters:
    df (pd.DataFrame): Preprocessed dataset containing features and target_class.
    """
    # Data loading
    data_path = resolve_path('airpollutionlevels/raw_data/air_pollution_data_encoded_class.csv')
    df = pd.read_csv(data_path)
    # Specify the model file path
    model_dir = resolve_path('airpollutionlevels/models/')
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
    model_filename = os.path.join(model_dir, 'decision_tree_model.pkl')

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['target_class', 'unique_id'])
    y = df['target_class']

    # Initialize the Decision Tree model with best parameters
    dt = DecisionTreeClassifier(
    criterion='gini',
    max_depth=30,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
    )
    # Save the model for evaluation
    model_evaluate_filename = os.path.join(model_dir, 'decision_tree_model_evaluation.pkl')
    joblib.dump(dt, model_evaluate_filename)
    # Fit the model
    dt.fit(X, y)

    # Save the model
    joblib.dump(dt, model_filename)
    print(f"Model saved at {model_filename}")

def evaluate_model():
    """
    Load a saved Decision Tree model and evaluate its performance on the provided dataset.

    Parameters:
    df (pd.DataFrame): Preprocessed dataset containing features and target_class.

    """
    # Load data
    data_path = resolve_path('airpollutionlevels/raw_data/air_pollution_data_encoded_class.csv')
    df = pd.read_csv(data_path)
    # Load the model
    model_filename = resolve_path('airpollutionlevels/models/decision_tree_model_evaluation.pkl')
    dt = joblib.load(model_filename)

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['target_class' , 'unique_id'])
    y = df['target_class']
     # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt.fit(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

def predict(city, year):
    """
    Load a saved Decision Tree model and predict pollution level for a given city and year.
    If the year does not exist in the dataset, use the latest available data.

    Parameters:
    city (str): Name of the city.
    year (int): Year for the prediction.

    Returns:
    int: Predicted pollution level.
    """

    model_filename = resolve_path('airpollutionlevels/models/decision_tree_model.pkl')
    csv_path = resolve_path('airpollutionlevels/raw_data/data_lib.csv')
    data = pd.read_csv(csv_path)

    # Verify if the 'city' column exists in the DataFrame
    if 'city' not in data.columns:
        raise ValueError("'city' column not found in the dataset. Please ensure the dataset contains a 'city' column.")

    # Check if the city exists in the dataset
    if city not in data['city'].values:
        raise ValueError(f"No data found for '{city}' in the dataset.")

    # Find the latest row for the provided city in the original dataset
    city_data = data[(data['city'] == city) & (data['year'] == data[data['city'] == city]['year'].max())].copy()

    # Encode and scale the entire data
    try:
        encoded_data = encode_scale_data(data)
    except ValueError as e:
        print(f"Not enough data available for '{city}'.") # Handle insufficient data
        return None
    # Get the unique_id for the latest city data
    unique_id = city_data.iloc[0]['unique_id']

    # Locate the corresponding row in the encoded data using the unique identifier
    encoded_city_data = encoded_data[encoded_data['unique_id'] == unique_id].copy()

    # Replace the year in the encoded city data with the input year
    encoded_city_data.loc[:, 'year'] = year

    # Drop the unique_id column as it's no longer needed
    encoded_city_data = encoded_city_data.drop(columns=['unique_id'])

    # Prepare features for prediction (drop the 'target_class' column if it exists)
    X_predict = encoded_city_data.drop(columns=['target_class'], errors='ignore')

    # Load the model
    dt = joblib.load(model_filename)

    # Predict
    prediction = dt.predict(X_predict)[0]

    print(f"Predicted pollution level for {city} in {year}: {prediction}")
    return prediction

def train_and_save_model_rf():
    """
    Train a Random Forest Regressor on the provided dataset and save it.

    The model is trained on the preprocessed data and saved to a file for future use.
    """
    # Specify the model file path
    model_dir = resolve_path('airpollutionlevels/models/')
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
    model_filename = os.path.join(model_dir, 'random_forest_model.pkl')
    csv_path = resolve_path('airpollutionlevels/raw_data/air_pollution_data_encoded_rf.csv')
    df = pd.read_csv(csv_path)

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['pm25_concentration','unique_id'])
    y = df['pm25_concentration']

    # Define Random Forest Regressor parameters
    params = {
        'bootstrap': True,
        'ccp_alpha': 0.0,
        'criterion': 'squared_error',
        'max_depth': None,
        'max_features': 1.0,
        'max_leaf_nodes': None,
        'max_samples': None,
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 100,
        'n_jobs': -1,
        'oob_score': False,
        'random_state': 42,
        'verbose': 0,
        'warm_start': False
    }

    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(**params)

    # Fit the model
    rf.fit(X, y)

    # Save the model
    joblib.dump(rf, model_filename)
    print(f"Model saved at {model_filename}")

def evaluate_model_rf(test_size=0.2, random_state=123):
    """
    Train a Random Forest model, split the dataset into train and test sets, and evaluate the model's performance.

    Parameters:
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    """
    csv_path = resolve_path('airpollutionlevels/raw_data/air_pollution_data_encoded_rf.csv')
    df = pd.read_csv(csv_path)
    # Split the data into features (X) and target (y)
    X = df.drop(columns=['pm25_concentration'])
    y = df['pm25_concentration']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model_filename = resolve_path('airpollutionlevels/models/random_forest_model.pkl')

    # Load the model
    rf = joblib.load(model_filename)


    # Fit the model on the training data
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

def predict_rf(city, year):
    """
    Load a saved Random Forest model and predict pollution level (PM2.5) for a given city and year.
    If the city does not exist in the dataset, raise an error.

    Parameters:
    city (str): Name of the city.
    year (int): Year for the prediction.

    Returns:
    float: Predicted PM2.5 concentration.
    """
    pm25_scaler_filename = resolve_path('airpollutionlevels/models/pm25_scaler.pkl')
    model_filename = resolve_path('airpollutionlevels/models/random_forest_model.pkl')
    csv_path = resolve_path('airpollutionlevels/raw_data/data_lib_rf.csv')
    data = pd.read_csv(csv_path)

    # Verify if the 'city' column exists in the DataFrame
    if 'city' not in data.columns:
        raise ValueError("'city' column not found in the dataset. Please ensure the dataset contains a 'city' column.")

    # Check if the city exists in the dataset
    if city not in data['city'].values:
        raise ValueError(f"No data found for '{city}' in the dataset.")

    # Find the latest row for the provided city in the original dataset
    city_data = data[(data['city'] == city) & (data['year'] == data[data['city'] == city]['year'].max())].copy()


    # Encode and scale the entire data

    encoded_data = encode_scale_data_rf(data)


    # Get the unique_id for the latest city data
    unique_id = city_data.iloc[0]['unique_id']

    # Locate the corresponding row in the encoded data using the unique identifier
    encoded_city_data = encoded_data[encoded_data['unique_id'] == unique_id].copy()

    # Replace the year in the encoded city data with the input year
    encoded_city_data.loc[:, 'year'] = year

    # Drop the unique_id column as it's no longer needed
    encoded_city_data = encoded_city_data.drop(columns=['unique_id'])

    # Load the model
    rf_model = joblib.load(model_filename)

    # Predict pm25_concentration
    prediction = rf_model.predict(encoded_city_data.drop(columns=['pm25_concentration']))[0]

    # Load the pm25 scaler
    pm25_scaler = joblib.load(pm25_scaler_filename)

    # Inverse transform the predicted value
    predicted_pm25 = pm25_scaler.inverse_transform([[prediction]])[0][0]

    print(f"Predicted PM2.5 concentration for {city} in {year}: {predicted_pm25}")
    return predicted_pm25


def get_coordinates_opendatasoft(city_name, country_name):
    # Validate inputs
    if not city_name or not country_name:
        print("City name and country name are required.")
        return None, None

    # Further validation logic if needed (e.g., minimum length)
    if len(city_name) < 3 or len(country_name) < 2:
        print("City name and country name should have minimum lengths.")
        return None, None

    base_url = "https://public.opendatasoft.com/api/records/1.0/search/"
    params = {
        'dataset': 'geonames-all-cities-with-a-population-500',
        'q': f'{city_name}, {country_name}',
        'format': 'json'
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['nhits'] > 0:
            record = data['records'][0]['fields']
            lat = float(record['latitude'])
            lon = float(record['longitude'])
            return round(lat, 2), round(lon, 2)
        else:
            print(f"No results found for {city_name}, {country_name}")
            return None, None
    else:
        print(f"Failed to get coordinates for {city_name}, {country_name}. Response code: {response.status_code}")
        return None, None


def find_nearest_coordinates(target_lat, target_lon, dataset):
    '''
    Find the nearest coordinates to the target coordinates in the dataset.
    Parameters: target_lat (float): The target latitude.
                target_lon (float): The target longitude.
                dataset (DataFrame): The dataset containing the coordinates.
    Returns: Tuple of floats (nearest_lat, nearest_lon) of the nearest coordinates in the dataset.
    '''

    # Create a KDTree with dataset coordinates
    tree = KDTree(dataset[['latitude', 'longitude']])

    # Query the KDTree for nearest neighbor to the target coordinates
    dist, idx = tree.query([(target_lat, target_lon)])

    # Get the nearest coordinates from the dataset
    nearest_lat = dataset.iloc[idx[0]]['latitude']
    nearest_lon = dataset.iloc[idx[0]]['longitude']

    return nearest_lat, nearest_lon


def forecast_pm25(city_name, country_name, future_periods):
    '''
    Forecast PM2.5 levels for a city using the Prophet model.
    Parameters:
        city_name (str): The name of the city.
        country_name (str): The name of the country.
        future_periods (int): The number of future months to forecast.
    '''
    # Load the cleaned data
    dataset = pd.read_csv(resolve_path('airpollutionlevels/raw_data/cleaned_europe_data.csv'), parse_dates=['ds'])

    # Get city coordinates
    city_latitude, city_longitude = get_coordinates_opendatasoft(city_name, country_name)
    if city_latitude is None or city_longitude is None:
        return None, f"Could not find coordinates for {city_name}, {country_name}."

    # Find nearest coordinates in the dataset
    nearest_lat, nearest_lon = find_nearest_coordinates(city_latitude, city_longitude, dataset)

    # Filter the dataset for the nearest coordinates
    city_data = dataset[(dataset['latitude'] == nearest_lat) & (dataset['longitude'] == nearest_lon)].reset_index(drop=True)

    if city_data.empty:
        return None, f"No data available for the nearest coordinates to {city_name}, {country_name}."

    # Prepare training data
    train_data = city_data[['ds', 'y', 'latitude', 'longitude']]

    # Initialize and train Prophet model with best parameters
    best_params = {'changepoint_prior_scale': 0.01, 'holidays_prior_scale': 1.0, 'interval_width': 0.5, 'seasonality_mode': 'multiplicative', 'seasonality_prior_scale': 10.0}
    model = Prophet(**best_params)
    model.add_regressor('latitude')
    model.add_regressor('longitude')
    model.fit(train_data)

    # Forecast future PM2.5 levels
    future_dates = model.make_future_dataframe(periods=future_periods, freq='M')
    future_dates['latitude'] = nearest_lat
    future_dates['longitude'] = nearest_lon
    forecast = model.predict(future_dates)

    # Calculate mean values for the forecast period
    yhat_mean = forecast['yhat'][-future_periods:].mean()
    yhat_lower_mean = forecast['yhat_lower'][-future_periods:].mean()
    yhat_upper_mean = forecast['yhat_upper'][-future_periods:].mean()

    # Extract the trend component
    trend = forecast[['ds', 'trend']]

    # Plot the forecast and actual values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Actual vs. Forecasted PM2.5 Levels
    ax1.plot(city_data['ds'], city_data['y'], label='Actual', color='blue')
    ax1.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--', color='red')
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], label='Uncertainty Range', color='lightgreen', alpha=0.3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PM2.5 (µg/m³)')
    ax1.set_title(f'Actual vs. Forecasted PM2.5 Levels for {city_name}, {country_name}')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Trend of PM2.5 Levels
    ax2.plot(trend['ds'], trend['trend'], label='Trend', color='lightblue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Trend Value')
    ax2.set_title(f'Trend of PM2.5 Levels for {city_name}, {country_name}')
    ax2.legend()
    ax2.grid(True)

    # Generate a summary text
    summary_text = (f"For the next {future_periods} months, the forecasted PM2.5 levels range from "
          f"{yhat_lower_mean:.2f} to {yhat_upper_mean:.2f} µg/m³ with an average prediction of {yhat_mean:.2f} µg/m³.")

    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode plot image as base64 string
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return encoded_image, summary_text
