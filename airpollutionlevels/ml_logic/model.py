import pandas as pd

from airpollutionlevels.ml_logic.data import encode_scale_data, encode_scale_data_rf

from scipy.stats import randint

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score , mean_absolute_error, mean_squared_error, r2_score , root_mean_squared_error
from sklearn.model_selection import train_test_split

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
    dt = DecisionTreeClassifier(criterion='gini', max_depth=30, min_samples_leaf=4, min_samples_split=2, random_state=42)

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
    model_filename = resolve_path('airpollutionlevels/models/decision_tree_model.pkl')
    dt = joblib.load(model_filename)

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['target_class' , 'unique_id'])
    y = df['target_class']

    # Make predictions
    y_pred = dt.predict(X)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y, y_pred))

    # Calculate and print accuracy
    accuracy = accuracy_score(y, y_pred)
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

     # Add a unique identifier before encoding
    city_data.loc[:, 'unique_id'] = 0  # Set a temporary unique_id for tracking

    # Encode and scale the entire data
    encoded_data = encode_scale_data(data)

    # Locate the corresponding row in the encoded data using the unique identifier
    encoded_city_data = encoded_data[encoded_data['unique_id'] == 0].copy()

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

    # Add a unique identifier before encoding
    city_data.loc[:, 'unique_id'] = 0  # Set a temporary unique_id for tracking

    # Encode and scale the entire data
    encoded_data = encode_scale_data_rf(data)

    # Locate the corresponding row in the encoded data using the unique identifier
    encoded_city_data = encoded_data[encoded_data['unique_id'] == 0].copy()

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
