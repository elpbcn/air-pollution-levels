import pandas as pd

from data import encode_scale_data
from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

import joblib
import os


def train_and_save_model(df):
    """
    Train a Decision Tree model on the provided dataset and save it.

    Parameters:
    df (pd.DataFrame): Preprocessed dataset containing features and target_class.
    """
    # Specify the model file path

    model_dir = '../models/'
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
    model_filename = os.path.join(model_dir, 'decision_tree_model.pkl')

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['target_class'])
    y = df['target_class']

    # Initialize the Decision Tree model with best parameters
    dt = DecisionTreeClassifier(criterion='gini', max_depth=30, min_samples_leaf=4, min_samples_split=2, random_state=42)

    # Fit the model
    dt.fit(X, y)

    # Save the model
    joblib.dump(dt, model_filename)
    print(f"Model saved at {model_filename}")

def evaluate_model(df):
    """
    Load a saved Decision Tree model and evaluate its performance on the provided dataset.

    Parameters:
    df (pd.DataFrame): Preprocessed dataset containing features and target_class.

    """
    model_filename = '../models/decision_tree_model.pkl'
    # Load the model
    dt = joblib.load(model_filename)

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['target_class'])
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

    model_filename = '../models/decision_tree_model.pkl'
    data = pd.read_csv('../raw_data/data_lib.csv')

    # Find the last row for the provided city in the original dataset
    city_data_index = data[data['city'] == city].index[-1]

    if city_data_index is None:
        raise ValueError(f"No data found for '{city}' in the dataset.")

    # Encode and scale the entire data
    encoded_data = encode_scale_data(data)

    # Locate the corresponding row in the encoded data using the index
    encoded_city_data = encoded_data.loc[[city_data_index]].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Replace the year in the encoded city data with the input year
    encoded_city_data.loc[:, 'year'] = year

    # Prepare features for prediction (drop the 'target_class' column if it exists)
    X_predict = encoded_city_data.drop(columns=['target_class'], errors='ignore')

    # Load the model
    dt = joblib.load(model_filename)

    # Predict
    prediction = dt.predict(X_predict)[0]

    print(f"Predicted pollution level for {city} in {year}: {prediction}")
    return prediction
