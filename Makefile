.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y air-pollution-levels || :
	@pip install -e .


# Define the environment variable for the project directory
export PROJECT_DIR := $(shell pwd)

# Preprocess Classification Data
preprocess_classification:
	@echo "Running classification preprocessing..."
	python3 -c "from airpollutionlevels.interface.main import preprocess_classification; preprocess_classification()"

# Preprocess Regression Data
preprocess_regression:
	@echo "Running regression preprocessing..."
	python3 -c "from airpollutionlevels.interface.main import preprocess_regression; preprocess_regression()"

# Train and Save Classification Model
train_and_save_model:
	@echo "Training and saving the classification model..."
	python3 -c "from airpollutionlevels.ml_logic.model import train_and_save_model; train_and_save_model()"

# Evaluate Classification Model
evaluate_model:
	@echo "Evaluating the classification model..."
	python3 -c "from airpollutionlevels.ml_logic.model import evaluate_model; evaluate_model()"

# Predict Pollution Level (Classification)
predict:
	@echo "Predicting pollution level..."
	@read -p 'Enter city: ' city; \
	read -p 'Enter year: ' year; \
	python3 -c "from airpollutionlevels.ml_logic.model import predict; predict('$${city}', int($${year}))"

# Train and Save Regression Model
train_and_save_model_rf:
	@echo "Training and saving the regression model..."
	python3 -c "from airpollutionlevels.ml_logic.model import train_and_save_model_rf; train_and_save_model_rf()"

# Evaluate Regression Model
evaluate_model_rf:
	@echo "Evaluating the regression model..."
	python3 -c "from airpollutionlevels.ml_logic.model import evaluate_model_rf; evaluate_model_rf()"

# Predict PM2.5 Concentration (Regression)
predict_rf:
	@echo "Predicting PM2.5 concentration..."
	@read -p 'Enter city: ' city; \
	read -p 'Enter year: ' year; \
	python3 -c "from airpollutionlevels.ml_logic.model import predict_rf; predict_rf('$${city}', int($${year}))"

# Data Preparation for Time Series
data_prep_timeseries:
	@echo "Preparing data for time series..."
	python3 -c "from airpollutionlevels.ml_logic.data import data_prep_timeseries; data_prep_timeseries()"

# Help target to display available targets and their descriptions
help:
	@echo "  Available targets:"
	@echo "  preprocess_classification:    Preprocess data for classification. Run only once."
	@echo "  preprocess_regression:        Preprocess data for regression. Run only once."
	@echo "  train_and_save_model:         Train and save the classification model."
	@echo "  evaluate_model:               Evaluate the classification model."
	@echo "  predict:                      Predict pollution level using the classification model."
	@echo "  train_and_save_model_rf:      Train and save the regression model."
	@echo "  evaluate_model_rf:            Evaluate the regression model."
	@echo "  predict_rf:                   Predict PM2.5 concentration using the regression model."
	@echo "  data_prep_timeseries:         Prepare data for time series analysis."
	@echo "  help:                         Help display."
