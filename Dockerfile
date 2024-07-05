# Use Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /prod

# Copy project files
COPY airpollutionlevels/api airpollutionlevels/api
COPY airpollutionlevels/ml_logic airpollutionlevels/ml_logic
COPY airpollutionlevels/models airpollutionlevels/models
COPY airpollutionlevels/raw_data airpollutionlevels/raw_data
COPY airpollutionlevels/config.py airpollutionlevels/config.py
COPY Makefile Makefile
COPY requirements.txt requirements.txt
COPY setup.py setup.py

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD uvicorn airpollutionlevels.api.api:app --host 0.0.0.0 --port $PORT
