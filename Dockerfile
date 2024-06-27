FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Set the working directory
WORKDIR /airpollutionlevels/app

# Copy the requirements file
COPY ./requirements.txt /airpollutionlevels/app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /airpollutionlevels/app/requirements.txt

# Copy the application code
COPY ./airpollutionlevels/app /airpollutionlevels/app

# Command to run the application
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]
