FROM python:latest

WORKDIR /prod

COPY airpollutionlevels/api airpollutionlevels/api
COPY airpollutionlevels/ml_logic airpollutionlevels/ml_logic
COPY airpollutionlevels/models airpollutionlevels/models
COPY airpollutionlevels/raw_data airpollutionlevels/raw_data
COPY airpollutionlevels/config.py airpollutionlevels/config.py
COPY .env .env
COPY Makefile Makefile

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY setup.py setup.py
RUN pip install .

CMD uvicorn airpollutionlevels.api.api:app --host 0.0.0.0 --port $PORT
