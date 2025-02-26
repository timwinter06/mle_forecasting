
# MLE forecasting

A MLE project that implements a forecasting model.

## TODO

- [x] Refactor code from notebook into modules: preprocess.py, train.py, & predict.py.
- [x] Set up mlflow tracking in train.py.
- [x] Create API in api.py to serve the model.
- [x] Add logic in predict.py to load model from mlflow model registry.
- [x] Create docker-container to serve mlflow.
- [x] Create docker-container to train model and register model to mlflow.
- [x] Create docker-container for api.py.
- [x] Add some model monitoring.
- [x] Orchestration of training
- [x] Set a schedule for retraining
- [ ] Create a batch predict endpoint.
- [ ] Log batch input and output in API, so that the drift-detector can read this data.
- [ ] Automatic retraining based on drift
- [ ] Think about CI/CD, promoting models, dev -> main environments.
- [ ] Unit tests, integration tests.
- [ ] Mount volume to mflow?
- [ ] Set up S3 & postgres for better storage management.


## Usage

This project consists of multiple services that run in docker-containers. These are as follows:

1. A training pipeline orchestrated by Prefect (check the Prefect UI to monitor this).
2. MLFlow UI to check the training parameters, metrics, and registered models.
3. A Prefect UI to monitor the training pipeline.
4. A fast-API prediction endpoint to make predictions with your model. NOTE: there is a `wait_for_mlflow.py` script that checks if the model has been registered before starting the API.
5. A drift-detection pipeline that runs every 5 minutes. This is a dummy pipeline that detects drift on simulated drifted production data. In future, this should read in data from the predict-API. 

You can start all containers by running the docker-compose file (NOTE: you need to place the dataset.csv in the 'data/raw/' folder!):

```
docker compose up --build
```

You can reach the different services here:

- MLFlow: http://localhost:5050/
- Prefect: http://localhost:4200/
- Prediction-API: http://localhost:8000/


## Setup for development

Anaconda environment:
```
conda create -n mle_forecasting python=3.11
conda activate mle_forecasting
pip install -e '.[lint,test]'
```

Next install your pre-commit hooks:
```
pre-commit install
pre-commit run --all-files
```
