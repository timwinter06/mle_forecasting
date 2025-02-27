
# MLE forecasting

This repo consists of code that trains a model that forecasts the Unit Sales one week ahead in case of a promotion or no-promotion on an article level. The training is scheduled to run weekly orchestrated by prefect. Model metrics and hyperparameters are logged to MLFlow for each run and the model is registered in the model registry. The trained model is deployed to Fast-API which has an endpoint with whiich you can request predictions for single records. There is also a (dummy) drift-detection pipeline (again orchestrated by prefect) in place which runs an automated drift-detection with Evidently ona the training data and a simulated drift dataset. Drift metrics are now logged to MLFlow under the 'Drift Detection' experiment. In future work, this pipeline should request the real production data from the predict API to perform automated drift detections.


## Plan & todos

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

1. A training pipeline orchestrated by Prefect to run every Sunday at midnight (check the Prefect UI to monitor this).
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


### Testing
This is still WIP, but there is one testing module for the preprocessing.py module. These tests can be run by running `pytest` in the command prompt.