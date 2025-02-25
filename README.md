
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
- [ ] Add some model monitoring.
- [x] Orchestration of training
- [ ] Setting a schedule for retraining
- [ ] Automatic retraining based on drift
- [ ] Think about CI/CD, promoting models, dev -> main environments.
- [ ] Unit tests, integration tests.
- [ ] Set up S3 & postgres for better storage management.


## Usage

This project consists of multiple docker containers. You can start all containers by running the docker-compose file (NOTE: you need to place the dataset.csv in the 'data/raw/' folder!):

```
docker compose up --build
```

You can rach the different services here:

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


## Deployment

- Create new Build pipeline, based on the `azure-pipelines.yml` script
  - manually using the gui
- Run the pipeline, this should at least run linters and pytest

