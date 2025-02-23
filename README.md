
# MLE forecasting

A MLE project that implements a forecasting model.

## TODO

- [x] Refactor code from notebook into modules: preprocess.py, train.py, & predict.py.
- [x] Set up mlflow tracking in train.py.
- [x] Create API in api.py to serve the model.
- [ ] Add logic in predict.py to load model from mlflow model registry.
- [x] Create docker-container to serve mlflow.
- [x] Create docker-container to train model and register model to mlflow.
- [ ] Create docker-container for api.py.
- [ ] Add some model monitoring.
- [ ] Orchestration/ Automatic retrain?
- [ ] Think about CI/CD and dev -> main environments.

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

## Usage

This project consists of multiple docker containers. You can start all containers by running the docker-compose file (NOTE: you need to place the dataset.csv in the 'data/raw/' folder!):

```
docker compose -f docker-compose.yml up --build
```

MLFlow: http://localhost:5050/


- Running the API: `uvicorn src.api:app --reload`

## Deployment

- Create new Build pipeline, based on the `azure-pipelines.yml` script
  - manually using the gui
- Run the pipeline, this should at least run linters and pytest

