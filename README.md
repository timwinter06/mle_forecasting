![Build Status](https://dev.azure.com/data-science-lab/test/_apis/build/status/data-science-lab.mle_forecasting?branchName=master)

# mle_forecasting

A MLE project that implements a forecasting model.

## Setup

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

*describe how to run the scripts*
- Pre-processing: ???
- Training: ???
- Run tests: `python -m pytest --junitxml=junit/test-results.xml --cov=$src --cov-report xml`
- Running the API: `uvicorn src.api:app --reload`

## Deployment

- Create new Build pipeline, based on the `azure-pipelines.yml` script
  - manually using the gui
- Run the pipeline, this should at least run linters and pytest

# Project Organization

    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   └── __init__.py    <- Makes src a Python module
    │
    ├── tests              <- Test scripts
    │    
    ├── .dockerignore      <- Definition of files to ignore in the docker image
    │    
    ├── .env               <- for setting up environment variables to use in your code
    │
    ├── .gitignore         <- to specify which files should be ignored for git
    │
    ├── azure-pipelines.yml <- The CI/CD pipeline script for Azure Pipelines
    │
    ├── README.md          <- The top-level README for developers using this project.
    │                       generated with `pip freeze > requirements.txt`
    │
    └── pyproject.toml     <- Configuration file that specifies the project's metadata, 
    │                         dependencies, and setup options for packaging tools.
    │                         Includes configurations for linters, formatters, and more.
    │
    └── .pre-commit-config.yml  <- Configuration file for pre-commit hooks to automate code quality checks
    │                             such as linting, formatting, and tests.
    │                             Ensures all code meets style standards before commits.
    │
    └── Dockerfile         <- for setting up environment variables to use in your code


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
