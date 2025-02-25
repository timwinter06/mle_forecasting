"""Module for checking if a model is registered in MLflow."""

import logging
import time

from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from settings import MLFLOW_TRACKING_URI, MODEL_NAME

logging.basicConfig(level=logging.INFO)
logging.info("Checking if model is registered in MLflow...")

POLL_INTERVAL = 10  # Seconds to wait between checks


def is_model_registered(model_name: str) -> bool:
    """Check if the model has at least one registered version.

    Args:
        model_name (str): The name of the model.

    Returns:
        bool: True if the model is registered, False otherwise.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return len(versions) > 0
    except RestException as e:
        logging.info(f"Error checking model registry: {e}")
        return False


def wait_for_model_registration():
    """Waits until the model is registered in MLflow."""
    logging.info(f"Checking MLflow Model Registry for model: {MODEL_NAME}")

    while True:
        if is_model_registered(MODEL_NAME):
            logging.info("Model is registered! Starting the API...")
            break
        else:
            logging.info(f"Model not found. Waiting {POLL_INTERVAL}s before retrying.")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    wait_for_model_registration()
