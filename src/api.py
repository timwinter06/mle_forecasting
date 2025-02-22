""" "This is the entrypoint for the API, it"""

import logging

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from predict import ModelPredictor
from settings import COLUMNS, MODEL_PATH
from utils.schemas import PredictionRequest

logging.basicConfig(level=logging.INFO)

# initialize objects required for API
logging.info("Initializing API")
app = FastAPI(title="Forecasting API", description="Simple API to forecast units.", version="0.1.0")

logging.info("Initializing model predictor")
predictor = ModelPredictor(MODEL_PATH)


@app.get("/", include_in_schema=False)
def docs_redirect():
    """Home redirects to docs (auto-generated Swagger / OpenAPI documentation)"""
    return RedirectResponse("/docs")


@app.get("/test", tags=["Main"])
async def read_test():
    """Purely for testing"""
    return {"Hello": "World"}


@app.post("/predict_units", tags=["Model"])
def predict_units(request: PredictionRequest):
    logging.info("/predict_units")
    data = [
        (
            request.StoreCount,
            request.ShelfCapacity,
            request.PromoShelfCapacity,
            request.IsPromo,
            request.ItemNumber,
            request.CategoryCode,
            request.GroupCode,
            request.month,
            request.weekday,
            request.UnitSales_minus_7,
            request.UnitSales_minus_14,
            request.UnitSales_minus_21,
        )
    ]
    input_df = pd.DataFrame(data, columns=COLUMNS)
    units = predictor.predict_units(input_df)

    return {"values": units}


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
