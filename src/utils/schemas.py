"""
This is for defining
- custom types (type aliases)
- pydantic models (for use in the api, see https://fastapi.tiangolo.com/python-types/#pydantic-models)
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NewType, Union

import numpy
from pydantic import BaseModel, Field

PathType = Union[str, Path]
RowId = Union[str, int, float]
NumpyArray = NewType("NumpyArray", numpy.ndarray)


class DataframeModel(BaseModel):
    """For any pandas dataframe. Defined by column names and values."""

    columns: List[str]
    values: NumpyArray


class PredictionsResponse(BaseModel):
    predictions: Dict[RowId, Dict]


class ExplanationModel(BaseModel):
    """
    From the ModelExplainer that gives Lime / SHAP explanations,
    this is output for one single feature
    """

    feature_name: str
    feature_value: Union[str, float, int]
    feature_importance: float


class ExplanationsResponse(BaseModel):
    """This contains explanations for each feature, for each row in the dataset"""

    explanations: Dict[RowId, List[ExplanationModel]]


class MonitoringResponse(BaseModel):
    """Output of model monitoring"""

    model_metrics: Dict[str, float]
    data_metrics: Dict[str, float]


class RowResponse(BaseModel):
    """Full output of api fior a single record"""

    id: RowId
    data: Dict[str, Union[str, float, int]]
    prediction: float
    explanation: List[ExplanationModel]


class MainResponse(BaseModel):
    """Full output for the entire dataset"""

    result: Dict[RowId, RowResponse]


class BatchRequestModel(BaseModel):
    input_ids: List[RowId]
    callback_url: str


class BlobJob(BaseModel):
    """Background job stored in Blob storage"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created: str = Field(default_factory=lambda: str(datetime.now()))
    status: str = "in_progress"
    output: Dict = Field(default_factory=dict)


class TableJob(BaseModel):
    """Background job stored in Table storage"""

    PartitionKey: str = "001"
    RowKey: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created: str = Field(default_factory=lambda: str(datetime.now()))
    status: str = "in_progress"
    output: Dict = Field(default_factory=dict)
