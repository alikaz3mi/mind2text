"""
Reporting entities for evaluation metrics and visualization.
"""

from typing import Dict, List
from pydantic import BaseModel, Field, ConfigDict

class ConfusionMatrix(BaseModel):
    """Dense confusion matrix with label order."""
    model_config = ConfigDict(frozen=True)
    labels: List[str] = Field(..., description="Class order", example=["memory","mathematic","music","eyesopen","eyesclosed"])
    matrix: List[List[int]] = Field(..., description="Counts (len=labels x labels)", example=[[50,3,2,1,0],[4,48,5,2,1],[3,4,45,6,2],[2,1,5,49,3],[1,2,3,4,50]])
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class ClassificationMetrics(BaseModel):
    """Per-class and overall classification metrics."""
    model_config = ConfigDict(frozen=True)
    accuracy: float = Field(..., description="Overall accuracy", example=0.78)
    macro_f1: float = Field(..., description="Macro F1-score", example=0.76)
    weighted_f1: float = Field(..., description="Weighted F1-score", example=0.77)
    per_class_precision: Dict[str, float] = Field(..., description="Precision per class", example={"memory": 0.82, "mathematic": 0.75})
    per_class_recall: Dict[str, float] = Field(..., description="Recall per class", example={"memory": 0.80, "mathematic": 0.78})
    per_class_f1: Dict[str, float] = Field(..., description="F1-score per class", example={"memory": 0.81, "mathematic": 0.76})
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class CalibrationMetrics(BaseModel):
    """Probability calibration quality metrics."""
    model_config = ConfigDict(frozen=True)
    ece: float = Field(..., description="Expected calibration error", example=0.04)
    ace: float = Field(..., description="Average calibration error", example=0.03)
    mce: float = Field(..., description="Maximum calibration error", example=0.12)
    brier_score: float = Field(..., description="Brier score", example=0.18)
    nll: float = Field(..., description="Negative log-likelihood", example=0.72)
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class Report(BaseModel):
    """Comprehensive evaluation report."""
    model_config = ConfigDict(frozen=True)
    run_id: str = Field(..., description="Training/eval run id", example="run_2025-09-10_12-00")
    classification_metrics: ClassificationMetrics = Field(..., description="Classification performance metrics")
    calibration_metrics: CalibrationMetrics = Field(..., description="Calibration quality metrics")
    confusion: ConfusionMatrix = Field(..., description="Confusion matrix entity")
    dataset_split: str = Field(..., description="Dataset split evaluated", example="test")
    model_type: str = Field(..., description="Type of model evaluated", example="distilgpt2_lora")
    evaluation_timestamp: str = Field(..., description="When evaluation was performed", example="2025-09-10T12:00:00Z")
    version: str = Field("1.0", description="Entity schema version", example="1.0")
