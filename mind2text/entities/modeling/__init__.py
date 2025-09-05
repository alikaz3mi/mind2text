"""
Modeling entities for training configurations, runs, and predictions.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class ModelConfig(BaseModel):
    """Training configuration for LLM or baseline."""
    model_config = ConfigDict(frozen=True)
    base_model: Literal["distilgpt2","llama2-7b","gpt2","cnn","svm"] = Field(..., description="Backbone or baseline type", example="distilgpt2")
    lora_r: Optional[int] = Field(8, description="LoRA rank (LLM only)", example=8)
    lora_alpha: Optional[int] = Field(16, description="LoRA alpha (LLM only)", example=16)
    lr: float = Field(2e-4, description="Learning rate", example=0.0002)
    batch_size: int = Field(16, description="Global batch size", example=16)
    epochs: int = Field(5, description="Number of epochs", example=5)
    seed: int = Field(42, description="Random seed for reproducibility", example=42)
    max_sequence_length: int = Field(512, description="Maximum token sequence length", example=512)
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class TrainingRun(BaseModel):
    """Immutable record of a training run."""
    model_config = ConfigDict(frozen=True)
    run_id: str = Field(..., description="Unique run identifier", example="run_2025-09-10_12-00")
    commit: str = Field(..., description="Git commit hash", example="a1b2c3d")
    dataset_hash: str = Field(..., description="Hash of split/indices", example="ds004148_split_v1")
    config: ModelConfig = Field(..., description="Training configuration used")
    metrics: Dict[str, float] = Field(..., description="Key metrics (val accuracy, F1)", example={"val_acc":0.78,"val_f1":0.76})
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class Prediction(BaseModel):
    """Raw model prediction (pre-calibration)."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Trial id", example="sub-01_ses-session1_task-memory")
    logits: List[float] = Field(..., description="Raw logits for cognitive tasks", example=[1.2,-0.3,0.1,0.0,-0.5])
    probs: List[float] = Field(..., description="Softmax probabilities", example=[0.54,0.12,0.19,0.10,0.05])
    pred_class: Literal["memory","mathematic","music","eyesopen","eyesclosed"] = Field(..., description="Argmax class", example="memory")
    confidence: float = Field(..., description="Prediction confidence score", example=0.54)
    rationale_text: Optional[str] = Field(None, description="Generated short rationale", example="Predicted memory task — enhanced theta activity in frontal regions.")
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class CalibrationParams(BaseModel):
    """Parameters learned for probability calibration."""
    model_config = ConfigDict(frozen=True)
    method: Literal["temperature","isotonic"] = Field(..., description="Calibration method", example="temperature")
    temperature: Optional[float] = Field(1.0, description="Temperature scaling factor", example=1.2)
    per_class_params: Optional[Dict[str, float]] = Field(None, description="Optional per-class params", example={"memory":1.1})
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class PredictionCalibrated(BaseModel):
    """Post-calibrated prediction."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Trial id", example="sub-01_ses-session1_task-memory")
    probs_calibrated: List[float] = Field(..., description="Calibrated probabilities", example=[0.51,0.14,0.20,0.10,0.05])
    pred_class: Literal["memory","mathematic","music","eyesopen","eyesclosed"] = Field(..., description="Class after calibration/thresholds", example="memory")
    confidence_calibrated: float = Field(..., description="Calibrated confidence score", example=0.51)
    abstained: bool = Field(False, description="True if low-confidence abstention triggered", example=False)
    version: str = Field("1.0", description="Entity schema version", example="1.0")
