"""
Dataset-specific entities for EEG data handling.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class BinningRule(BaseModel):
    """Rules for discretizing continuous features into symbolic tokens."""
    model_config = ConfigDict(frozen=True)
    feature_name: str = Field(..., description="Name of the feature being binned", example="alpha_power")
    channel_name: str = Field(..., description="EEG channel name", example="C3")
    bin_edges: List[float] = Field(..., description="Bin boundaries for discretization", example=[0.0, 1.0, 2.0, 3.0])
    bin_labels: List[str] = Field(..., description="Labels for each bin", example=["LOW", "MEDIUM", "HIGH"])
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class DatasetInfo(BaseModel):
    """Metadata about the EEG dataset."""
    model_config = ConfigDict(frozen=True)
    name: str = Field(..., description="Dataset name", example="ds004148")
    n_subjects: int = Field(..., description="Number of subjects", example=60)
    n_channels: int = Field(..., description="Number of EEG channels", example=61)
    sampling_rate: float = Field(..., description="Sampling frequency (Hz)", example=500.0)
    tasks: List[str] = Field(..., description="Available cognitive tasks", example=["memory", "mathematic", "music"])
    recording_duration: float = Field(..., description="Recording duration per task (s)", example=300.0)
    version: str = Field("1.0", description="Entity schema version", example="1.0")
