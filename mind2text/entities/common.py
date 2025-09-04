"""
Common entities shared across the mind2text project.

These are public entities used across multiple layers and features.
All entities use Pydantic for validation and serialization.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

Band = Literal["delta", "theta", "alpha", "beta", "gamma"]

class ChannelInfo(BaseModel):
    """EEG channel metadata."""
    model_config = ConfigDict(frozen=True)
    name: str = Field(..., description="Channel name (10-20)", example="C3")
    index: int = Field(..., description="Zero-based channel index", example=12)

class Subject(BaseModel):
    """Dataset subject descriptor."""
    model_config = ConfigDict(frozen=True)
    subject_id: str = Field(..., description="Unique subject code", example="sub-01")
    age: Optional[int] = Field(None, description="Subject age (years)", example=25)
    sex: Optional[Literal["m","f"]] = Field(None, description="Subject sex", example="f")
    height: Optional[float] = Field(None, description="Subject height (cm)", example=170.0)
    weight: Optional[float] = Field(None, description="Subject weight (kg)", example=65.0)

class SplitTag(BaseModel):
    """Data split label."""
    model_config = ConfigDict(frozen=True)
    name: Literal["train","val","test"] = Field(..., description="Split type", example="train")
    fold: Optional[int] = Field(None, description="CV fold index if applicable", example=0)

class Trial(BaseModel):
    """Single cognitive task trial metadata."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Unique trial identifier", example="sub-01_ses-session1_task-memory")
    subject_id: str = Field(..., description="Subject identifier", example="sub-01")
    session_id: str = Field(..., description="Session identifier", example="session1")
    task: Literal["memory","mathematic","music","eyesopen","eyesclosed"] = Field(..., description="Cognitive task type", example="memory")
    tmin: float = Field(..., description="Trial start time (s)", example=0.0)
    tmax: float = Field(..., description="Trial end time (s)", example=300.0)
    sfreq: float = Field(..., description="Sampling rate (Hz)", example=500.0)
    channels: List[ChannelInfo] = Field(..., description="Ordered channel list", example=[{"name":"FCz","index":0},{"name":"C3","index":12}])
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class FeatureVector(BaseModel):
    """Band-power features per channel."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Origin trial id", example="sub-01_ses-session1_task-memory")
    bands: List[Band] = Field(..., description="Band order", example=["alpha","beta"])
    values: List[List[float]] = Field(..., description="Shape (n_channels x n_bands)", example=[[2.1, 0.9],[1.7,1.2]])
    channel_names: List[str] = Field(..., description="Channel order aligned to values rows", example=["FCz","C3"])
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class TokenSequence(BaseModel):
    """Symbolic tokenization of a trial."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Origin trial id", example="sub-01_ses-session1_task-memory")
    tokens: List[str] = Field(..., description="Ordered token list", example=["ALPHA_HIGH_FCz","BETA_LOW_C3"])
    vocab_version: str = Field(..., description="Vocab version used for IDs", example="vocab_2025-09-10")
    version: str = Field("1.0", description="Entity schema version", example="1.0")
