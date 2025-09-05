"""
Feature-related entities for EEG feature extraction and tokenization.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class VocabInfo(BaseModel):
    """Vocabulary information for tokenization."""
    model_config = ConfigDict(frozen=True)
    vocab_size: int = Field(..., description="Total vocabulary size", example=1000)
    token_to_id: Dict[str, int] = Field(..., description="Token to ID mapping", example={"ALPHA_HIGH_C3": 1, "BETA_LOW_FCz": 2})
    id_to_token: Dict[int, str] = Field(..., description="ID to token mapping", example={1: "ALPHA_HIGH_C3", 2: "BETA_LOW_FCz"})
    special_tokens: Dict[str, int] = Field(..., description="Special token mappings", example={"<PAD>": 0, "<UNK>": 999})
    creation_date: str = Field(..., description="Vocabulary creation date", example="2025-09-10")
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class SpectralFeatures(BaseModel):
    """Advanced spectral features beyond basic band power."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Origin trial id", example="sub-01_ses-session1_task-memory")
    spectral_entropy: List[float] = Field(..., description="Spectral entropy per channel", example=[0.85, 0.92])
    peak_frequency: List[float] = Field(..., description="Peak frequency per channel (Hz)", example=[10.5, 22.3])
    spectral_edge_frequency: List[float] = Field(..., description="Spectral edge frequency per channel (Hz)", example=[35.2, 28.9])
    channel_names: List[str] = Field(..., description="Channel order aligned to feature arrays", example=["FCz", "C3"])
    version: str = Field("1.0", description="Entity schema version", example="1.0")
