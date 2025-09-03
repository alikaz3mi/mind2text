"""
Pydantic domain entities for mind2text project.

This module contains all data contracts and domain models using Pydantic BaseModel.
Entities are organized by feature scope and follow strict validation rules.
"""

from .common import (
    Band,
    ChannelInfo,
    Subject,
    SplitTag,
    Trial,
    FeatureVector,
    TokenSequence
)

__all__ = [
    'Band',
    'ChannelInfo', 
    'Subject',
    'SplitTag',
    'Trial',
    'FeatureVector',
    'TokenSequence'
]
