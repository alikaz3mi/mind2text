"""
EEG Data Loading for ds004148 Cognitive State Dataset

Handles loading BIDS-formatted EEG data with cognitive tasks:
- memory, mathematic, music, eyesopen, eyesclosed
- Returns Pydantic entities for type safety
"""

import os
import numpy as np
import pandas as pd
import mne
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json

from ..entities.common import Subject, Trial, ChannelInfo
from ..entities.dataset import DatasetInfo

LOGGER = logging.getLogger(__name__)

class EEGDataLoader:
    """
    Handles loading and initial processing of BIDS EEG data for cognitive state classification.
    Returns validated Pydantic entities.
    """
    
    def __init__(self, 
                 data_path: str,
                 subject_ids: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Initialize EEG data loader for ds004148 dataset.
        
        Parameters
        ----------
        data_path : str
            Path to the ds004148 dataset directory
        subject_ids : Optional[List[str]]
            List of subject IDs to load (e.g., ['sub-01', 'sub-02']). If None, loads all subjects
        verbose : bool
            Whether to print detailed loading information
        """
        self.data_path = Path(data_path)
        self.subject_ids = subject_ids
        self.verbose = verbose
        
        # Cognitive task mapping for ds004148
        self.tasks = ['memory', 'mathematic', 'music', 'eyesopen', 'eyesclosed']
        self.sessions = ['session1', 'session2', 'session3']
        
        # Load dataset metadata
        self.dataset_info = self._load_dataset_info()
        
    def _load_dataset_info(self) -> DatasetInfo:
        """
        Load dataset metadata from BIDS files.
        
        Returns
        -------
        DatasetInfo
            Validated dataset information entity
        """
        # Load dataset description
        desc_file = self.data_path / "dataset_description.json"
        with open(desc_file, 'r') as f:
            desc = json.load(f)
            
        # Load participants info to count subjects
        participants_file = self.data_path / "participants.tsv"
        participants_df = pd.read_csv(participants_file, sep='\t')
        n_subjects = len(participants_df)
        
        # Get channel count from first subject
        sample_eeg_file = self.data_path / "sub-01" / "ses-session1" / "eeg" / "sub-01_ses-session1_task-memory_eeg.json"
        with open(sample_eeg_file, 'r') as f:
            eeg_info = json.load(f)
            
        return DatasetInfo(
            name="ds004148",
            n_subjects=n_subjects,
            n_channels=eeg_info['EEGChannelCount'],
            sampling_rate=eeg_info['SamplingFrequency'],
            tasks=self.tasks,
            recording_duration=eeg_info['RecordingDuration'],
            version="1.0"
        )
    
    def load_subjects_metadata(self) -> List[Subject]:
        """
        Load subject metadata from participants.tsv.
        
        Returns
        -------
        List[Subject]
            List of validated Subject entities
        """
        participants_file = self.data_path / "participants.tsv"
        participants_df = pd.read_csv(participants_file, sep='\t')
        
        subjects = []
        for _, row in participants_df.iterrows():
            subject = Subject(
                subject_id=row['participant_id'],
                age=int(row['age']) if pd.notna(row['age']) else None,
                sex=row['sex'] if pd.notna(row['sex']) else None,
                height=float(row['Height']) if pd.notna(row['Height']) else None,
                weight=float(row['Weight']) if pd.notna(row['Weight']) else None
            )
            subjects.append(subject)
            
        if self.subject_ids:
            subjects = [s for s in subjects if s.subject_id in self.subject_ids]
            
        return subjects
    
    def get_standard_montage(self) -> mne.channels.DigMontage:
        """
        Get standard 10-20 electrode montage for 64-channel EEG.
        
        Returns:
            Standard montage for consistent channel positioning
        """
        return mne.channels.make_standard_montage('standard_1020')
    
    def validate_data_quality(self, raw: mne.io.Raw, events: np.ndarray) -> Dict[str, bool]:
        """
        Validate EEG data quality and completeness.
        
        Args:
            raw: Raw EEG data
            events: Events array
            
        Returns:
            Dictionary of validation results
        """
        validation = {
            'has_64_channels': raw.info['nchan'] == 64,
            'sampling_rate_1000hz': raw.info['sfreq'] == 1000,
            'has_all_classes': len(np.unique(events[:, 2])) == 4,
            'sufficient_trials': len(events) >= 100,
            'no_flat_channels': not np.any(np.var(raw.get_data(), axis=1) < 1e-12)
        }
        
        return validation
