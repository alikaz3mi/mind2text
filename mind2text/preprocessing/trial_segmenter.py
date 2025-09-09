"""
EEG Trial Segmentation for Motor Imagery Tasks
"""

import numpy as np
import mne
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TrialSegmenter:
    """
    Handles segmentation of continuous EEG data into motor imagery trials.
    Provides trial extraction, baseline correction, and epoch creation.
    """
    
    def __init__(self, 
                 tmin: float = -1.0,
                 tmax: float = 4.0,
                 baseline: Optional[Tuple[float, float]] = (-1.0, 0.0),
                 preload: bool = True):
        """
        Initialize trial segmenter.
        
        Args:
            tmin: Start time of epochs relative to events (seconds)
            tmax: End time of epochs relative to events (seconds)
            baseline: Baseline period for correction (start, end) in seconds
            preload: Whether to preload epoch data
        """
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.preload = preload
        
        # Motor imagery event mapping
        self.event_mapping = {
            1: 'left_hand',
            2: 'right_hand',
            3: 'feet', 
            4: 'tongue'
        }
        
    def segment_trials(self, 
                      raw: mne.io.Raw,
                      events: np.ndarray,
                      event_ids: Optional[Dict[str, int]] = None,
                      reject_criteria: Optional[Dict[str, float]] = None) -> mne.Epochs:
        """
        Segment continuous EEG data into trials/epochs.
        
        Args:
            raw: Continuous EEG data
            events: Events array with shape (n_events, 3)
            event_ids: Dictionary mapping event names to IDs
            reject_criteria: Rejection criteria for bad epochs
            
        Returns:
            MNE Epochs object containing segmented trials
        """
        if event_ids is None:
            event_ids = self.event_mapping
            
        if reject_criteria is None:
            # Default rejection criteria (in microvolts)
            reject_criteria = {
                'eeg': 100e-6,  # 100 µV
                'eog': 150e-6   # 150 µV (if EOG channels present)
            }
            
        # Create epochs
        epochs = mne.Epochs(
            raw=raw,
            events=events,
            event_id=event_ids,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            reject=reject_criteria,
            preload=self.preload,
            verbose=False
        )
        
        logger.info(f"Created {len(epochs)} epochs from {len(events)} events")
        logger.info(f"Epoch duration: {self.tmax - self.tmin:.1f} seconds")
        
        return epochs
    
    def extract_motor_imagery_epochs(self,
                                   raw: mne.io.Raw,
                                   events: np.ndarray,
                                   task_duration: float = 3.0) -> mne.Epochs:
        """
        Extract motor imagery epochs with task-specific timing.
        
        Args:
            raw: Continuous EEG data
            events: Events array
            task_duration: Duration of motor imagery task in seconds
            
        Returns:
            Epochs focused on motor imagery period
        """
        # Adjust timing for motor imagery task
        # Typically: cue at 0s, motor imagery from 1s to 4s
        epochs = self.segment_trials(
            raw=raw,
            events=events,
            event_ids=self.event_mapping
        )
        
        # Crop to motor imagery period (exclude cue period)
        mi_start = 0.5  # Start motor imagery extraction 0.5s after cue
        mi_end = mi_start + task_duration
        
        epochs_mi = epochs.copy().crop(tmin=mi_start, tmax=mi_end)
        
        logger.info(f"Extracted motor imagery epochs: {mi_start}-{mi_end}s relative to cue")
        
        return epochs_mi
    
    def apply_trial_selection(self,
                            epochs: mne.Epochs,
                            min_trials_per_class: int = 50,
                            balance_classes: bool = True) -> mne.Epochs:
        """
        Apply trial selection criteria to ensure balanced dataset.
        
        Args:
            epochs: Input epochs
            min_trials_per_class: Minimum number of trials required per class
            balance_classes: Whether to balance trial counts across classes
            
        Returns:
            Filtered epochs with balanced classes
        """
        # Get trial counts per class
        trial_counts = {}
        for event_name, event_id in epochs.event_id.items():
            trial_counts[event_name] = len(epochs[event_name])
            
        logger.info(f"Original trial counts: {trial_counts}")
        
        # Check minimum trials requirement
        insufficient_classes = [
            name for name, count in trial_counts.items() 
            if count < min_trials_per_class
        ]
        
        if insufficient_classes:
            logger.warning(f"Classes with insufficient trials: {insufficient_classes}")
            
        # Balance classes if requested
        if balance_classes:
            min_count = min(trial_counts.values())
            if min_count < min_trials_per_class:
                logger.warning(f"Minimum trial count ({min_count}) below threshold ({min_trials_per_class})")
                min_count = min_trials_per_class
                
            # Subsample each class to minimum count
            balanced_epochs = []
            for event_name in trial_counts.keys():
                class_epochs = epochs[event_name]
                if len(class_epochs) >= min_count:
                    # Randomly select trials
                    indices = np.random.choice(
                        len(class_epochs), 
                        size=min_count, 
                        replace=False
                    )
                    balanced_epochs.append(class_epochs[indices])
                else:
                    logger.warning(f"Insufficient trials for {event_name}: {len(class_epochs)} < {min_count}")
                    balanced_epochs.append(class_epochs)
                    
            # Concatenate balanced epochs
            epochs_balanced = mne.concatenate_epochs(balanced_epochs)
            
            # Get final trial counts
            final_counts = {}
            for event_name in epochs_balanced.event_id.keys():
                final_counts[event_name] = len(epochs_balanced[event_name])
                
            logger.info(f"Balanced trial counts: {final_counts}")
            
            return epochs_balanced
        
        return epochs
    
    def create_cross_validation_splits(self,
                                     epochs: mne.Epochs,
                                     n_folds: int = 5,
                                     stratified: bool = True,
                                     random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits for epochs.
        
        Args:
            epochs: Input epochs
            n_folds: Number of cross-validation folds
            stratified: Whether to use stratified splits
            random_state: Random seed for reproducibility
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # Get labels
        labels = epochs.events[:, 2]  # Event IDs
        
        # Create cross-validation object
        if stratified:
            cv = StratifiedKFold(
                n_splits=n_folds, 
                shuffle=True, 
                random_state=random_state
            )
        else:
            cv = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state
            )
            
        # Generate splits
        splits = list(cv.split(np.arange(len(epochs)), labels))
        
        logger.info(f"Created {n_folds}-fold cross-validation splits")
        
        return splits
    
    def get_epoch_info(self, epochs: mne.Epochs) -> Dict:
        """
        Get comprehensive information about epochs.
        
        Args:
            epochs: Input epochs
            
        Returns:
            Dictionary with epoch information
        """
        info = {
            'n_epochs': len(epochs),
            'n_channels': len(epochs.ch_names),
            'n_times': len(epochs.times),
            'sampling_freq': epochs.info['sfreq'],
            'duration': epochs.times[-1] - epochs.times[0],
            'baseline': epochs.baseline,
            'event_counts': {},
            'bad_epochs': epochs.drop_log_stats(),
            'channel_types': set(epochs.get_channel_types())
        }
        
        # Count trials per class
        for event_name, event_id in epochs.event_id.items():
            info['event_counts'][event_name] = len(epochs[event_name])
            
        return info
    
    def export_epoch_data(self,
                         epochs: mne.Epochs,
                         format: str = 'array') -> Tuple[np.ndarray, np.ndarray]:
        """
        Export epoch data in various formats.
        
        Args:
            epochs: Input epochs
            format: Export format ('array', 'dataframe')
            
        Returns:
            Tuple of (data, labels)
        """
        # Get data array (n_epochs, n_channels, n_times)
        data = epochs.get_data()
        
        # Get labels
        labels = epochs.events[:, 2]
        
        if format == 'dataframe':
            import pandas as pd
            
            # Reshape data for DataFrame (n_epochs, n_features)
            n_epochs, n_channels, n_times = data.shape
            data_flat = data.reshape(n_epochs, n_channels * n_times)
            
            # Create column names
            columns = []
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                for time_idx, time_point in enumerate(epochs.times):
                    columns.append(f"{ch_name}_{time_point:.3f}s")
                    
            # Create DataFrame
            df = pd.DataFrame(data_flat, columns=columns)
            df['label'] = labels
            
            return df, labels
        
        return data, labels
