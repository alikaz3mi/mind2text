"""
EEG Signal Processing and Filtering
"""

import numpy as np
import mne
from typing import Tuple, Optional, Dict, List
import scipy.signal
from scipy.stats import zscore
import logging

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Handles EEG signal preprocessing including filtering, artifact removal,
    and signal quality enhancement.
    """
    
    def __init__(self, 
                 sfreq: float = 1000.0,
                 l_freq: float = 0.5,
                 h_freq: float = 100.0,
                 notch_freq: float = 50.0):
        """
        Initialize signal processor.
        
        Args:
            sfreq: Sampling frequency in Hz
            l_freq: Low-pass filter frequency
            h_freq: High-pass filter frequency  
            notch_freq: Notch filter frequency (line noise)
        """
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        
        # Define frequency bands for feature extraction
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
    def apply_basic_filters(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply basic filtering pipeline to raw EEG data.
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Filtered raw data
        """
        # Create a copy to avoid modifying original
        raw_filtered = raw.copy()
        
        # Apply bandpass filter
        raw_filtered.filter(
            l_freq=self.l_freq, 
            h_freq=self.h_freq,
            picks='eeg',
            verbose=False
        )
        
        # Apply notch filter for line noise
        raw_filtered.notch_filter(
            freqs=self.notch_freq,
            picks='eeg',
            verbose=False
        )
        
        return raw_filtered
    
    def remove_artifacts_ica(self, 
                           raw: mne.io.Raw, 
                           n_components: int = 20,
                           random_state: int = 42) -> Tuple[mne.io.Raw, mne.preprocessing.ICA]:
        """
        Remove artifacts using Independent Component Analysis.
        
        Args:
            raw: Raw EEG data
            n_components: Number of ICA components
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (cleaned raw data, fitted ICA object)
        """
        # Fit ICA
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            verbose=False
        )
        
        # Filter data for ICA (1-40 Hz recommended)
        raw_ica = raw.copy().filter(1, 40, verbose=False)
        ica.fit(raw_ica, picks='eeg', verbose=False)
        
        # Automatically detect and exclude eye artifacts
        eog_indices, eog_scores = ica.find_bads_eog(raw_ica, verbose=False)
        ica.exclude = eog_indices
        
        # Apply ICA to remove artifacts
        raw_clean = ica.apply(raw.copy(), verbose=False)
        
        return raw_clean, ica
    
    def detect_bad_channels(self, raw: mne.io.Raw, 
                          z_threshold: float = 3.0) -> List[str]:
        """
        Detect bad channels based on statistical measures.
        
        Args:
            raw: Raw EEG data
            z_threshold: Z-score threshold for bad channel detection
            
        Returns:
            List of bad channel names
        """
        data = raw.get_data(picks='eeg')
        
        # Calculate channel statistics
        channel_vars = np.var(data, axis=1)
        channel_means = np.mean(np.abs(data), axis=1)
        
        # Z-score normalization
        var_z = np.abs(zscore(channel_vars))
        mean_z = np.abs(zscore(channel_means))
        
        # Find outliers
        bad_channels = []
        eeg_ch_names = [ch for ch in raw.ch_names if ch in mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True))['ch_names']]
        
        for i, ch_name in enumerate(eeg_ch_names):
            if var_z[i] > z_threshold or mean_z[i] > z_threshold:
                bad_channels.append(ch_name)
                
        return bad_channels
    
    def interpolate_bad_channels(self, raw: mne.io.Raw, 
                               bad_channels: Optional[List[str]] = None) -> mne.io.Raw:
        """
        Interpolate bad channels using spherical spline interpolation.
        
        Args:
            raw: Raw EEG data
            bad_channels: List of bad channel names. If None, auto-detect
            
        Returns:
            Raw data with interpolated channels
        """
        if bad_channels is None:
            bad_channels = self.detect_bad_channels(raw)
            
        if len(bad_channels) > 0:
            logger.info(f"Interpolating {len(bad_channels)} bad channels: {bad_channels}")
            raw.info['bads'] = bad_channels
            raw_interp = raw.copy().interpolate_bads(reset_bads=True, verbose=False)
        else:
            raw_interp = raw.copy()
            
        return raw_interp
    
    def apply_common_average_reference(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply common average reference (CAR) to EEG data.
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Re-referenced raw data
        """
        raw_car = raw.copy()
        raw_car.set_eeg_reference('average', projection=True, verbose=False)
        raw_car.apply_proj(verbose=False)
        
        return raw_car
    
    def preprocess_pipeline(self, raw: mne.io.Raw, 
                          apply_ica: bool = True,
                          interpolate_bads: bool = True) -> Dict[str, mne.io.Raw]:
        """
        Complete preprocessing pipeline.
        
        Args:
            raw: Raw EEG data
            apply_ica: Whether to apply ICA artifact removal
            interpolate_bads: Whether to interpolate bad channels
            
        Returns:
            Dictionary containing processed data at different stages
        """
        pipeline_data = {'raw': raw.copy()}
        
        # 1. Basic filtering
        raw_filtered = self.apply_basic_filters(raw)
        pipeline_data['filtered'] = raw_filtered
        
        # 2. Bad channel detection and interpolation
        if interpolate_bads:
            raw_interp = self.interpolate_bad_channels(raw_filtered)
            pipeline_data['interpolated'] = raw_interp
            current_raw = raw_interp
        else:
            current_raw = raw_filtered
            
        # 3. ICA artifact removal
        if apply_ica:
            raw_clean, ica = self.remove_artifacts_ica(current_raw)
            pipeline_data['ica_cleaned'] = raw_clean
            pipeline_data['ica_object'] = ica
            current_raw = raw_clean
            
        # 4. Common average reference
        raw_car = self.apply_common_average_reference(current_raw)
        pipeline_data['final'] = raw_car
        
        return pipeline_data
