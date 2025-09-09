"""
EEG Feature Extraction for Cognitive State Classification

Extracts various features from EEG signals and returns Pydantic entities.
Features include band power, spectral features, and connectivity measures.
"""

import numpy as np
import mne
from typing import Dict, List, Tuple, Optional, Union
import scipy.signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

from ..entities.common import FeatureVector, Band
from ..entities.features import SpectralFeatures

LOGGER = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts various features from EEG signals for cognitive state classification.
    Returns validated Pydantic entities for type safety.
    """
    
    def __init__(self, sfreq: float = 500.0):
        """
        Initialize feature extractor.
        
        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz
        """
        self.sfreq = sfreq
        
        # Define frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Relevant channels for cognitive tasks (frontal, central, parietal)
        self.cognitive_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',  # Frontal
            'C3', 'C4', 'Cz',  # Central
            'P3', 'P4', 'Pz', 'P7', 'P8',  # Parietal
            'O1', 'O2', 'Oz',  # Occipital
            'T7', 'T8'  # Temporal
        ]
        
    def extract_band_power_features(self, 
                                  epochs: mne.Epochs,
                                  channels: Optional[List[str]] = None,
                                  normalize: bool = True) -> np.ndarray:
        """
        Extract band power features from epoched EEG data.
        
        Args:
            epochs: MNE epochs object
            channels: List of channel names to use. If None, uses motor channels
            normalize: Whether to normalize band powers
            
        Returns:
            Feature array of shape (n_epochs, n_channels * n_bands)
        """
        if channels is None:
            # Use available motor channels
            channels = [ch for ch in self.motor_channels if ch in epochs.ch_names]
            
        # Pick channels
        epochs_picked = epochs.copy().pick_channels(channels)
        
        n_epochs = len(epochs_picked)
        n_channels = len(channels)
        n_bands = len(self.freq_bands)
        
        # Initialize feature array
        features = np.zeros((n_epochs, n_channels * n_bands))
        
        for epoch_idx in range(n_epochs):
            epoch_data = epochs_picked[epoch_idx].get_data()[0]  # Shape: (n_channels, n_times)
            
            feature_idx = 0
            for ch_idx, channel in enumerate(channels):
                signal = epoch_data[ch_idx]
                
                for band_name, (low_freq, high_freq) in self.freq_bands.items():
                    # Compute band power using Welch's method
                    freqs, psd = scipy.signal.welch(
                        signal, 
                        fs=self.sfreq,
                        nperseg=int(self.sfreq * 2),  # 2-second windows
                        noverlap=int(self.sfreq * 1)   # 1-second overlap
                    )
                    
                    # Find frequency indices
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    
                    # Compute band power (area under PSD curve)
                    band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                    
                    features[epoch_idx, feature_idx] = band_power
                    feature_idx += 1
                    
        # Normalize features if requested
        if normalize:
            # Log transform to handle skewed distributions
            features = np.log1p(features)
            
            # Z-score normalization
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            
        return features
    
    def extract_spectral_features(self, 
                                epochs: mne.Epochs,
                                channels: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract spectral features including entropy, peak frequency, etc.
        
        Args:
            epochs: MNE epochs object
            channels: List of channel names to use
            
        Returns:
            Feature array with spectral features
        """
        if channels is None:
            channels = [ch for ch in self.motor_channels if ch in epochs.ch_names]
            
        epochs_picked = epochs.copy().pick_channels(channels)
        n_epochs = len(epochs_picked)
        n_channels = len(channels)
        
        # Features: spectral entropy, peak frequency, spectral centroid
        n_features_per_channel = 3
        features = np.zeros((n_epochs, n_channels * n_features_per_channel))
        
        for epoch_idx in range(n_epochs):
            epoch_data = epochs_picked[epoch_idx].get_data()[0]
            
            feature_idx = 0
            for ch_idx, channel in enumerate(channels):
                signal = epoch_data[ch_idx]
                
                # Compute power spectral density
                freqs, psd = scipy.signal.welch(
                    signal,
                    fs=self.sfreq,
                    nperseg=int(self.sfreq * 2)
                )
                
                # Remove DC component
                freqs = freqs[1:]
                psd = psd[1:]
                
                # Normalize PSD
                psd_norm = psd / np.sum(psd)
                
                # 1. Spectral entropy
                spec_entropy = entropy(psd_norm)
                features[epoch_idx, feature_idx] = spec_entropy
                feature_idx += 1
                
                # 2. Peak frequency
                peak_freq = freqs[np.argmax(psd)]
                features[epoch_idx, feature_idx] = peak_freq
                feature_idx += 1
                
                # 3. Spectral centroid
                spec_centroid = np.sum(freqs * psd_norm)
                features[epoch_idx, feature_idx] = spec_centroid
                feature_idx += 1
                
        return features
    
    def extract_connectivity_features(self, 
                                    epochs: mne.Epochs,
                                    method: str = 'coh') -> np.ndarray:
        """
        Extract connectivity features between electrode pairs.
        
        Args:
            epochs: MNE epochs object
            method: Connectivity method ('coh', 'plv', 'pli')
            
        Returns:
            Connectivity feature array
        """
        # Use motor channels for connectivity
        channels = [ch for ch in self.motor_channels if ch in epochs.ch_names]
        epochs_picked = epochs.copy().pick_channels(channels)
        
        n_epochs = len(epochs_picked)
        n_channels = len(channels)
        n_connections = (n_channels * (n_channels - 1)) // 2  # Upper triangular
        
        features = np.zeros((n_epochs, n_connections))
        
        for epoch_idx in range(n_epochs):
            epoch_data = epochs_picked[epoch_idx].get_data()[0]
            
            # Compute connectivity matrix
            if method == 'coh':
                # Coherence
                conn_matrix = self._compute_coherence_matrix(epoch_data)
            elif method == 'plv':
                # Phase Locking Value
                conn_matrix = self._compute_plv_matrix(epoch_data)
            else:
                raise ValueError(f"Unknown connectivity method: {method}")
            
            # Extract upper triangular values
            triu_indices = np.triu_indices(n_channels, k=1)
            features[epoch_idx] = conn_matrix[triu_indices]
            
        return features
    
    def _compute_coherence_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute coherence matrix between all channel pairs."""
        n_channels = data.shape[0]
        coherence_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    coherence_matrix[i, j] = 1.0
                else:
                    freqs, coh = scipy.signal.coherence(
                        data[i], data[j], 
                        fs=self.sfreq,
                        nperseg=int(self.sfreq * 2)
                    )
                    
                    # Average coherence in alpha and beta bands
                    alpha_mask = (freqs >= 8) & (freqs <= 13)
                    beta_mask = (freqs >= 13) & (freqs <= 30)
                    
                    coherence_value = np.mean(coh[alpha_mask | beta_mask])
                    coherence_matrix[i, j] = coherence_value
                    coherence_matrix[j, i] = coherence_value
                    
        return coherence_matrix
    
    def _compute_plv_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute Phase Locking Value matrix."""
        n_channels = data.shape[0]
        plv_matrix = np.zeros((n_channels, n_channels))
        
        # Compute analytic signals using Hilbert transform
        analytic_signals = scipy.signal.hilbert(data, axis=1)
        phases = np.angle(analytic_signals)
        
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    plv_matrix[i, j] = 1.0
                else:
                    # Compute phase difference
                    phase_diff = phases[i] - phases[j]
                    
                    # Compute PLV
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_matrix[i, j] = plv
                    plv_matrix[j, i] = plv
                    
        return plv_matrix
    
    def extract_all_features(self, 
                           epochs: mne.Epochs,
                           include_spectral: bool = True,
                           include_connectivity: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract all available features from epochs.
        
        Args:
            epochs: MNE epochs object
            include_spectral: Whether to include spectral features
            include_connectivity: Whether to include connectivity features
            
        Returns:
            Dictionary with feature arrays
        """
        features_dict = {}
        
        # Band power features (always included)
        features_dict['band_power'] = self.extract_band_power_features(epochs)
        
        # Spectral features
        if include_spectral:
            features_dict['spectral'] = self.extract_spectral_features(epochs)
            
        # Connectivity features
        if include_connectivity:
            features_dict['connectivity'] = self.extract_connectivity_features(epochs)
            
        return features_dict
    
    def create_feature_names(self, 
                           channels: Optional[List[str]] = None,
                           include_spectral: bool = True,
                           include_connectivity: bool = False) -> List[str]:
        """
        Create descriptive names for all features.
        
        Args:
            channels: List of channel names
            include_spectral: Whether spectral features are included
            include_connectivity: Whether connectivity features are included
            
        Returns:
            List of feature names
        """
        if channels is None:
            channels = self.motor_channels
            
        feature_names = []
        
        # Band power feature names
        for channel in channels:
            for band_name in self.freq_bands.keys():
                feature_names.append(f"{band_name}_{channel}")
                
        # Spectral feature names
        if include_spectral:
            spectral_features = ['entropy', 'peak_freq', 'centroid']
            for channel in channels:
                for spec_feature in spectral_features:
                    feature_names.append(f"{spec_feature}_{channel}")
                    
        # Connectivity feature names
        if include_connectivity:
            for i, ch1 in enumerate(channels):
                for j, ch2 in enumerate(channels[i+1:], i+1):
                    feature_names.append(f"conn_{ch1}_{ch2}")
                    
        return feature_names
