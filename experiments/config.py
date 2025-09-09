"""
Configuration management for Mind2Text experiments.

This module provides structured configuration management using Pydantic
for type safety and validation.
"""

from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, validator
import yaml
import json

class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""
    model_config = ConfigDict(frozen=True)
    
    data_path: str = Field(..., description="Path to dataset root", example="data/ds004148")
    n_subjects: int = Field(10, description="Number of subjects to use", example=10)
    sessions: List[str] = Field(["session1"], description="Sessions to include", example=["session1"])
    tasks: List[str] = Field(
        ["memory", "mathematic", "music", "eyesopen", "eyesclosed"],
        description="Tasks to include",
        example=["memory", "mathematic"]
    )
    
    # Preprocessing parameters
    low_freq: float = Field(1.0, description="High-pass filter frequency (Hz)", example=1.0)
    high_freq: float = Field(40.0, description="Low-pass filter frequency (Hz)", example=40.0)
    notch_freq: Optional[float] = Field(50.0, description="Notch filter frequency (Hz)", example=50.0)
    
    # Segmentation parameters
    segment_length: float = Field(4.0, description="Segment length in seconds", example=4.0)
    overlap: float = Field(0.5, description="Overlap ratio between segments", example=0.5)
    
    # Feature extraction parameters
    bands: List[str] = Field(
        ["delta", "theta", "alpha", "beta", "gamma"],
        description="Frequency bands to extract",
        example=["alpha", "beta"]
    )
    
    @validator('overlap')
    def validate_overlap(cls, v):
        if not 0 <= v < 1:
            raise ValueError('Overlap must be between 0 and 1')
        return v
    
    @validator('n_subjects')
    def validate_n_subjects(cls, v):
        if v <= 0:
            raise ValueError('Number of subjects must be positive')
        return v

class FeatureConfig(BaseModel):
    """Configuration for feature extraction and tokenization."""
    model_config = ConfigDict(frozen=True)
    
    # Binning parameters
    n_bins: int = Field(3, description="Number of bins for discretization", example=3)
    binning_strategy: Literal["uniform", "quantile", "kmeans"] = Field(
        "quantile", description="Binning strategy", example="quantile"
    )
    
    # Encoding parameters
    include_channel_info: bool = Field(True, description="Include channel info in tokens", example=True)
    include_band_info: bool = Field(True, description="Include band info in tokens", example=True)
    include_spatial_info: bool = Field(True, description="Include spatial organization", example=True)
    
    # Tokenization parameters
    max_vocab_size: int = Field(10000, description="Maximum vocabulary size", example=10000)
    min_frequency: int = Field(2, description="Minimum token frequency", example=2)
    add_special_tokens: bool = Field(True, description="Add special tokens", example=True)
    
    @validator('n_bins')
    def validate_n_bins(cls, v):
        if v < 2:
            raise ValueError('Number of bins must be at least 2')
        return v

class ModelConfig(BaseModel):
    """Configuration for model architecture and training."""
    model_config = ConfigDict(frozen=True)
    
    # Model type
    model_type: Literal["distilgpt2", "gpt2", "cnn", "svm"] = Field(
        "distilgpt2", description="Model architecture", example="distilgpt2"
    )
    
    # LLM-specific parameters
    use_lora: bool = Field(True, description="Use LoRA for efficient training", example=True)
    lora_r: int = Field(8, description="LoRA rank", example=8)
    lora_alpha: int = Field(16, description="LoRA alpha scaling", example=16)
    lora_dropout: float = Field(0.1, description="LoRA dropout rate", example=0.1)
    
    # Training parameters
    learning_rate: float = Field(2e-4, description="Learning rate", example=0.0002)
    batch_size: int = Field(16, description="Training batch size", example=16)
    eval_batch_size: Optional[int] = Field(None, description="Evaluation batch size", example=32)
    epochs: int = Field(5, description="Number of training epochs", example=5)
    max_length: int = Field(512, description="Maximum sequence length", example=512)
    
    # Regularization
    dropout: float = Field(0.1, description="Dropout rate", example=0.1)
    weight_decay: float = Field(0.01, description="Weight decay", example=0.01)
    
    # Optimizer parameters
    optimizer: Literal["adam", "adamw", "sgd"] = Field("adamw", description="Optimizer", example="adamw")
    warmup_steps: int = Field(100, description="Learning rate warmup steps", example=100)
    
    # Baseline-specific parameters (for CNN/SVM)
    hidden_dim: int = Field(128, description="Hidden dimension for CNN", example=128)
    kernel: Literal["linear", "rbf", "poly"] = Field("rbf", description="SVM kernel", example="rbf")
    C: float = Field(1.0, description="SVM regularization parameter", example=1.0)
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError('Learning rate must be positive')
        return v
    
    @validator('eval_batch_size', always=True)
    def set_eval_batch_size(cls, v, values):
        if v is None:
            return values.get('batch_size', 16) * 2
        return v

class ExperimentConfig(BaseModel):
    """Configuration for experiment setup and tracking."""
    model_config = ConfigDict(frozen=True)
    
    # Experiment identification
    experiment_name: str = Field(..., description="Experiment name", example="cognitive_classification_v1")
    run_name: Optional[str] = Field(None, description="Specific run name", example="run_20231215_143022")
    description: str = Field("", description="Experiment description", example="Baseline comparison")
    
    # Random seed
    seed: int = Field(42, description="Random seed for reproducibility", example=42)
    
    # Data splitting
    test_size: float = Field(0.2, description="Test set proportion", example=0.2)
    val_size: float = Field(0.1, description="Validation set proportion", example=0.1)
    stratify: bool = Field(True, description="Use stratified splitting", example=True)
    
    # Cross-validation
    use_cv: bool = Field(False, description="Use cross-validation", example=False)
    cv_folds: int = Field(5, description="Number of CV folds", example=5)
    
    # Output and logging
    output_dir: str = Field("outputs", description="Output directory", example="outputs")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level", example="INFO"
    )
    save_model: bool = Field(True, description="Save trained model", example=True)
    save_predictions: bool = Field(True, description="Save predictions", example=True)
    
    # Evaluation settings
    calibration_method: Literal["temperature", "isotonic", "none"] = Field(
        "temperature", description="Calibration method", example="temperature"
    )
    generate_plots: bool = Field(True, description="Generate evaluation plots", example=True)
    
    @validator('test_size', 'val_size')
    def validate_split_size(cls, v):
        if not 0 < v < 1:
            raise ValueError('Split sizes must be between 0 and 1')
        return v

class Mind2TextConfig(BaseModel):
    """Complete configuration for Mind2Text experiments."""
    model_config = ConfigDict(frozen=True)
    
    data: DataConfig = Field(..., description="Data configuration")
    features: FeatureConfig = Field(..., description="Feature configuration") 
    model: ModelConfig = Field(..., description="Model configuration")
    experiment: ExperimentConfig = Field(..., description="Experiment configuration")
    
    version: str = Field("1.0", description="Config schema version", example="1.0")
    
    @validator('experiment')
    def validate_split_consistency(cls, experiment, values):
        test_size = experiment.test_size
        val_size = experiment.val_size
        
        if test_size + val_size >= 1.0:
            raise ValueError('Test size + validation size must be < 1.0')
        
        return experiment
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        path = Path(path)
        
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self.model_dump(), f, default_flow_style=False)
        elif path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self.model_dump(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: str) -> 'Mind2TextConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls(**data)
    
    @classmethod
    def create_default(cls) -> 'Mind2TextConfig':
        """Create default configuration."""
        return cls(
            data=DataConfig(
                data_path="data/ds004148",
                n_subjects=10,
                sessions=["session1"],
                tasks=["memory", "mathematic", "music", "eyesopen", "eyesclosed"]
            ),
            features=FeatureConfig(
                n_bins=3,
                binning_strategy="quantile"
            ),
            model=ModelConfig(
                model_type="distilgpt2",
                use_lora=True,
                epochs=5
            ),
            experiment=ExperimentConfig(
                experiment_name="cognitive_classification_default",
                description="Default configuration for cognitive state classification"
            )
        )
    
    @classmethod
    def create_baseline_cnn(cls) -> 'Mind2TextConfig':
        """Create configuration for CNN baseline."""
        base_dict = cls.create_default().model_dump()
        base_dict['model']['model_type'] = "cnn"
        base_dict['experiment']['experiment_name'] = "cnn_baseline"
        base_dict['experiment']['description'] = "CNN baseline for comparison"
        return cls(**base_dict)
    
    @classmethod
    def create_baseline_svm(cls) -> 'Mind2TextConfig':
        """Create configuration for SVM baseline."""
        base_dict = cls.create_default().model_dump()
        base_dict['model']['model_type'] = "svm"
        base_dict['experiment']['experiment_name'] = "svm_baseline" 
        base_dict['experiment']['description'] = "SVM baseline for comparison"
        return cls(**base_dict)

def create_preset_configs() -> Dict[str, Mind2TextConfig]:
    """Create a set of preset configurations for common experiments."""
    
    configs = {}
    
    # 1. Default LLM configuration
    configs['llm_default'] = Mind2TextConfig.create_default()
    
    # 2. LLM with more aggressive LoRA
    base_config = Mind2TextConfig.create_default()
    llm_aggressive_dict = base_config.model_dump()
    llm_aggressive_dict['model']['lora_r'] = 16
    llm_aggressive_dict['model']['lora_alpha'] = 32
    llm_aggressive_dict['experiment']['experiment_name'] = "llm_aggressive_lora"
    llm_aggressive_dict['experiment']['description'] = "LLM with higher rank LoRA"
    configs['llm_aggressive'] = Mind2TextConfig(**llm_aggressive_dict)
    
    # 3. LLM with longer sequences
    llm_long_dict = base_config.model_dump()
    llm_long_dict['model']['max_length'] = 1024
    llm_long_dict['features']['max_vocab_size'] = 15000
    llm_long_dict['experiment']['experiment_name'] = "llm_long_sequences"
    llm_long_dict['experiment']['description'] = "LLM with longer sequence support"
    configs['llm_long'] = Mind2TextConfig(**llm_long_dict)
    
    # 4. LLM with more bins
    llm_fine_dict = base_config.model_dump()
    llm_fine_dict['features']['n_bins'] = 5
    llm_fine_dict['experiment']['experiment_name'] = "llm_fine_binning"
    llm_fine_dict['experiment']['description'] = "LLM with finer feature discretization"
    configs['llm_fine'] = Mind2TextConfig(**llm_fine_dict)
    
    # 5. CNN baseline
    configs['cnn_baseline'] = Mind2TextConfig.create_baseline_cnn()
    
    # 6. SVM baseline
    configs['svm_baseline'] = Mind2TextConfig.create_baseline_svm()
    
    # 7. Small dataset configuration (for quick testing)
    small_dict = base_config.model_dump()
    small_dict['data']['n_subjects'] = 3
    small_dict['data']['tasks'] = ["memory", "mathematic"]
    small_dict['model']['epochs'] = 2
    small_dict['experiment']['experiment_name'] = "quick_test"
    small_dict['experiment']['description'] = "Quick test with small dataset"
    configs['quick_test'] = Mind2TextConfig(**small_dict)
    
    return configs

if __name__ == "__main__":
    # Example usage
    
    # Create and save default config
    config = Mind2TextConfig.create_default()
    config.save("configs/default.yaml")
    
    # Create all preset configs
    configs = create_preset_configs()
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    for name, config in configs.items():
        config.save(f"configs/{name}.yaml")
    
    print(f"Created {len(configs)} preset configurations in configs/")
    
    # Example of loading and modifying a config
    loaded_config = Mind2TextConfig.load("configs/default.yaml")
    print(f"Loaded config: {loaded_config.experiment.experiment_name}")
