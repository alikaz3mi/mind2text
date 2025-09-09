"""
Comprehensive experiment runner for Mind2Text.

This script provides a unified interface for running experiments with
proper configuration management, logging, and result tracking.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import Mind2TextConfig, create_preset_configs
from experiments.train_models import main as train_main
from experiments.evaluate_model import main as eval_main

def setup_logging(log_level: str, output_dir: Path) -> None:
    """Setup logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Logging configured - Level: {log_level}, File: {log_file}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Mind2Text experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python run_experiment.py
  
  # Use a preset configuration
  python run_experiment.py --config llm_default
  
  # Load custom configuration file
  python run_experiment.py --config_file configs/my_experiment.yaml
  
  # Run only training
  python run_experiment.py --mode train
  
  # Run evaluation on existing model
  python run_experiment.py --mode eval --model_path outputs/llm_default_20231215/
  
  # Quick test run
  python run_experiment.py --config quick_test
        """
    )
    
    # Configuration arguments
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config", type=str, default="llm_default",
        choices=list(create_preset_configs().keys()),
        help="Preset configuration to use"
    )
    config_group.add_argument(
        "--config_file", type=str,
        help="Path to custom configuration file"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["train", "eval", "full"],
        help="Experiment mode"
    )
    
    # Training overrides
    train_group = parser.add_argument_group("Training overrides")
    train_group.add_argument("--epochs", type=int, help="Override number of epochs")
    train_group.add_argument("--batch_size", type=int, help="Override batch size")
    train_group.add_argument("--learning_rate", type=float, help="Override learning rate")
    train_group.add_argument("--n_subjects", type=int, help="Override number of subjects")
    
    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--model_path", type=str, help="Path to trained model (for eval mode)")
    eval_group.add_argument("--test_data_path", type=str, help="Path to test data")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--run_name", type=str, help="Override run name")
    parser.add_argument("--dry_run", action="store_true", help="Show configuration without running")
    
    return parser.parse_args()

def load_configuration(args) -> Mind2TextConfig:
    """Load and validate configuration."""
    logger = logging.getLogger(__name__)
    
    if args.config_file:
        logger.info(f"📄 Loading configuration from {args.config_file}")
        config = Mind2TextConfig.load(args.config_file)
    else:
        logger.info(f"🎛️  Using preset configuration: {args.config}")
        preset_configs = create_preset_configs()
        config = preset_configs[args.config]
    
    # Apply command line overrides
    config_dict = config.model_dump()
    
    if args.epochs is not None:
        config_dict['model']['epochs'] = args.epochs
    if args.batch_size is not None:
        config_dict['model']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config_dict['model']['learning_rate'] = args.learning_rate
    if args.n_subjects is not None:
        config_dict['data']['n_subjects'] = args.n_subjects
    if args.output_dir is not None:
        config_dict['experiment']['output_dir'] = args.output_dir
    if args.run_name is not None:
        config_dict['experiment']['run_name'] = args.run_name
    
    # Recreate config with overrides
    config = Mind2TextConfig(**config_dict)
    
    # Set run name if not provided
    if config.experiment.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.experiment.experiment_name}_{timestamp}"
        config_dict['experiment']['run_name'] = run_name
        config = Mind2TextConfig(**config_dict)
    
    return config

def prepare_output_directory(config: Mind2TextConfig) -> Path:
    """Prepare output directory and save configuration."""
    output_dir = Path(config.experiment.output_dir) / config.experiment.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(str(output_dir / "config.yaml"))
    
    # Create subdirectories
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    
    return output_dir

def create_training_args(config: Mind2TextConfig, output_dir: Path):
    """Create argument namespace for training script."""
    class Args:
        pass
    
    args = Args()
    
    # Data arguments
    args.data_path = config.data.data_path
    args.n_subjects = config.data.n_subjects
    args.sessions = config.data.sessions
    args.tasks = config.data.tasks
    
    # Feature arguments
    args.segment_length = config.data.segment_length
    args.overlap = config.data.overlap
    args.n_bins = config.features.n_bins
    
    # Model arguments
    args.model_type = config.model.model_type
    args.use_lora = config.model.use_lora
    args.lora_r = config.model.lora_r
    args.lora_alpha = config.model.lora_alpha
    
    # Training arguments
    args.batch_size = config.model.batch_size
    args.learning_rate = config.model.learning_rate
    args.epochs = config.model.epochs
    args.max_length = config.model.max_length
    args.seed = config.experiment.seed
    
    # Output arguments
    args.output_dir = str(output_dir)
    args.run_name = config.experiment.run_name
    args.save_model = config.experiment.save_model
    
    return args

def create_evaluation_args(config: Mind2TextConfig, output_dir: Path, model_path: Optional[str] = None):
    """Create argument namespace for evaluation script."""
    class Args:
        pass
    
    args = Args()
    
    # Model and data paths
    if model_path:
        args.model_path = model_path
    else:
        args.model_path = str(output_dir)
    
    # For now, use a placeholder - in real implementation this would be saved test data
    args.test_data_path = str(output_dir / "test_data.json")
    
    # Evaluation arguments
    args.calibration_method = config.experiment.calibration_method
    args.batch_size = config.model.eval_batch_size or config.model.batch_size
    args.n_bins = 15
    
    # Output arguments
    args.output_dir = str(output_dir / "evaluation")
    args.save_predictions = config.experiment.save_predictions
    args.generate_plots = config.experiment.generate_plots
    
    return args

def run_training(config: Mind2TextConfig, output_dir: Path):
    """Run training phase."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting training phase...")
    
    # Create training arguments
    train_args = create_training_args(config, output_dir)
    
    # Import and run training
    from experiments.train_models import (
        load_and_preprocess_data, create_token_sequences, split_data,
        train_baseline_model, create_training_report
    )
    
    try:
        # Load and preprocess data
        feature_vectors, labels = load_and_preprocess_data(train_args)
        
        # Create token sequences
        token_sequences, tokenizer, binner, encoder = create_token_sequences(
            feature_vectors, labels, train_args
        )
        
        # Split data
        (train_sequences, val_sequences, test_sequences,
         train_labels, val_labels, test_labels) = split_data(
            token_sequences, labels, 
            test_size=config.experiment.test_size,
            val_size=config.experiment.val_size,
            seed=config.experiment.seed
        )
        
        # Save test data for evaluation
        test_data = {
            'token_sequences': test_sequences,
            'labels': test_labels
        }
        import json
        with open(output_dir / "test_data.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save preprocessing artifacts
        tokenizer.save_vocabulary(str(output_dir / "vocabulary.json"))
        binner.save_binning_rules(str(output_dir / "binning_rules.json"))
        
        # Train model
        if config.model.model_type in ['cnn', 'svm']:
            model, classification_metrics, confusion_matrix_entity = train_baseline_model(
                train_sequences, train_labels, val_sequences, val_labels,
                test_sequences, test_labels, tokenizer, train_args
            )
        else:
            # TODO: Implement LLM training
            logger.warning("⚠️  LLM training not fully implemented yet")
            return output_dir
        
        # Create training report
        report = create_training_report(
            train_args, classification_metrics, confusion_matrix_entity, 
            datetime.now()
        )
        
        # Save report
        with open(output_dir / "training_report.json", 'w') as f:
            json.dump(report.model_dump(), f, indent=2)
        
        # Save model
        if config.experiment.save_model and hasattr(model, 'save'):
            model.save(str(output_dir / "model"))
        
        logger.info("✅ Training completed successfully!")
        return output_dir
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

def run_evaluation(config: Mind2TextConfig, output_dir: Path, model_path: Optional[str] = None):
    """Run evaluation phase."""
    logger = logging.getLogger(__name__)
    logger.info("🔬 Starting evaluation phase...")
    
    # Check if test data exists
    test_data_path = output_dir / "test_data.json"
    if not test_data_path.exists():
        logger.warning("⚠️  No test data found - skipping evaluation")
        return
    
    # Create evaluation arguments
    eval_args = create_evaluation_args(config, output_dir, model_path)
    
    try:
        # Import and run evaluation functions manually for better control
        from experiments.evaluate_model import (
            load_model_and_artifacts, load_test_data, generate_predictions,
            apply_calibration, evaluate_predictions, generate_plots
        )
        from mind2text.postprocessing import ReportGenerator
        
        # Load model and artifacts
        model, tokenizer, binner, model_config = load_model_and_artifacts(eval_args.model_path)
        
        # Load test data
        token_sequences, labels = load_test_data(eval_args.test_data_path)
        
        # Generate predictions
        predictions, class_names = generate_predictions(
            model, tokenizer, token_sequences, labels, model_config
        )
        
        # Apply calibration
        calibrated_predictions, calibration_params = apply_calibration(
            predictions, eval_args.calibration_method
        )
        
        # Evaluate predictions
        classification_metrics, calibration_metrics = evaluate_predictions(
            calibrated_predictions, class_names
        )
        
        # Create evaluation report
        eval_output_dir = output_dir / "evaluation"
        eval_output_dir.mkdir(exist_ok=True)
        
        report_generator = ReportGenerator()
        report = report_generator.create_evaluation_report(
            run_id=config.experiment.run_name,
            classification_metrics=classification_metrics,
            calibration_metrics=calibration_metrics,
            model_type=config.model.model_type,
            dataset_split="test"
        )
        
        # Save evaluation results
        with open(eval_output_dir / "evaluation_report.json", 'w') as f:
            json.dump(report.model_dump(), f, indent=2)
        
        if eval_args.save_predictions:
            predictions_data = {
                'predictions': [pred.model_dump() for pred in calibrated_predictions]
            }
            with open(eval_output_dir / "predictions.json", 'w') as f:
                json.dump(predictions_data, f, indent=2)
        
        # Generate plots
        if eval_args.generate_plots:
            generate_plots(
                calibrated_predictions, classification_metrics,
                calibration_metrics, eval_output_dir, eval_args.n_bins
            )
        
        logger.info("✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

def print_configuration_summary(config: Mind2TextConfig):
    """Print a summary of the configuration."""
    logger = logging.getLogger(__name__)
    
    logger.info("📋 Experiment Configuration Summary:")
    logger.info(f"   • Experiment: {config.experiment.experiment_name}")
    logger.info(f"   • Run: {config.experiment.run_name}")
    logger.info(f"   • Model: {config.model.model_type}")
    logger.info(f"   • Subjects: {config.data.n_subjects}")
    logger.info(f"   • Tasks: {', '.join(config.data.tasks)}")
    logger.info(f"   • Epochs: {config.model.epochs}")
    logger.info(f"   • Batch Size: {config.model.batch_size}")
    logger.info(f"   • Learning Rate: {config.model.learning_rate}")
    logger.info(f"   • Seed: {config.experiment.seed}")

def main():
    """Main experiment runner."""
    args = parse_arguments()
    
    # Load configuration
    config = load_configuration(args)
    
    # Prepare output directory
    output_dir = prepare_output_directory(config)
    
    # Setup logging
    setup_logging(config.experiment.log_level, output_dir)
    logger = logging.getLogger(__name__)
    
    # Print configuration summary
    print_configuration_summary(config)
    
    if args.dry_run:
        logger.info("🔍 Dry run - configuration saved but no training/evaluation performed")
        return
    
    try:
        if args.mode in ["train", "full"]:
            # Run training
            trained_model_dir = run_training(config, output_dir)
            
            if args.mode == "full":
                # Run evaluation on the trained model
                run_evaluation(config, output_dir)
        
        elif args.mode == "eval":
            # Run evaluation only
            if not args.model_path:
                logger.error("❌ Model path required for evaluation mode")
                sys.exit(1)
            run_evaluation(config, output_dir, args.model_path)
        
        logger.info(f"🎉 Experiment completed! Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"💥 Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
