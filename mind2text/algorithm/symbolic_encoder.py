"""
Symbolic encoding of EEG features for LLM processing.

Converts binned EEG features into meaningful token sequences that LLMs can process.
Returns Pydantic entities for type safety.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging

from ..entities.common import FeatureVector, TokenSequence
from .binning import FeatureBinner

LOGGER = logging.getLogger(__name__)
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SymbolicEncoder:
    """
    Encodes EEG token sequences into symbolic representations suitable for LLM input.
    Handles sequence formatting, special tokens, and prompt generation.
    """
    
    def __init__(self,
                 max_sequence_length: int = 512,
                 include_special_tokens: bool = True,
                 task_descriptions: Optional[Dict[int, str]] = None):
        """
        Initialize symbolic encoder.
        
        Args:
            max_sequence_length: Maximum length of encoded sequences
            include_special_tokens: Whether to include special tokens
            task_descriptions: Descriptions for motor imagery tasks
        """
        self.max_sequence_length = max_sequence_length
        self.include_special_tokens = include_special_tokens
        
        # Default task descriptions
        if task_descriptions is None:
            self.task_descriptions = {
                1: "left hand motor imagery",
                2: "right hand motor imagery", 
                3: "feet motor imagery",
                4: "tongue motor imagery"
            }
        else:
            self.task_descriptions = task_descriptions
            
        # Special tokens
        self.special_tokens = {
            'PAD': '<PAD>',
            'UNK': '<UNK>',
            'CLS': '<CLS>',
            'SEP': '<SEP>',
            'BOS': '<BOS>',  # Beginning of sequence
            'EOS': '<EOS>',  # End of sequence
            'MASK': '<MASK>'
        }
        
        # Task-specific tokens
        self.task_tokens = {
            1: '<LEFT_HAND>',
            2: '<RIGHT_HAND>',
            3: '<FEET>',
            4: '<TONGUE>'
        }
        
    def encode_classification_sequence(self, 
                                     tokens: List[str],
                                     label: Optional[int] = None) -> str:
        """
        Encode token sequence for classification task.
        
        Args:
            tokens: List of EEG tokens
            label: Optional class label for training
            
        Returns:
            Formatted sequence string
        """
        sequence_parts = []
        
        # Add beginning of sequence token
        if self.include_special_tokens:
            sequence_parts.append(self.special_tokens['BOS'])
            
        # Add classification prompt
        sequence_parts.append("Classify EEG motor imagery:")
        
        # Add EEG tokens
        eeg_tokens = ' '.join(tokens[:self.max_sequence_length - 10])  # Reserve space for special tokens
        sequence_parts.append(eeg_tokens)
        
        # Add separator
        if self.include_special_tokens:
            sequence_parts.append(self.special_tokens['SEP'])
            
        # Add label for training
        if label is not None:
            task_name = self.task_descriptions.get(label, "unknown")
            sequence_parts.append(f"Prediction: {task_name}")
            
        # Add end of sequence token
        if self.include_special_tokens:
            sequence_parts.append(self.special_tokens['EOS'])
            
        return ' '.join(sequence_parts)
    
    def encode_explanation_sequence(self,
                                  tokens: List[str],
                                  prediction: int,
                                  confidence: Optional[float] = None) -> str:
        """
        Encode sequence for explanation generation.
        
        Args:
            tokens: List of EEG tokens
            prediction: Predicted class
            confidence: Optional prediction confidence
            
        Returns:
            Formatted sequence for explanation
        """
        sequence_parts = []
        
        # Add beginning of sequence
        if self.include_special_tokens:
            sequence_parts.append(self.special_tokens['BOS'])
            
        # Add explanation prompt
        task_name = self.task_descriptions.get(prediction, "unknown")
        sequence_parts.append(f"Explain {task_name} prediction:")
        
        # Add EEG tokens
        eeg_tokens = ' '.join(tokens)
        sequence_parts.append(eeg_tokens)
        
        # Add separator
        if self.include_special_tokens:
            sequence_parts.append(self.special_tokens['SEP'])
            
        # Start explanation
        explanation_start = f"Analysis: The EEG pattern indicates {task_name} because"
        if confidence is not None:
            explanation_start += f" (confidence: {confidence:.2f})"
            
        sequence_parts.append(explanation_start)
        
        return ' '.join(sequence_parts)
    
    def create_training_pairs(self,
                            token_sequences: List[List[str]], 
                            labels: np.ndarray,
                            include_explanations: bool = True) -> List[Dict[str, str]]:
        """
        Create training input-output pairs for LLM fine-tuning.
        
        Args:
            token_sequences: List of EEG token sequences
            labels: Array of class labels
            include_explanations: Whether to include explanation examples
            
        Returns:
            List of training examples
        """
        training_pairs = []
        
        for tokens, label in zip(token_sequences, labels):
            # Classification example
            input_seq = self.encode_classification_sequence(tokens)
            task_name = self.task_descriptions.get(label, "unknown")
            output_seq = f"Prediction: {task_name}"
            
            training_pairs.append({
                'input': input_seq,
                'output': output_seq,
                'task': 'classification',
                'label': label
            })
            
            # Explanation example (if requested)
            if include_explanations:
                explanation_input = self.encode_explanation_sequence(tokens, label)
                explanation_output = self._generate_explanation_template(tokens, label)
                
                training_pairs.append({
                    'input': explanation_input,
                    'output': explanation_output,
                    'task': 'explanation',
                    'label': label
                })
                
        return training_pairs
    
    def _generate_explanation_template(self, 
                                     tokens: List[str], 
                                     label: int) -> str:
        """
        Generate explanation template based on EEG patterns.
        
        Args:
            tokens: EEG token sequence
            label: True class label
            
        Returns:
            Explanation text
        """
        task_name = self.task_descriptions.get(label, "unknown")
        
        # Analyze tokens for key patterns
        explanation_parts = []
        
        # Check for relevant frequency bands
        token_string = ' '.join(tokens)
        
        if 'beta' in token_string.lower():
            if label in [1, 2]:  # Hand movements
                explanation_parts.append("beta band suppression in motor cortex indicates hand movement preparation")
            else:
                explanation_parts.append("beta band activity suggests motor planning")
                
        if 'alpha' in token_string.lower():
            explanation_parts.append("alpha rhythm modulation shows attention to motor task")
            
        # Check for lateralization
        if 'left' in token_string.lower() and label == 2:  # Right hand
            explanation_parts.append("left hemisphere motor cortex activation for right hand control")
        elif 'right' in token_string.lower() and label == 1:  # Left hand
            explanation_parts.append("right hemisphere motor cortex activation for left hand control")
            
        # Default explanation if no specific patterns found
        if not explanation_parts:
            explanation_parts.append(f"EEG patterns consistent with {task_name} motor imagery")
            
        explanation = ". ".join(explanation_parts).capitalize() + "."
        
        return explanation
    
    def encode_inference_sequence(self, tokens: List[str]) -> str:
        """
        Encode sequence for inference (no labels).
        
        Args:
            tokens: List of EEG tokens
            
        Returns:
            Formatted sequence for inference
        """
        sequence_parts = []
        
        # Add beginning of sequence
        if self.include_special_tokens:
            sequence_parts.append(self.special_tokens['BOS'])
            
        # Add classification prompt
        sequence_parts.append("Classify EEG motor imagery:")
        
        # Add EEG tokens
        eeg_tokens = ' '.join(tokens[:self.max_sequence_length - 10])
        sequence_parts.append(eeg_tokens)
        
        # Add separator
        if self.include_special_tokens:
            sequence_parts.append(self.special_tokens['SEP'])
            
        # Add prompt for prediction
        sequence_parts.append("Prediction:")
        
        return ' '.join(sequence_parts)
    
    def create_conversation_format(self,
                                 tokens: List[str],
                                 label: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Create conversation format for chat-based LLMs.
        
        Args:
            tokens: EEG token sequence
            label: Optional true label
            
        Returns:
            List of conversation messages
        """
        eeg_data = ' '.join(tokens)
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert in EEG analysis and motor imagery classification. "
                          "Analyze the provided EEG features and classify the motor imagery task."
            },
            {
                "role": "user", 
                "content": f"Please classify this EEG motor imagery pattern: {eeg_data}"
            }
        ]
        
        if label is not None:
            task_name = self.task_descriptions.get(label, "unknown")
            messages.append({
                "role": "assistant",
                "content": f"Based on the EEG patterns, this represents {task_name}. "
                          f"{self._generate_explanation_template(tokens, label)}"
            })
            
        return messages
    
    def batch_encode(self,
                    token_sequences: List[List[str]],
                    labels: Optional[np.ndarray] = None,
                    format_type: str = 'classification') -> List[str]:
        """
        Batch encode multiple sequences.
        
        Args:
            token_sequences: List of token sequences
            labels: Optional labels array
            format_type: Type of encoding ('classification', 'explanation', 'inference')
            
        Returns:
            List of encoded sequences
        """
        encoded_sequences = []
        
        for i, tokens in enumerate(token_sequences):
            label = labels[i] if labels is not None else None
            
            if format_type == 'classification':
                encoded = self.encode_classification_sequence(tokens, label)
            elif format_type == 'explanation':
                if label is not None:
                    encoded = self.encode_explanation_sequence(tokens, label)
                else:
                    encoded = self.encode_inference_sequence(tokens)
            elif format_type == 'inference':
                encoded = self.encode_inference_sequence(tokens)
            else:
                raise ValueError(f"Unknown format_type: {format_type}")
                
            encoded_sequences.append(encoded)
            
        return encoded_sequences
    
    def get_special_tokens_dict(self) -> Dict[str, str]:
        """
        Get dictionary of all special tokens.
        
        Returns:
            Dictionary mapping token names to token strings
        """
        all_tokens = {}
        all_tokens.update(self.special_tokens)
        all_tokens.update(self.task_tokens)
        
        return all_tokens
