"""
Sequence Generation for EEG-to-Text Processing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random
import logging

logger = logging.getLogger(__name__)

class SequenceGenerator:
    """
    Generates various types of sequences for EEG-to-text processing,
    including data augmentation and template-based generation.
    """
    
    def __init__(self,
                 augmentation_prob: float = 0.1,
                 noise_level: float = 0.05,
                 random_state: int = 42):
        """
        Initialize sequence generator.
        
        Args:
            augmentation_prob: Probability of applying augmentation
            noise_level: Level of noise for feature augmentation
            random_state: Random seed for reproducibility
        """
        self.augmentation_prob = augmentation_prob
        self.noise_level = noise_level
        self.random_state = random_state
        
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Template explanations for each motor imagery task
        self.explanation_templates = {
            1: [  # Left hand
                "The EEG shows increased beta activity over right motor cortex, indicating left hand motor imagery.",
                "Beta suppression in the right hemisphere suggests preparation for left hand movement.",
                "Motor cortex lateralization pattern consistent with left hand motor imagery task.",
                "Right-sided sensorimotor rhythm changes indicate left hand movement imagination."
            ],
            2: [  # Right hand
                "Beta band suppression over left motor cortex indicates right hand motor imagery.",
                "Left hemisphere activation pattern suggests right hand movement preparation.", 
                "EEG lateralization shows left motor cortex engagement for right hand task.",
                "Sensorimotor rhythm changes in left hemisphere indicate right hand imagery."
            ],
            3: [  # Feet
                "Central motor cortex activation pattern indicates feet motor imagery.",
                "Bilateral sensorimotor changes suggest foot movement imagination.",
                "EEG patterns over central regions consistent with feet motor imagery task.",
                "Motor cortex activity in foot representation area indicates feet movement."
            ],
            4: [  # Tongue
                "Motor cortex activity in tongue representation area indicates tongue imagery.",
                "EEG patterns suggest tongue motor imagery with characteristic central activation.",
                "Sensorimotor rhythm changes consistent with tongue movement imagination.",
                "Central motor cortex patterns indicate tongue motor imagery task."
            ]
        }
        
    def augment_token_sequence(self, tokens: List[str]) -> List[str]:
        """
        Apply augmentation to token sequence.
        
        Args:
            tokens: Original token sequence
            
        Returns:
            Augmented token sequence
        """
        if random.random() > self.augmentation_prob:
            return tokens
            
        augmented_tokens = tokens.copy()
        
        # Random token permutation (within reason)
        if len(augmented_tokens) > 4:
            # Swap adjacent tokens occasionally
            swap_idx = random.randint(0, len(augmented_tokens) - 2)
            augmented_tokens[swap_idx], augmented_tokens[swap_idx + 1] = \
                augmented_tokens[swap_idx + 1], augmented_tokens[swap_idx]
                
        # Token substitution (replace HIGH with VERY_HIGH occasionally)
        for i, token in enumerate(augmented_tokens):
            if '_HIGH' in token and random.random() < 0.1:
                augmented_tokens[i] = token.replace('_HIGH', '_MEDIUM')
            elif '_LOW' in token and random.random() < 0.1:
                augmented_tokens[i] = token.replace('_LOW', '_MEDIUM')
                
        return augmented_tokens
    
    def generate_synthetic_explanations(self,
                                      token_sequences: List[List[str]],
                                      labels: np.ndarray,
                                      n_variations: int = 3) -> List[Dict[str, str]]:
        """
        Generate synthetic explanations for training data.
        
        Args:
            token_sequences: List of EEG token sequences
            labels: Array of class labels
            n_variations: Number of explanation variations per sample
            
        Returns:
            List of explanation examples
        """
        explanations = []
        
        for tokens, label in zip(token_sequences, labels):
            # Generate multiple explanation variations
            base_templates = self.explanation_templates.get(label, ["Generic motor imagery pattern."])
            
            for _ in range(n_variations):
                # Select random template
                template = random.choice(base_templates)
                
                # Add token-specific details
                enhanced_explanation = self._enhance_explanation_with_tokens(template, tokens, label)
                
                explanations.append({
                    'tokens': tokens,
                    'label': label,
                    'explanation': enhanced_explanation,
                    'template': template
                })
                
        return explanations
    
    def _enhance_explanation_with_tokens(self,
                                       template: str,
                                       tokens: List[str],
                                       label: int) -> str:
        """
        Enhance explanation template with specific token information.
        
        Args:
            template: Base explanation template
            tokens: EEG token sequence
            label: Class label
            
        Returns:
            Enhanced explanation
        """
        token_string = ' '.join(tokens).lower()
        enhancements = []
        
        # Add frequency band details
        if 'alpha' in token_string:
            enhancements.append("Alpha rhythm modulation supports the motor imagery task.")
        if 'beta' in token_string:
            enhancements.append("Beta band activity indicates motor cortex engagement.")
        if 'gamma' in token_string:
            enhancements.append("Gamma activity suggests focused attention during imagery.")
            
        # Add spatial details
        if 'c3' in token_string or 'left' in token_string:
            enhancements.append("Left motor cortex involvement is evident.")
        if 'c4' in token_string or 'right' in token_string:
            enhancements.append("Right motor cortex activation is observed.")
        if 'cz' in token_string or 'central' in token_string:
            enhancements.append("Central motor areas show characteristic patterns.")
            
        # Combine template with enhancements
        if enhancements:
            enhanced = template + " " + " ".join(enhancements)
        else:
            enhanced = template
            
        return enhanced
    
    def create_instruction_tuning_data(self,
                                     token_sequences: List[List[str]],
                                     labels: np.ndarray) -> List[Dict[str, str]]:
        """
        Create instruction tuning dataset for LLM training.
        
        Args:
            token_sequences: List of EEG token sequences
            labels: Array of class labels
            
        Returns:
            List of instruction-response pairs
        """
        instructions = []
        
        task_names = {
            1: "left hand motor imagery",
            2: "right hand motor imagery",
            3: "feet motor imagery", 
            4: "tongue motor imagery"
        }
        
        instruction_templates = [
            "Analyze the following EEG motor imagery data and classify the intended movement:",
            "Given these EEG features, determine which motor imagery task was performed:",
            "Classify the motor imagery task based on these EEG patterns:",
            "What type of motor imagery does this EEG data represent?",
            "Examine these EEG features and identify the motor imagery task:"
        ]
        
        for tokens, label in zip(token_sequences, labels):
            # Random instruction template
            instruction_template = random.choice(instruction_templates)
            eeg_data = ' '.join(tokens)
            
            # Create instruction
            instruction = f"{instruction_template}\n\nEEG Data: {eeg_data}"
            
            # Create response
            task_name = task_names.get(label, "unknown")
            response = f"This EEG pattern indicates {task_name}."
            
            # Add explanation
            explanation = random.choice(self.explanation_templates.get(label, ["Generic pattern."]))
            enhanced_explanation = self._enhance_explanation_with_tokens(explanation, tokens, label)
            response += f" {enhanced_explanation}"
            
            instructions.append({
                'instruction': instruction,
                'input': eeg_data,
                'output': response,
                'label': label,
                'task_name': task_name
            })
            
        return instructions
    
    def generate_few_shot_examples(self,
                                 token_sequences: List[List[str]],
                                 labels: np.ndarray,
                                 n_shots: int = 3) -> List[Dict[str, str]]:
        """
        Generate few-shot learning examples.
        
        Args:
            token_sequences: List of EEG token sequences
            labels: Array of class labels
            n_shots: Number of examples per class
            
        Returns:
            List of few-shot examples
        """
        # Group by class
        class_examples = {label: [] for label in np.unique(labels)}
        
        for tokens, label in zip(token_sequences, labels):
            class_examples[label].append(tokens)
            
        # Select few-shot examples
        few_shot_examples = []
        
        for label, examples in class_examples.items():
            selected_examples = random.sample(examples, min(n_shots, len(examples)))
            
            for tokens in selected_examples:
                task_name = {1: "left hand", 2: "right hand", 3: "feet", 4: "tongue"}.get(label, "unknown")
                
                few_shot_examples.append({
                    'tokens': tokens,
                    'label': label,
                    'task_name': task_name,
                    'formatted': f"EEG: {' '.join(tokens)} -> Task: {task_name}"
                })
                
        return few_shot_examples
    
    def create_chain_of_thought_examples(self,
                                       token_sequences: List[List[str]],
                                       labels: np.ndarray) -> List[Dict[str, str]]:
        """
        Create chain-of-thought reasoning examples.
        
        Args:
            token_sequences: List of EEG token sequences
            labels: Array of class labels
            
        Returns:
            List of chain-of-thought examples
        """
        cot_examples = []
        
        for tokens, label in zip(token_sequences, labels):
            # Create step-by-step reasoning
            reasoning_steps = []
            
            # Step 1: Feature analysis
            token_string = ' '.join(tokens).lower()
            reasoning_steps.append("Step 1: Analyze EEG features")
            
            if 'beta' in token_string:
                reasoning_steps.append("- Beta band activity detected in motor cortex")
            if 'alpha' in token_string:
                reasoning_steps.append("- Alpha rhythm modulation observed")
            if 'c3' in token_string or 'left' in token_string:
                reasoning_steps.append("- Left motor cortex activation present")
            if 'c4' in token_string or 'right' in token_string:
                reasoning_steps.append("- Right motor cortex activation present")
                
            # Step 2: Pattern interpretation
            reasoning_steps.append("\nStep 2: Interpret spatial patterns")
            
            if label == 1:  # Left hand
                reasoning_steps.append("- Right hemisphere dominance suggests left hand control")
            elif label == 2:  # Right hand
                reasoning_steps.append("- Left hemisphere dominance suggests right hand control")
            elif label == 3:  # Feet
                reasoning_steps.append("- Central activation suggests feet motor imagery")
            elif label == 4:  # Tongue
                reasoning_steps.append("- Central motor area activity indicates tongue imagery")
                
            # Step 3: Final classification
            task_name = {1: "left hand", 2: "right hand", 3: "feet", 4: "tongue"}.get(label, "unknown")
            reasoning_steps.append(f"\nStep 3: Classification")
            reasoning_steps.append(f"- Pattern consistent with {task_name} motor imagery")
            
            # Create full reasoning chain
            reasoning_chain = '\n'.join(reasoning_steps)
            
            cot_examples.append({
                'tokens': tokens,
                'label': label,
                'reasoning': reasoning_chain,
                'conclusion': f"Therefore, this represents {task_name} motor imagery."
            })
            
        return cot_examples
    
    def generate_contrastive_examples(self,
                                    token_sequences: List[List[str]],
                                    labels: np.ndarray) -> List[Dict[str, str]]:
        """
        Generate contrastive examples showing what patterns distinguish different tasks.
        
        Args:
            token_sequences: List of EEG token sequences
            labels: Array of class labels
            
        Returns:
            List of contrastive examples
        """
        contrastive_examples = []
        
        # Group examples by class
        class_tokens = {label: [] for label in np.unique(labels)}
        for tokens, label in zip(token_sequences, labels):
            class_tokens[label].append(tokens)
            
        # Create contrasts between classes
        task_names = {1: "left hand", 2: "right hand", 3: "feet", 4: "tongue"}
        
        for label1 in class_tokens:
            for label2 in class_tokens:
                if label1 >= label2:  # Avoid duplicates
                    continue
                    
                # Sample examples from each class
                if class_tokens[label1] and class_tokens[label2]:
                    tokens1 = random.choice(class_tokens[label1])
                    tokens2 = random.choice(class_tokens[label2])
                    
                    # Create contrast explanation
                    task1 = task_names.get(label1, "unknown")
                    task2 = task_names.get(label2, "unknown")
                    
                    contrast_text = f"Compare {task1} vs {task2} motor imagery:\n\n"
                    contrast_text += f"{task1.capitalize()}: {' '.join(tokens1)}\n"
                    contrast_text += f"{task2.capitalize()}: {' '.join(tokens2)}\n\n"
                    
                    # Add distinguishing features
                    distinguishing_features = self._find_distinguishing_features(tokens1, tokens2, label1, label2)
                    contrast_text += f"Key differences: {distinguishing_features}"
                    
                    contrastive_examples.append({
                        'task1': task1,
                        'task2': task2,
                        'tokens1': tokens1,
                        'tokens2': tokens2,
                        'contrast_explanation': contrast_text
                    })
                    
        return contrastive_examples
    
    def _find_distinguishing_features(self,
                                    tokens1: List[str],
                                    tokens2: List[str],
                                    label1: int,
                                    label2: int) -> str:
        """
        Find distinguishing features between two token sequences.
        
        Args:
            tokens1: First token sequence
            tokens2: Second token sequence  
            label1: First class label
            label2: Second class label
            
        Returns:
            Description of distinguishing features
        """
        features = []
        
        str1 = ' '.join(tokens1).lower()
        str2 = ' '.join(tokens2).lower()
        
        # Check lateralization differences
        if (label1 in [1, 2] and label2 in [1, 2]):  # Both hand tasks
            if 'left' in str1 and 'right' in str2:
                features.append("hemispheric lateralization (left vs right motor cortex)")
            elif 'right' in str1 and 'left' in str2:
                features.append("hemispheric lateralization (right vs left motor cortex)")
                
        # Check frequency band differences
        bands1 = set([band for band in ['alpha', 'beta', 'gamma', 'theta'] if band in str1])
        bands2 = set([band for band in ['alpha', 'beta', 'gamma', 'theta'] if band in str2])
        
        if bands1 != bands2:
            unique1 = bands1 - bands2
            unique2 = bands2 - bands1
            if unique1:
                features.append(f"unique {', '.join(unique1)} activity in first task")
            if unique2:
                features.append(f"unique {', '.join(unique2)} activity in second task")
                
        # Default if no specific features found
        if not features:
            features.append("distinct spatial and spectral patterns")
            
        return ', '.join(features)
