#!/usr/bin/env python3
"""
Main Bot Detection System

This script demonstrates the complete bot detection system working with all three components:
1. Web Log Detection (web_logs.py)
2. Mouse Movement Detection (mouse_movements.py)  
3. Fusion (fusion.py)

The system follows the research paper's approach:
- Train separate models for web logs and mouse movements
- Use decision-level fusion to combine scores
- Provide robust bot detection that's harder to evade
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List
import importlib.util

# Load core modules directly from file paths to avoid import path issues
CORE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))

def _load_module(name: str, filename: str):
    module_path = os.path.join(CORE_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_web_log_mod = _load_module('web_log_detection_bot', 'web_logs.py')
_mouse_mod = _load_module('mouse_movements_detection_bot', 'mouse_movements.py')
_fusion_mod = _load_module('fusion', 'fusion.py')

WebLogDetectionBot = _web_log_mod.WebLogDetectionBot
MouseMovementDetectionBot = _mouse_mod.MouseMovementDetectionBot
BotDetectionFusion = _fusion_mod.BotDetectionFusion

class BotDetectionSystem:
    """
    Comprehensive bot detection system that combines web log and mouse movement analysis
    with decision-level fusion for robust bot detection.
    """
    
    def __init__(self, 
                 dataset_base_path: str = None,
                 web_log_model_path: str = None,
                 mouse_movement_model_path: str = None):
        """
        Initialize the comprehensive bot detection system.
        
        Args:
            dataset_base_path: Base path to the dataset directory
            web_log_model_path: Path to save/load the web log detection model
            mouse_movement_model_path: Path to save/load the mouse movement detection model
        """
        # Resolve repository root and default paths
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        self.dataset_base_path = dataset_base_path or os.path.join(repo_root, 'data')
        self.web_log_model_path = web_log_model_path or os.path.join(repo_root, 'models', 'logs_model.pkl')
        self.mouse_movement_model_path = mouse_movement_model_path or os.path.join(repo_root, 'models', 'mouse_model.h5')
        
        # Initialize detection modules
        self.web_log_detector = WebLogDetectionBot()
        self.mouse_movement_detector = MouseMovementDetectionBot()
        self.fusion = BotDetectionFusion()
        
        # Training status
        self.web_log_trained = False
        self.mouse_movement_trained = False
        
        # Load pre-trained models if they exist
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing trained models if available."""
        if os.path.exists(self.web_log_model_path):
            try:
                self.web_log_detector.load_model(self.web_log_model_path)
                self.web_log_trained = True
                print(f"Loaded existing web log model from {self.web_log_model_path}")
            except Exception as e:
                print(f"Error loading web log model: {e}")
        
        if os.path.exists(self.mouse_movement_model_path):
            try:
                self.mouse_movement_detector.load_model(self.mouse_movement_model_path)
                self.mouse_movement_trained = True
                print(f"Loaded existing mouse movement model from {self.mouse_movement_model_path}")
            except Exception as e:
                print(f"Error loading mouse movement model: {e}")
    
    def train_web_log_detector(self):
        """Training requires prepared datasets. Not implemented in this runner."""
        raise RuntimeError(
            "Web log model not found. Please place pre-trained model at "
            f"{self.web_log_model_path} or provide a training script with datasets."
        )
    
    def train_mouse_movement_detector(self):
        """Training requires prepared datasets. Not implemented in this runner."""
        raise RuntimeError(
            "Mouse movement model not found. Please place pre-trained model at "
            f"{self.mouse_movement_model_path} or provide a training script with datasets."
        )
    
    def train_all_models(self):
        """Train both detection models."""
        print("=== COMPREHENSIVE BOT DETECTION SYSTEM MODEL LOADING ===")
        
        # Require pre-trained models; provide clear guidance if missing
        if not self.web_log_trained:
            raise RuntimeError(
                "Missing web log model. Copy 'logs_model.pkl' to "
                f"{self.web_log_model_path}"
            )
        
        if not self.mouse_movement_trained:
            raise RuntimeError(
                "Missing mouse movement model. Copy 'mouse_model.h5' to "
                f"{self.mouse_movement_model_path}"
            )
        
        # Initialize fusion with trained models
        self.fusion = BotDetectionFusion(
            web_log_model_path=self.web_log_model_path,
            mouse_movement_model_path=self.mouse_movement_model_path
        )
        
        print("\n=== ALL MODELS TRAINED AND READY ===")
    
    def detect_bot_session(self, 
                          session_id: str,
                          web_log_features: pd.DataFrame = None,
                          mouse_movement_matrices: List[np.ndarray] = None,
                          web_log_score: float = None,
                          mouse_score: float = None) -> Dict:
        """
        Detect if a session is from a bot using the comprehensive system.
        
        Args:
            session_id: Unique identifier for the session
            web_log_features: Web log features for the session
            mouse_movement_matrices: Mouse movement matrices for the session
            web_log_score: Pre-computed web log score (optional)
            mouse_score: Pre-computed mouse movement score (optional)
            
        Returns:
            Dict containing detection results
        """
        if not self.web_log_trained or not self.mouse_movement_trained:
            raise ValueError("Both detection models must be trained before detection.")
        
        print(f"\n--- Detecting Bot for Session: {session_id} ---")
        
        # Get individual scores
        if web_log_score is None and web_log_features is not None:
            web_log_score = self.web_log_detector.predict(web_log_features)[0][1]
            print(f"Web Log Score: {web_log_score:.3f}")
        
        if mouse_score is None and mouse_movement_matrices is not None:
            mouse_score = self.mouse_movement_detector.predict_session(mouse_movement_matrices)
            print(f"Mouse Movement Score: {mouse_score:.3f}")
        
        # Use fusion to get final result
        result = self.fusion.process_session(
            web_log_features=web_log_features,
            mouse_movement_matrices=mouse_movement_matrices,
            mouse_score=mouse_score,
            web_log_score=web_log_score
        )
        
        result['session_id'] = session_id
        return result
    
    def evaluate_system_performance(self, test_sessions: List[Dict], true_labels: List[bool]) -> Dict:
        """
        Evaluate the performance of the complete system.
        
        Args:
            test_sessions: List of test session data
            true_labels: List of true labels (True for bot, False for human)
            
        Returns:
            Dict containing performance metrics
        """
        print("\n" + "="*60)
        print("EVALUATING SYSTEM PERFORMANCE")
        print("="*60)
        
        results = []
        for session_data in test_sessions:
            result = self.detect_bot_session(**session_data)
            results.append(result)
        
        # Calculate metrics
        predictions = [r['is_bot'] for r in results]
        final_scores = [r['final_score'] for r in results]
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(true_labels) if true_labels else 0
        
        # Calculate precision, recall, F1-score
        tp = sum(1 for pred, true in zip(predictions, true_labels) if pred and true)
        fp = sum(1 for pred, true in zip(predictions, true_labels) if pred and not true)
        fn = sum(1 for pred, true in zip(predictions, true_labels) if not pred and true)
        tn = sum(1 for pred, true in zip(predictions, true_labels) if not pred and not true)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate individual module performance
        web_log_scores = [r['web_log_score'] for r in results]
        mouse_scores = [r['mouse_score'] for r in results]
        
        web_log_predictions = [score > 0.5 for score in web_log_scores]
        mouse_predictions = [score > 0.5 for score in mouse_scores]
        
        web_log_correct = sum(1 for pred, true in zip(web_log_predictions, true_labels) if pred == true)
        mouse_correct = sum(1 for pred, true in zip(mouse_predictions, true_labels) if pred == true)
        
        web_log_accuracy = web_log_correct / len(true_labels) if true_labels else 0
        mouse_accuracy = mouse_correct / len(true_labels) if true_labels else 0
        
        return {
            'fusion': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            },
            'web_log_only': {
                'accuracy': web_log_accuracy
            },
            'mouse_only': {
                'accuracy': mouse_accuracy
            },
            'total_sessions': len(true_labels),
            'detailed_results': results
        }
    
    def demonstrate_fusion_logic(self):
        """Demonstrate the fusion logic with example scenarios."""
        print("\n" + "="*60)
        print("FUSION LOGIC DEMONSTRATION")
        print("="*60)
        
        # Initialize fusion with trained models
        if self.web_log_trained and self.mouse_movement_trained:
            self.fusion = BotDetectionFusion(
                web_log_model_path=self.web_log_model_path,
                mouse_movement_model_path=self.mouse_movement_model_path
            )
        
        # Example scenarios
        scenarios = [
            {
                'name': 'High Confidence Mouse Movement (Bot)',
                'mouse_score': 0.85,
                'web_log_score': 0.45,
                'expected_fusion': 'mouse_only'
            },
            {
                'name': 'Low Confidence Mouse Movement (Human)',
                'mouse_score': 0.15,
                'web_log_score': 0.65,
                'expected_fusion': 'mouse_only'
            },
            {
                'name': 'Moderate Mouse Movement (Weighted Average)',
                'mouse_score': 0.55,
                'web_log_score': 0.60,
                'expected_fusion': 'weighted_average'
            },
            {
                'name': 'Conflicting Signals (Weighted Average)',
                'mouse_score': 0.40,
                'web_log_score': 0.70,
                'expected_fusion': 'weighted_average'
            }
        ]
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            result = self.fusion.process_session(
                mouse_score=scenario['mouse_score'],
                web_log_score=scenario['web_log_score']
            )
            print(f"Mouse Score: {scenario['mouse_score']:.3f}")
            print(f"Web Log Score: {scenario['web_log_score']:.3f}")
            print(f"Fusion Method: {result['fusion_method']}")
            print(f"Final Score: {result['final_score']:.3f}")
            print(f"Classification: {'BOT' if result['is_bot'] else 'HUMAN'}")
            print(f"Expected Fusion: {scenario['expected_fusion']}")

def main():
    """Main function to demonstrate the complete bot detection system."""
    print("=== COMPREHENSIVE BOT DETECTION SYSTEM ===")
    print("This system implements the research paper's approach:")
    print("1. Web Log Detection (Ensemble Classifier)")
    print("2. Mouse Movement Detection (CNN)")
    print("3. Decision-Level Fusion")
    print("="*60)
    
    # Initialize the system
    system = BotDetectionSystem()
    
    # Load pre-trained models (expects files under /models)
    system.train_all_models()
    
    # Demonstrate fusion logic
    system.demonstrate_fusion_logic()
    
    # Example session detection
    print("\n" + "="*60)
    print("EXAMPLE SESSION DETECTION")
    print("="*60)
    
    # Example 1: Likely bot session
    print("\n--- Example 1: Likely Bot Session ---")
    bot_result = system.detect_bot_session(
        session_id="example_bot_001",
        mouse_score=0.85,  # High bot probability from mouse movements
        web_log_score=0.75  # High bot probability from web logs
    )
    print(f"Result: {bot_result}")
    
    # Example 2: Likely human session
    print("\n--- Example 2: Likely Human Session ---")
    human_result = system.detect_bot_session(
        session_id="example_human_001",
        mouse_score=0.15,  # Low bot probability from mouse movements
        web_log_score=0.25  # Low bot probability from web logs
    )
    print(f"Result: {human_result}")
    
    # Example 3: Ambiguous session
    print("\n--- Example 3: Ambiguous Session ---")
    ambiguous_result = system.detect_bot_session(
        session_id="example_ambiguous_001",
        mouse_score=0.45,  # Moderate bot probability from mouse movements
        web_log_score=0.55  # Moderate bot probability from web logs
    )
    print(f"Result: {ambiguous_result}")
    
    print("\n" + "="*60)
    print("SYSTEM READY FOR PRODUCTION USE")
    print("="*60)
    print("The comprehensive bot detection system is now ready to:")
    print("1. Process web server logs for session analysis")
    print("2. Analyze mouse movement patterns")
    print("3. Combine both signals using intelligent fusion")
    print("4. Provide robust bot detection that's harder to evade")
    print("="*60)

if __name__ == "__main__":
    main()
