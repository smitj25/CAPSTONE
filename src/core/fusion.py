import numpy as np
import pickle
import os
import tensorflow as tf
from typing import Tuple, List, Dict, Union

class BotDetectionFusion:
    """
    Fusion module that combines scores from web log and mouse movement detection modules
    to produce a final, robust decision about whether a session is from a human or bot.
    
    This implements the decision-level fusion method described in the research paper:
    - If mouse movement score is very high (>0.7) or very low (<0.3), rely solely on it
    - Otherwise, use weighted average of both scores (0.5 * mouse + 0.5 * web_log)
    """
    
    def __init__(self, 
                 web_log_model_path: str = None,
                 mouse_movement_model_path: str = None,
                 high_threshold: float = 0.65,
                 low_threshold: float = 0.35,
                 final_threshold: float = 0.45,
                 mouse_weight: float = 0.6,
                 web_log_weight: float = 0.4):
        """
        Initialize the fusion module.
        
        Args:
            web_log_model_path: Path to the trained web log detection model
            mouse_movement_model_path: Path to the trained mouse movement detection model
            high_threshold: High threshold for mouse movement score (default: 0.7)
            low_threshold: Low threshold for mouse movement score (default: 0.3)
            final_threshold: Final decision threshold (default: 0.5)
            mouse_weight: Weight for mouse movement score in fusion (default: 0.5)
            web_log_weight: Weight for web log score in fusion (default: 0.5)
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.final_threshold = final_threshold
        self.mouse_weight = mouse_weight
        self.web_log_weight = web_log_weight
        
        # Load models if paths are provided
        self.web_log_model = None
        self.mouse_movement_model = None
        
        if web_log_model_path and os.path.exists(web_log_model_path):
            self.load_web_log_model(web_log_model_path)
            
        if mouse_movement_model_path and os.path.exists(mouse_movement_model_path):
            self.load_mouse_movement_model(mouse_movement_model_path)
    
    def load_web_log_model(self, model_path: str):
        """
        Load the trained web log detection model.
        
        Args:
            model_path: Path to the saved web log model
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.web_log_model = model_data['model']
            self.web_log_scaler = model_data['scaler']
            self.web_log_features = model_data['selected_features']
            print(f"Web log model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading web log model: {e}")
    
    def load_mouse_movement_model(self, model_path: str):
        """
        Load the trained mouse movement detection model.
        
        Args:
            model_path: Path to the saved mouse movement model
        """
        try:
            self.mouse_movement_model = tf.keras.models.load_model(model_path)
            print(f"Mouse movement model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading mouse movement model: {e}")
    
    def predict_web_log_score(self, web_log_data) -> float:
        """
        Predict bot probability using the web log detection model.
        
        Args:
            web_log_data: Web log features for a session
            
        Returns:
            float: Bot probability score (0-1)
        """
        if self.web_log_model is None:
            raise ValueError("Web log model not loaded. Please load the model first.")
        
        try:
            # Select features and scale
            selected_features = web_log_data[self.web_log_features]
            scaled_features = self.web_log_scaler.transform(selected_features)
            
            # Predict probability
            probabilities = self.web_log_model.predict_proba(scaled_features)
            return probabilities[0][1]  # Return bot probability
        except Exception as e:
            print(f"Error predicting web log score: {e}")
            return 0.5  # Return neutral score on error
    
    def predict_mouse_movement_score(self, mouse_movement_matrices: List[np.ndarray]) -> float:
        """
        Predict bot probability using the mouse movement detection model.
        
        Args:
            mouse_movement_matrices: List of mouse movement matrices for a session
            
        Returns:
            float: Bot probability score (0-1)
        """
        if self.mouse_movement_model is None:
            raise ValueError("Mouse movement model not loaded. Please load the model first.")
        
        try:
            if not mouse_movement_matrices:
                return 0.5  # Return neutral score if no matrices
            
            # Convert matrices to the expected format
            matrices = np.array(mouse_movement_matrices)
            
            # Ensure matrices have the correct shape (add channel dimension if needed)
            if len(matrices.shape) == 3:
                matrices = matrices.reshape(matrices.shape[0], matrices.shape[1], matrices.shape[2], 1)
            
            # Predict probabilities for each matrix
            probabilities = self.mouse_movement_model.predict(matrices)
            
            # Perform majority voting across all matrices
            bot_votes = np.sum(probabilities[:, 1] > 0.5)
            total_votes = len(probabilities)
            
            # Return the ratio of bot votes
            return bot_votes / total_votes if total_votes > 0 else 0.5
            
        except Exception as e:
            print(f"Error predicting mouse movement score: {e}")
            return 0.5  # Return neutral score on error
    
    def fuse_scores(self, mouse_score: float, web_log_score: float) -> float:
        """
        Fuse the scores from both detection modules using the decision-level fusion method.
        
        Args:
            mouse_score: Bot probability from mouse movement detection (0-1)
            web_log_score: Bot probability from web log detection (0-1)
            
        Returns:
            float: Final fused bot probability score (0-1)
        """
        # Check if mouse movement score is very high or very low
        if mouse_score > self.high_threshold or mouse_score < self.low_threshold:
            # Rely solely on mouse movement score
            print(f"Mouse score ({mouse_score:.3f}) outside thresholds [{self.low_threshold}, {self.high_threshold}]. Using mouse score only.")
            return mouse_score
        else:
            # Use weighted average of both scores
            fused_score = (self.mouse_weight * mouse_score + 
                          self.web_log_weight * web_log_score)
            print(f"Mouse score ({mouse_score:.3f}) within thresholds. Using weighted average: {fused_score:.3f}")
            return fused_score
    
    def classify_session(self, mouse_score: float, web_log_score: float) -> Tuple[bool, float]:
        """
        Classify a session as human or bot based on fused scores.
        
        Args:
            mouse_score: Bot probability from mouse movement detection (0-1)
            web_log_score: Bot probability from web log detection (0-1)
            
        Returns:
            Tuple[bool, float]: (is_bot, final_score)
                - is_bot: True if classified as bot, False if human
                - final_score: Final fused bot probability score
        """
        # Fuse the scores
        final_score = self.fuse_scores(mouse_score, web_log_score)
        
        # Make final decision
        is_bot = final_score > self.final_threshold
        
        return is_bot, final_score
    
    def process_session(self, 
                       web_log_features=None, 
                       mouse_movement_matrices=None,
                       mouse_score=None,
                       web_log_score=None) -> Dict[str, Union[bool, float]]:
        """
        Process a complete session and return classification results.
        
        Args:
            web_log_features: Web log features for the session (optional if web_log_score provided)
            mouse_movement_matrices: Mouse movement matrices for the session (optional if mouse_score provided)
            mouse_score: Pre-computed mouse movement score (optional)
            web_log_score: Pre-computed web log score (optional)
            
        Returns:
            Dict containing:
                - is_bot: Final classification (True/False)
                - final_score: Final fused score
                - mouse_score: Mouse movement score used
                - web_log_score: Web log score used
                - fusion_method: Method used for fusion ("mouse_only" or "weighted_average")
        """
        # Get scores (either from models or provided directly)
        if mouse_score is None and mouse_movement_matrices is not None:
            mouse_score = self.predict_mouse_movement_score(mouse_movement_matrices)
        elif mouse_score is None:
            mouse_score = 0.5  # Default neutral score
        
        if web_log_score is None and web_log_features is not None:
            web_log_score = self.predict_web_log_score(web_log_features)
        elif web_log_score is None:
            web_log_score = 0.5  # Default neutral score
        
        # Determine fusion method
        if mouse_score > self.high_threshold or mouse_score < self.low_threshold:
            fusion_method = "mouse_only"
        else:
            fusion_method = "weighted_average"
        
        # Fuse scores and classify
        is_bot, final_score = self.classify_session(mouse_score, web_log_score)
        
        return {
            'is_bot': is_bot,
            'final_score': final_score,
            'mouse_score': mouse_score,
            'web_log_score': web_log_score,
            'fusion_method': fusion_method
        }
    
    def evaluate_fusion_performance(self, 
                                  test_sessions: List[Dict],
                                  true_labels: List[bool]) -> Dict[str, float]:
        """
        Evaluate the performance of the fusion system.
        
        Args:
            test_sessions: List of session data dictionaries
            true_labels: List of true labels (True for bot, False for human)
            
        Returns:
            Dict containing performance metrics
        """
        predictions = []
        final_scores = []
        
        for session in test_sessions:
            result = self.process_session(**session)
            predictions.append(result['is_bot'])
            final_scores.append(result['final_score'])
        
        # Calculate metrics
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(true_labels) if true_labels else 0
        
        # Calculate precision, recall, F1-score
        tp = sum(1 for pred, true in zip(predictions, true_labels) if pred and true)
        fp = sum(1 for pred, true in zip(predictions, true_labels) if pred and not true)
        fn = sum(1 for pred, true in zip(predictions, true_labels) if not pred and true)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_sessions': len(true_labels)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize fusion module
    fusion = BotDetectionFusion(
        web_log_model_path='models/web_log_detector_comprehensive.pkl',
        mouse_movement_model_path='models/mouse_movement_detector_comprehensive.h5'
    )
    
    # Example session processing
    print("=== Bot Detection Fusion System ===")
    
    # Example 1: High confidence mouse movement score
    print("\n--- Example 1: High confidence mouse movement ---")
    result1 = fusion.process_session(
        mouse_score=0.85,  # Very high bot probability from mouse movements
        web_log_score=0.45  # Moderate bot probability from web logs
    )
    print(f"Result: {result1}")
    
    # Example 2: Low confidence mouse movement score
    print("\n--- Example 2: Low confidence mouse movement ---")
    result2 = fusion.process_session(
        mouse_score=0.15,  # Very low bot probability from mouse movements
        web_log_score=0.65  # High bot probability from web logs
    )
    print(f"Result: {result2}")
    
    # Example 3: Moderate mouse movement score (use weighted average)
    print("\n--- Example 3: Moderate mouse movement (weighted average) ---")
    result3 = fusion.process_session(
        mouse_score=0.55,  # Moderate bot probability from mouse movements
        web_log_score=0.60  # Moderate bot probability from web logs
    )
    print(f"Result: {result3}")
    
    print("\n=== Fusion System Ready ===")