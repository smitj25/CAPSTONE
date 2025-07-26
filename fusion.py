import os
import json
from web_log_detection_bot import WebLogDetectionBot
from mouse_movements_detection_bot import MouseMovementDetectionBot

class BotDetectionFusion:
    def __init__(self, dataset_type='D1', thres_high=0.7, thres_low=0.3, weight_mv=0.5, weight_wl=0.5):
        """
        Initialize the Bot Detection Fusion module with the specified parameters.
        
        Args:
            dataset_type (str): 'D1' or 'D2' to specify which dataset parameters to use for web log detection
            thres_high (float): Upper threshold for mouse movement score (default: 0.7)
            thres_low (float): Lower threshold for mouse movement score (default: 0.3)
            weight_mv (float): Weight for mouse movement score in fusion (default: 0.5)
            weight_wl (float): Weight for web log score in fusion (default: 0.5)
        """
        self.dataset_type = dataset_type
        self.thres_high = thres_high
        self.thres_low = thres_low
        self.weight_mv = weight_mv
        self.weight_wl = weight_wl
        
        # Ensure weights sum to 1
        total_weight = self.weight_mv + self.weight_wl
        if total_weight != 1.0:
            self.weight_mv = self.weight_mv / total_weight
            self.weight_wl = self.weight_wl / total_weight
        
        # Initialize the detection modules
        self.web_log_detector = WebLogDetectionBot(dataset_type=dataset_type)
        self.mouse_movement_detector = MouseMovementDetectionBot()
    
    def fuse_scores(self, score_mv, score_wl):
        """
        Fuse the scores from the mouse movement and web log detection modules.
        
        The fusion mechanism prioritizes the mouse movement detection module:
        - If score_mv is either very high (≥thres_high) or very low (≤thres_low), 
          only this score is considered for the final decision.
        - Otherwise, a weighted average of the two modules' scores is used.
        
        Args:
            score_mv (float): Score from the mouse movement detection module (0-1)
            score_wl (float): Score from the web log detection module (0-1)
            
        Returns:
            float: Fused score between 0 and 1 (0 = human, 1 = bot)
        """
        # Prioritize mouse movement score if it's very high or very low
        if score_mv >= self.thres_high or score_mv <= self.thres_low:
            return score_mv
        
        # Otherwise, use weighted average
        return (self.weight_mv * score_mv) + (self.weight_wl * score_wl)
    
    def classify(self, score_tot, threshold=0.5):
        """
        Classify a session as bot or human based on the total score.
        
        Args:
            score_tot (float): Total fused score
            threshold (float): Classification threshold (default: 0.5)
            
        Returns:
            str: 'bot' if score_tot ≥ threshold, 'human' otherwise
        """
        return 'bot' if score_tot >= threshold else 'human'
    
    def process_session(self, session_id, web_log_path, mouse_data_dir, phase='phase1'):
        """
        Process a session using both detection modules and fuse their scores.
        
        Args:
            session_id (str): PHP session ID
            web_log_path (str): Path to the Apache2 log file
            mouse_data_dir (str): Directory containing mouse movement data
            phase (str): 'phase1' or 'phase2' to specify which dataset format to use
            
        Returns:
            dict: Dictionary containing individual scores, fused score, and classification
        """
        # Get web log score
        web_log_scores = self.web_log_detector.get_web_log_score(web_log_path)
        score_wl = web_log_scores.get(session_id, 0.5)  # Default to neutral if not found
        
        # Get mouse movement score
        score_mv = self.mouse_movement_detector.process_session_data(session_id, mouse_data_dir, phase)
        
        # Fuse scores
        score_tot = self.fuse_scores(score_mv, score_wl)
        
        # Classify
        classification = self.classify(score_tot)
        
        return {
            'session_id': session_id,
            'score_wl': score_wl,
            'score_mv': score_mv,
            'score_tot': score_tot,
            'classification': classification
        }
    
    def evaluate_dataset(self, dataset_dir, phase='phase1', ground_truth_path=None):
        """
        Evaluate the fusion model on a dataset.
        
        Args:
            dataset_dir (str): Directory containing the dataset
            phase (str): 'phase1' or 'phase2' to specify which dataset format to use
            ground_truth_path (str): Path to ground truth labels (optional)
            
        Returns:
            dict: Dictionary containing evaluation metrics if ground truth is provided,
                  otherwise just the classification results
        """
        results = []
        web_log_path = os.path.join(dataset_dir, 'web_logs.log')  # Adjust path as needed
        
        # Get all session directories
        session_ids = [d for d in os.listdir(dataset_dir) 
                      if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
        
        for session_id in session_ids:
            result = self.process_session(session_id, web_log_path, dataset_dir, phase)
            results.append(result)
        
        # If ground truth is provided, calculate metrics
        if ground_truth_path and os.path.exists(ground_truth_path):
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
            
            # Calculate metrics
            tp = fp = tn = fn = 0
            for result in results:
                session_id = result['session_id']
                predicted = result['classification']
                actual = ground_truth.get(session_id)
                
                if actual == 'bot' and predicted == 'bot':
                    tp += 1
                elif actual == 'human' and predicted == 'bot':
                    fp += 1
                elif actual == 'human' and predicted == 'human':
                    tn += 1
                elif actual == 'bot' and predicted == 'human':
                    fn += 1
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision_bot = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_bot = tp / (tp + fn) if (tp + fn) > 0 else 0
            f_score_bot = 2 * (precision_bot * recall_bot) / (precision_bot + recall_bot) if (precision_bot + recall_bot) > 0 else 0
            
            precision_human = tn / (tn + fn) if (tn + fn) > 0 else 0
            recall_human = tn / (tn + fp) if (tn + fp) > 0 else 0
            f_score_human = 2 * (precision_human * recall_human) / (precision_human + recall_human) if (precision_human + recall_human) > 0 else 0
            
            return {
                'results': results,
                'metrics': {
                    'accuracy': accuracy,
                    'bot': {
                        'precision': precision_bot,
                        'recall': recall_bot,
                        'f_score': f_score_bot
                    },
                    'human': {
                        'precision': precision_human,
                        'recall': recall_human,
                        'f_score': f_score_human
                    }
                }
            }
        
        return {'results': results}

# Example usage
if __name__ == "__main__":
    # Initialize the fusion module with default parameters
    fusion = BotDetectionFusion(
        dataset_type='D1',  # Use D1 dataset parameters for web log detection
        thres_high=0.7,     # Upper threshold for mouse movement score
        thres_low=0.3,      # Lower threshold for mouse movement score
        weight_mv=0.5,      # Weight for mouse movement score
        weight_wl=0.5       # Weight for web log score
    )
    
    # Example: Process a single session
    # result = fusion.process_session(
    #     session_id='example_session_id',
    #     web_log_path='path_to_web_logs.log',
    #     mouse_data_dir='path_to_mouse_data_directory',
    #     phase='phase1'
    # )
    # print(f"Session {result['session_id']} classification: {result['classification']}")
    # print(f"Web log score: {result['score_wl']}, Mouse movement score: {result['score_mv']}, Fused score: {result['score_tot']}")
    
    # Example: Evaluate on a dataset
    # evaluation = fusion.evaluate_dataset(
    #     dataset_dir='path_to_dataset_directory',
    #     phase='phase1',
    #     ground_truth_path='path_to_ground_truth.json'
    # )
    # print(f"Accuracy: {evaluation['metrics']['accuracy']}")
    # print(f"Bot detection - Precision: {evaluation['metrics']['bot']['precision']}, Recall: {evaluation['metrics']['bot']['recall']}, F-score: {evaluation['metrics']['bot']['f_score']}")
    # print(f"Human detection - Precision: {evaluation['metrics']['human']['precision']}, Recall: {evaluation['metrics']['human']['recall']}, F-score: {evaluation['metrics']['human']['f_score']}")