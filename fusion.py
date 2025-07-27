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
            mouse_data_dir (str): Directory containing the dataset (should point to phase1 folder)
            phase (str): 'phase1' or 'phase2' to specify which dataset format to use
            
        Returns:
            dict: Dictionary containing individual scores, fused score, and classification
        """
        # Get web log score
        web_log_scores = self.web_log_detector.get_web_log_score(web_log_path)
        score_wl = web_log_scores.get(session_id, 0.5)  # Default to neutral if not found
        
        # Get mouse movement score (use the same dataset_type as the fusion module)
        score_mv = self.mouse_movement_detector.process_session_data(
            session_id, mouse_data_dir, phase, dataset_type=self.dataset_type
        )
        
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
            dataset_dir (str): Directory containing the dataset (should be the phase1 or phase2 folder)
            phase (str): 'phase1' or 'phase2' to specify which dataset format to use
            ground_truth_path (str): Path to the ground truth annotations file (optional)
            
        Returns:
            dict: Dictionary containing evaluation metrics if ground truth is provided,
                  otherwise just the classification results
        """
        results = {}
        
        # Load ground truth if provided
        ground_truth = {}
        if ground_truth_path and os.path.exists(ground_truth_path):
            with open(ground_truth_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        session_id = parts[0]
                        label = parts[1]
                        # Convert labels to binary: 1 for any bot type, 0 for human
                        ground_truth[session_id] = 1 if 'bot' in label.lower() else 0
        
        # Process each session
        if phase == 'phase1':
            # Set up directory paths
            mouse_specific_dir = os.path.join(dataset_dir, self.dataset_type, 'data', 'mouse_movements', 'humans_and_moderate_bots')
            web_logs_dir = os.path.join(dataset_dir, self.dataset_type, 'data', 'web_logs')
            human_logs_dir = os.path.join(web_logs_dir, 'humans')
            
            # Get all session IDs from mouse movement data
            session_ids = set()
            if os.path.exists(mouse_specific_dir):
                for item in os.listdir(mouse_specific_dir):
                    item_path = os.path.join(mouse_specific_dir, item)
                    if os.path.isdir(item_path):
                        session_ids.add(item)
            
            # Process each session
            for session_id in session_ids:
                # Find the appropriate web log file
                # Determine which log file to use based on session annotations
                web_log_path = None
                if os.path.exists(web_logs_dir):
                    # Check both human and bot log files
                  for i in range(1, 6):  # access_1.log to access_5.log
                    log_file = f'access_{i}.log'
                    human_log = os.path.join(human_logs_dir, log_file)
                    bot_log = os.path.join(web_logs_dir, 'bots', 'access_moderate_bots.log')
                    
                    # Try to find the session in both log files
                    if os.path.exists(human_log):
                        web_log_path = human_log
                    elif os.path.exists(bot_log):
                        web_log_path = bot_log
                
                if web_log_path:
                    session_result = self.process_session(
                        session_id, web_log_path, dataset_dir, phase
                    )
                    results[session_id] = session_result
        
        # Calculate metrics if ground truth is available
        if ground_truth:
            tp = fp = tn = fn = 0
            
            for session_id, result in results.items():
                if session_id in ground_truth:
                    predicted = 1 if result['classification'] == 'bot' else 0
                    actual = ground_truth[session_id]
                    
                    if predicted == 1 and actual == 1:
                        tp += 1
                    elif predicted == 1 and actual == 0:
                        fp += 1
                    elif predicted == 0 and actual == 0:
                        tn += 1
                    elif predicted == 0 and actual == 1:
                        fn += 1
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            bot_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            bot_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            bot_f_score = 2 * (bot_precision * bot_recall) / (bot_precision + bot_recall) if (bot_precision + bot_recall) > 0 else 0
            
            human_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
            human_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            human_f_score = 2 * (human_precision * human_recall) / (human_precision + human_recall) if (human_precision + human_recall) > 0 else 0
            
            return {
                'results': results,
                'metrics': {
                    'accuracy': accuracy,
                    'bot': {
                        'precision': bot_precision,
                        'recall': bot_recall,
                        'f_score': bot_f_score
                    },
                    'human': {
                        'precision': human_precision,
                        'recall': human_recall,
                        'f_score': human_f_score
                    }
                }
            }
        
        return {'results': results}

# Example usage
if __name__ == "__main__":
    # Dataset paths
    dataset_base = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase1'
    
    # Initialize the fusion module for D1 (humans vs moderate bots)
    fusion_d1 = BotDetectionFusion(
        dataset_type='D1',  # Use D1 dataset parameters for web log detection
        thres_high=0.7,     # Upper threshold for mouse movement score
        thres_low=0.3,      # Lower threshold for mouse movement score
        weight_mv=0.5,      # Weight for mouse movement score
        weight_wl=0.5       # Weight for web log score
    )
    
    # Initialize the fusion module for D2 (humans vs advanced bots)
    '''
    fusion_d2 = BotDetectionFusion(
        dataset_type='D2',  # Use D2 dataset parameters for web log detection
        thres_high=0.7,
        thres_low=0.3,
        weight_mv=0.5,
        weight_wl=0.5
    )
    '''
    
    # Example: Evaluate D1 dataset (humans vs moderate bots)
    ground_truth_d1 = os.path.join(dataset_base, 'D1/annotations/humans_and_moderate_bots/test')
    evaluation_d1 = fusion_d1.evaluate_dataset(dataset_base, phase='phase1', ground_truth_path=ground_truth_d1)
    print("D1 Results:")
    print(f"Accuracy: {evaluation_d1['metrics']['accuracy']:.3f}")
    print(f"Bot - Precision: {evaluation_d1['metrics']['bot']['precision']:.3f}, Recall: {evaluation_d1['metrics']['bot']['recall']:.3f}, F-score: {evaluation_d1['metrics']['bot']['f_score']:.3f}")
    print(f"Human - Precision: {evaluation_d1['metrics']['human']['precision']:.3f}, Recall: {evaluation_d1['metrics']['human']['recall']:.3f}, F-score: {evaluation_d1['metrics']['human']['f_score']:.3f}")
    
    # Example: Evaluate D2 dataset (humans vs advanced bots)
    '''
    ground_truth_d2 = os.path.join(dataset_base, 'D2/annotations/humans_and_moderate_bots/test')
    evaluation_d2 = fusion_d2.evaluate_dataset(dataset_base, phase='phase1', ground_truth_path=ground_truth_d2)
    print("\nD2 Results:")
    print(f"Accuracy: {evaluation_d2['metrics']['accuracy']:.3f}")
    print(f"Bot - Precision: {evaluation_d2['metrics']['bot']['precision']:.3f}, Recall: {evaluation_d2['metrics']['bot']['recall']:.3f}, F-score: {evaluation_d2['metrics']['bot']['f_score']:.3f}")
    print(f"Human - Precision: {evaluation_d2['metrics']['human']['precision']:.3f}, Recall: {evaluation_d2['metrics']['human']['recall']:.3f}, F-score: {evaluation_d2['metrics']['human']['f_score']:.3f}")
    '''