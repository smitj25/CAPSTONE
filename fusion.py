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


'''
My project is to create a web bot detections model. The basic idea is to detect a bot based on, 
first mouse_movements and second from web logs. I need to create 3 files, one file would detect bots from 
web logs, other would detect web bots from mouse movements and in the third file, the fusion of scores from
    the previous 2 files will be calulated from which the decision of bot/human will be made.

there is a folder called 'dataset' in the CAPSTONE folder. it is not staged for git dues to its size but i would like 
to implement it for testing purposes. understand the dataset structure and rewrite the file paths in the codes to 
make them run on the provided dataset

The Web Bot Detection Dataset is organized into two main folders, phase1 and phase2, each representing a distinct 
evaluation phase.
phase1 Folder Structure:

This folder is used for the first evaluation phase and contains two primary subfolders:
annotations and data.
annotations Folder: This subfolder stores the labels for the sessions.
humans_and_moderate_bots: Contains annotation files for sessions involving human and moderate web bot interactions.
train: A file with session_id and label (human or moderate bot) for training data.
test: A file with session_id and label (human or moderate bot) for testing data.

humans_and_advanced_bots: Contains annotation files for sessions involving human and advanced web bot interactions.
train: A file with session_id and label (human or advanced bot) for training data.
test: A file with session_id and label (human or advanced bot) for testing data.

The train and test files in these annotation folders have two space-separated columns: the PHP session ID and its corresponding annotation (human, moderate bot, or advanced bot).
data Folder: This subfolder contains the raw web logs and mouse movement data.
web_logs: Contains web server logs for humans, moderate bots, and advanced bots. These logs follow the Apache2 log format and include the PHP session ID in each request. Examples include
access_log_1.txt, access_log_2.txt, etc..
mouse_movements: Contains detailed mouse movement data.
humans_and_moderate_bots: Holds mouse movement data for human and moderate web bot sessions. Inside this, there are subfolders named after each {session_id}, which contain a mouse_movements.json file.
humans_and_advanced_bots: Holds mouse movement data for human and advanced web bot sessions. Similarly, it contains subfolders named after each {session_id}, each with a mouse_movements.json file.

Each mouse_movements.json file contains:
session_id: The PHP session ID.
total_behaviour: Mouse movement actions like m(x,y) for a move, c(l) for left click, c(r)for right click, and c(m) for middle click.
mousemove_times: Timestamps of each mouse move in the format {(t0), (t1), ..., (tn)}.
mousemove_total_behaviour: Coordinates of each mouse point in the format {(x0, y0), (x1, y1), ..., (xn, yn)}.

phase2 Folder Structure: This folder is used for the second evaluation phase and also contains
annotations and data subfolders.
annotations Folder: Stores labels for the sessions in this phase.
humans_and_advanced_bots: Contains an annotations.txt file with session_id and label for human and advanced web bot sessions used in the first part of this phase.
humans_and_moderate_and_advanced_bots: Contains an annotations.txt file with session_id and label for human, moderate, and advanced web bot sessions used in the second part of this phase.

These  annotations.txt files have two space-separated columns: the PHP session ID and whether the session is a human, moderate bot, or advanced bot.
data Folder: Contains the raw web logs and mouse movement data for the second phase.
web_logs: Contains web server logs for humans and bots, structured similarly to phase1 with Apache2 log format and PHP session IDs.
mouse_movements: Contains mouse movement data exported as JSON collections from a MongoDB instance. Each file, e.g.,
{session_id}.json, represents a session and includes:
session_id: The PHP session ID.
mousemove_client_height_width: Client's browser height and width during each mouse move {(height0, width0), ..., (heightn, widthn)}.
mousemove_times: Timestamps of each mouse move {(t0), (t1), ..., (tn)}.
mousemove_total_behaviour: Coordinates of the current mouse point {(x0, y0), ..., (xn, yn)}.
mousemove_visited_urls: The URL where the mouse move was performed {(url0), (url1), ..., (urln)}

Based on the provided context, the following folders within the dataset structure are similar to D1 and D2:

The data related to D1 (Humans - Moderate bots) would be found in:
phase1/annotations/humans_and_moderate_bots/train
phase1/annotations/humans_and_moderate_bots/test

The relevant web logs within phase1/data/web_logs that correspond to the session IDs in the humans_and_moderate_bots annotations.
phase1/data/mouse_movements/humans_and_moderate_bots
The data related to D2 (Humans - Advanced bots) would be found in:
phase1/annotations/humans_and_advanced_bots/train
phase1/annotations/humans_and_advanced_bots/test

The relevant web logs within phase1/data/web_logs that correspond to the session IDs in the humans_and_advanced_bots annotations.
phase1/data/mouse_movements/humans_and_advanced_bots use this dataset structure and replace D1 and D2 with corresponding provided paths.

Complete dataset path:
/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase1
/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase2
    '''