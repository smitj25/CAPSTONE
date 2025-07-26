import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import re
from datetime import datetime, timedelta
import os

class WebLogDetectionBot:
    def __init__(self, dataset_type='D1'):
        """
        Initialize the Web Log Detection Bot with dataset-specific parameters.
        
        Args:
            dataset_type (str): 'D1' or 'D2' to specify which dataset parameters to use
        """
        self.dataset_type = dataset_type
        self.scaler = StandardScaler()
        self.model = None
        self.selected_features = None
        
        # Initialize the model with dataset-specific parameters
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the ensemble model with dataset-specific parameters.
        """
        if self.dataset_type == 'D1':
            # SVC parameters for D1
            svc = SVC(kernel='rbf', C=1, gamma=0.03125, tol=0.001, probability=True)
            
            # MLP parameters for D1
            mlp = MLPClassifier(
                activation='relu', solver='sgd', alpha=0.001, beta_1=0.1, beta_2=0.1,
                epsilon=1e-8, hidden_layer_sizes=(100, 50), learning_rate='constant'
            )
            
            # Random Forest parameters for D1
            rf = RandomForestClassifier(
                n_estimators=200, criterion='gini', max_features=None,
                min_samples_leaf=1, min_samples_split=2, max_depth=10, oob_score=True
            )
            
            # Adaboost parameters for D1
            dt_base = DecisionTreeClassifier(criterion='entropy', max_depth=None, max_features=None, splitter='best')
            ada = AdaBoostClassifier(estimator=dt_base, n_estimators=1250, learning_rate=0.5)
            
        elif self.dataset_type == 'D2':
            # SVC parameters for D2
            svc = SVC(kernel='rbf', C=16, gamma=0.002, tol=0.001, probability=True)
            
            # MLP parameters for D2
            mlp = MLPClassifier(
                activation='relu', solver='adam', alpha=0.001, beta_1=0.9, beta_2=0.9,
                epsilon=1e-8, hidden_layer_sizes=(100, 50), learning_rate='constant'
            )
            
            # Random Forest parameters for D2
            rf = RandomForestClassifier(
                n_estimators=200, criterion='gini', max_features=None,
                min_samples_leaf=4, min_samples_split=10, max_depth=10, oob_score=True
            )
            
            # Adaboost parameters for D2
            dt_base = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None, splitter='best')
            ada = AdaBoostClassifier(estimator=dt_base, n_estimators=1250, learning_rate=1.0)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}. Use 'D1' or 'D2'.")
        
        # Create the voting classifier ensemble
        self.model = VotingClassifier(
            estimators=[
                ('svc', svc),
                ('mlp', mlp),
                ('rf', rf),
                ('ada', ada)
            ],
            voting='soft'  # Use probability estimates for voting
        )
        
        # Define the selected features based on SFFS algorithm results
        # These would be determined by your feature selection process
        # For now, we'll use all 14 features mentioned in the requirements
        self.selected_features = [
            'total_requests', 'total_session_bytes', 'http_get_requests', 
            'http_post_requests', 'http_head_requests', 'percent_http_3xx_requests',
            'percent_http_4xx_requests', 'percent_image_requests', 'percent_css_requests',
            'percent_js_requests', 'html_to_image_ratio', 'depth_sd',
            'max_requests_per_page', 'avg_requests_per_page', 'max_consecutive_sequential_requests',
            'percent_consecutive_sequential_requests', 'session_time', 'browse_speed',
            'sd_inter_request_times'
        ]
        
    def extract_sessions(self, log_file_path, session_timeout=30):
        """
        Extract visitor sessions from web log files using PHP session IDs.
        A session concludes if no new requests with its ID occur for over 30 minutes.
        
        Args:
            log_file_path (str): Path to the Apache2 log file
            session_timeout (int): Session timeout in minutes (default: 30)
            
        Returns:
            dict: Dictionary of sessions with session_id as key and list of log entries as value
        """
        sessions = {}
        
        # Apache log format pattern
        # This pattern assumes the standard Apache combined log format with PHP session ID
        log_pattern = r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "([^"]*)" "([^"]*)" (\S+)$'
        
        with open(log_file_path, 'r') as f:
            for line in f:
                match = re.match(log_pattern, line.strip())
                if match:
                    # Extract information from log entry
                    ip, _, _, timestamp_str, method, path, protocol, status, bytes_sent, referer, user_agent, session_id = match.groups()
                    
                    # Parse timestamp
                    timestamp = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
                    
                    # Create log entry dictionary
                    log_entry = {
                        'ip': ip,
                        'timestamp': timestamp,
                        'method': method,
                        'path': path,
                        'protocol': protocol,
                        'status': int(status),
                        'bytes_sent': int(bytes_sent),
                        'referer': referer,
                        'user_agent': user_agent,
                        'session_id': session_id
                    }
                    
                    # Add log entry to corresponding session
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append(log_entry)
        
        # Sort log entries in each session by timestamp
        for session_id in sessions:
            sessions[session_id].sort(key=lambda x: x['timestamp'])
        
        # Split sessions if there's a gap of more than session_timeout minutes
        final_sessions = {}
        for session_id, log_entries in sessions.items():
            current_session_id = session_id
            current_session_entries = [log_entries[0]]
            
            for i in range(1, len(log_entries)):
                time_diff = (log_entries[i]['timestamp'] - log_entries[i-1]['timestamp']).total_seconds() / 60
                
                if time_diff > session_timeout:
                    # End current session and start a new one
                    final_sessions[current_session_id] = current_session_entries
                    current_session_id = f"{session_id}_{i}"
                    current_session_entries = []
                
                current_session_entries.append(log_entries[i])
            
            # Add the last session
            final_sessions[current_session_id] = current_session_entries
        
        return final_sessions
    
    def extract_features(self, sessions):
        """
        Extract features from sessions for model training/prediction.
        
        Args:
            sessions (dict): Dictionary of sessions with session_id as key and list of log entries as value
            
        Returns:
            pandas.DataFrame: DataFrame with extracted features for each session
        """
        features = []
        
        for session_id, log_entries in sessions.items():
            # Skip sessions with too few requests
            if len(log_entries) < 2:
                continue
            
            # Basic session information
            total_requests = len(log_entries)
            total_session_bytes = sum(entry['bytes_sent'] for entry in log_entries)
            
            # HTTP request methods
            http_get_requests = sum(1 for entry in log_entries if entry['method'] == 'GET')
            http_post_requests = sum(1 for entry in log_entries if entry['method'] == 'POST')
            http_head_requests = sum(1 for entry in log_entries if entry['method'] == 'HEAD')
            
            # HTTP response codes
            http_3xx_requests = sum(1 for entry in log_entries if 300 <= entry['status'] < 400)
            http_4xx_requests = sum(1 for entry in log_entries if 400 <= entry['status'] < 500)
            
            percent_http_3xx_requests = (http_3xx_requests / total_requests) * 100 if total_requests > 0 else 0
            percent_http_4xx_requests = (http_4xx_requests / total_requests) * 100 if total_requests > 0 else 0
            
            # File types requested
            image_requests = sum(1 for entry in log_entries if any(ext in entry['path'].lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']))
            css_requests = sum(1 for entry in log_entries if '.css' in entry['path'].lower())
            js_requests = sum(1 for entry in log_entries if '.js' in entry['path'].lower())
            html_requests = sum(1 for entry in log_entries if any(ext in entry['path'].lower() for ext in ['.html', '.htm', '.php', '.asp', '.jsp']))
            
            percent_image_requests = (image_requests / total_requests) * 100 if total_requests > 0 else 0
            percent_css_requests = (css_requests / total_requests) * 100 if total_requests > 0 else 0
            percent_js_requests = (js_requests / total_requests) * 100 if total_requests > 0 else 0
            
            html_to_image_ratio = html_requests / image_requests if image_requests > 0 else html_requests
            
            # Browse behavior
            # Extract page depth from URL path
            depths = [len(entry['path'].strip('/').split('/')) for entry in log_entries]
            depth_sd = np.std(depths) if len(depths) > 1 else 0
            
            # Group requests by page (simplified by assuming a page change when HTML content is requested)
            pages = []
            current_page = [log_entries[0]]
            
            for i in range(1, len(log_entries)):
                if any(ext in log_entries[i]['path'].lower() for ext in ['.html', '.htm', '.php', '.asp', '.jsp']):
                    pages.append(current_page)
                    current_page = []
                current_page.append(log_entries[i])
            
            if current_page:
                pages.append(current_page)
            
            requests_per_page = [len(page) for page in pages]
            max_requests_per_page = max(requests_per_page) if requests_per_page else 0
            avg_requests_per_page = sum(requests_per_page) / len(requests_per_page) if requests_per_page else 0
            
            # Sequential HTTP requests (simplified by checking if paths are similar)
            consecutive_sequential = 0
            max_consecutive_sequential = 0
            current_consecutive = 0
            
            for i in range(1, len(log_entries)):
                prev_path_parts = log_entries[i-1]['path'].strip('/').split('/')
                curr_path_parts = log_entries[i]['path'].strip('/').split('/')
                
                # Check if paths share the same base directory
                if len(prev_path_parts) > 0 and len(curr_path_parts) > 0 and prev_path_parts[0] == curr_path_parts[0]:
                    current_consecutive += 1
                else:
                    current_consecutive = 0
                
                consecutive_sequential += 1 if current_consecutive > 0 else 0
                max_consecutive_sequential = max(max_consecutive_sequential, current_consecutive)
            
            percent_consecutive_sequential = (consecutive_sequential / (total_requests - 1)) * 100 if total_requests > 1 else 0
            
            # Session time and browse speed
            session_start = log_entries[0]['timestamp']
            session_end = log_entries[-1]['timestamp']
            session_time = (session_end - session_start).total_seconds()
            
            # Calculate inter-request times
            inter_request_times = [(log_entries[i]['timestamp'] - log_entries[i-1]['timestamp']).total_seconds() 
                                 for i in range(1, len(log_entries))]
            
            browse_speed = total_requests / session_time if session_time > 0 else 0
            sd_inter_request_times = np.std(inter_request_times) if len(inter_request_times) > 1 else 0
            
            # Compile all features
            session_features = {
                'session_id': session_id,
                'total_requests': total_requests,
                'total_session_bytes': total_session_bytes,
                'http_get_requests': http_get_requests,
                'http_post_requests': http_post_requests,
                'http_head_requests': http_head_requests,
                'percent_http_3xx_requests': percent_http_3xx_requests,
                'percent_http_4xx_requests': percent_http_4xx_requests,
                'percent_image_requests': percent_image_requests,
                'percent_css_requests': percent_css_requests,
                'percent_js_requests': percent_js_requests,
                'html_to_image_ratio': html_to_image_ratio,
                'depth_sd': depth_sd,
                'max_requests_per_page': max_requests_per_page,
                'avg_requests_per_page': avg_requests_per_page,
                'max_consecutive_sequential_requests': max_consecutive_sequential,
                'percent_consecutive_sequential_requests': percent_consecutive_sequential,
                'session_time': session_time,
                'browse_speed': browse_speed,
                'sd_inter_request_times': sd_inter_request_times
            }
            
            features.append(session_features)
        
        return pd.DataFrame(features)
    
    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training labels
        """
        # Select only the features determined by feature selection
        X_train_selected = X_train[self.selected_features]
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X_test):
        """
        Predict the class probabilities for the provided test data.
        
        Args:
            X_test (pandas.DataFrame): Test features
            
        Returns:
            numpy.ndarray: Class probabilities for each sample
        """
        # Select only the features determined by feature selection
        X_test_selected = X_test[self.selected_features]
        
        # Scale the features using the same scaler fitted on training data
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Predict probabilities
        return self.model.predict_proba(X_test_scaled)
    
    def predict_class(self, X_test):
        """
        Predict the class labels for the provided test data.
        
        Args:
            X_test (pandas.DataFrame): Test features
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        # Select only the features determined by feature selection
        X_test_selected = X_test[self.selected_features]
        
        # Scale the features using the same scaler fitted on training data
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Predict class labels
        return self.model.predict(X_test_scaled)
    
    def get_web_log_score(self, log_file_path):
        """
        Process a web log file and return bot detection scores for each session.
        
        Args:
            log_file_path (str): Path to the Apache2 log file
            
        Returns:
            dict: Dictionary with session_id as key and bot probability as value
        """
        # Extract sessions from the log file
        sessions = self.extract_sessions(log_file_path)
        
        # Extract features from the sessions
        features_df = self.extract_features(sessions)
        
        if features_df.empty:
            return {}
        
        # Predict bot probabilities
        probabilities = self.predict(features_df)
        
        # Create a dictionary of session_id to bot probability
        session_scores = {}
        for i, session_id in enumerate(features_df['session_id']):
            # Assuming class 1 is the bot class
            session_scores[session_id] = probabilities[i][1]
        
        return session_scores

# Example usage
if __name__ == "__main__":
    # Dataset paths
    dataset_base = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase1'
    
    # Initialize the bot detector for dataset D1 (humans vs moderate bots)
    detector_d1 = WebLogDetectionBot(dataset_type='D1')
    
    # Initialize the bot detector for dataset D2 (humans vs advanced bots)
    detector_d2 = WebLogDetectionBot(dataset_type='D2')
    
    # Example: Load training data for D1
    # train_annotations_d1 = os.path.join(dataset_base, 'annotations/humans_and_moderate_bots/train')
    # test_annotations_d1 = os.path.join(dataset_base, 'annotations/humans_and_moderate_bots/test')
    
    # Example: Load training data for D2
    # train_annotations_d2 = os.path.join(dataset_base, 'annotations/humans_and_advanced_bots/train')
    # test_annotations_d2 = os.path.join(dataset_base, 'annotations/humans_and_advanced_bots/test')
    
    # Example: Get bot detection scores from web log files
    # web_logs_dir = os.path.join(dataset_base, 'data/web_logs')
    # for log_file in os.listdir(web_logs_dir):
    #     if log_file.endswith('.txt'):
    #         log_path = os.path.join(web_logs_dir, log_file)
    #         scores = detector_d1.get_web_log_score(log_path)
    #         print(f"Bot detection scores for {log_file}: {scores}")