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
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# Suppress sklearn version compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class WebLogDetectionBot:
    def __init__(self):
        """
        Initialize the Web Log Detection Bot for sequential training across all phases.
        """
        self.scaler = StandardScaler()
        self.model = None
        self.selected_features = None
        self.is_trained = False
        self.training_histories = []
        
        # Define the selected features based on SFFS algorithm results
        self.selected_features = [
            'total_requests', 'total_session_bytes', 'http_get_requests', 
            'http_post_requests', 'http_head_requests', 'percent_http_3xx_requests',
            'percent_http_4xx_requests', 'percent_image_requests', 'percent_css_requests',
            'percent_js_requests', 'html_to_image_ratio', 'depth_sd',
            'max_requests_per_page', 'avg_requests_per_page', 'max_consecutive_sequential_requests',
            'percent_consecutive_sequential_requests', 'session_time', 'browse_speed',
            'sd_inter_request_times'
        ]
        
    def _initialize_model(self, phase='phase1'):
        """
        Initialize the ensemble model with phase-specific parameters.
        
        Args:
            phase (str): 'phase1' or 'phase2' to specify which phase parameters to use
        """
        if phase == 'phase1':
            # Phase 1 parameters (D1 and D2 combined)
            # SVC parameters
            svc = SVC(kernel='rbf', C=1, gamma=0.03125, tol=0.001, probability=True)
            
            # MLP parameters
            mlp = MLPClassifier(
                activation='relu', solver='sgd', alpha=0.001, beta_1=0.1, beta_2=0.1,
                epsilon=1e-8, hidden_layer_sizes=(100, 50), learning_rate='constant'
            )
            
            # Random Forest parameters
            rf = RandomForestClassifier(
                n_estimators=200, criterion='gini', max_features=None,
                min_samples_leaf=1, min_samples_split=2, max_depth=10, oob_score=True
            )
            
            # Adaboost parameters
            dt_base = DecisionTreeClassifier(criterion='entropy', max_depth=None, max_features=None, splitter='best')
            ada = AdaBoostClassifier(estimator=dt_base, n_estimators=1250, learning_rate=0.5)
            
        elif phase == 'phase2':
            # Phase 2 parameters (incremental learning)
            # SVC parameters
            svc = SVC(kernel='rbf', C=16, gamma=0.002, tol=0.001, probability=True)
            
            # MLP parameters
            mlp = MLPClassifier(
                activation='relu', solver='adam', alpha=0.001, beta_1=0.9, beta_2=0.9,
                epsilon=1e-8, hidden_layer_sizes=(100, 50), learning_rate='constant'
            )
            
            # Random Forest parameters
            rf = RandomForestClassifier(
                n_estimators=200, criterion='gini', max_features=None,
                min_samples_leaf=4, min_samples_split=10, max_depth=10, oob_score=True
            )
            
            # Adaboost parameters
            dt_base = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None, splitter='best')
            ada = AdaBoostClassifier(estimator=dt_base, n_estimators=1250, learning_rate=1.0)
        
        else:
            raise ValueError(f"Unknown phase: {phase}. Use 'phase1' or 'phase2'.")
        
        # Create the voting classifier ensemble
        self.model = VotingClassifier(
            estimators=[('svc', svc), ('mlp', mlp), ('rf', rf), ('ada', ada)], voting='soft')
        
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
        line_count = 0
        parsed_count = 0
        
        with open(log_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                line_count += 1
                if not line:  # Skip empty lines
                    continue
                
                # Parse the log line manually since the format is complex
                try:
                    # Find the timestamp section first
                    timestamp_start = line.find('[')
                    timestamp_end = line.find(']')
                    
                    if timestamp_start == -1 or timestamp_end == -1:
                        continue
                    
                    # Extract timestamp
                    timestamp_str = line[timestamp_start + 1:timestamp_end]
                    
                    # Split the rest of the line after the timestamp
                    rest_of_line = line[timestamp_end + 1:].strip()
                    parts = rest_of_line.split(' ')
                    
                    # Expected format after timestamp: "method path protocol" status bytes "referer" session_id "user_agent"
                    if len(parts) < 8:
                        continue
                    
                    # Extract method, path, protocol (parts[0], 1, 2)
                    method = parts[0].strip('"')
                    path = parts[1]
                    protocol = parts[2].strip('"')
                    
                    # Extract status and bytes
                    status_str = parts[3]
                    bytes_str = parts[4]
                    
                    # Skip if status or bytes are not valid numbers
                    if not status_str.isdigit() or not bytes_str.isdigit():
                        continue
                    
                    status = int(status_str)
                    bytes_sent = int(bytes_str)
                    
                    # Extract referer (part[5])
                    referer = parts[5].strip('"')
                    
                    # Extract session_id (part[6])
                    session_id = parts[6]
                    
                    # Extract user_agent (everything after session_id, joined back together)
                    user_agent = ' '.join(parts[7:]).strip('"')
                    
                    # Skip entries without session ID
                    if session_id == '-':
                        continue
                    
                    # Parse timestamp
                    timestamp = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
                    
                    # Create log entry dictionary
                    log_entry = {
                        'ip': '-',  # We don't have IP in this format
                        'timestamp': timestamp,
                        'method': method,
                        'path': path,
                        'protocol': protocol,
                        'status': status,
                        'bytes_sent': bytes_sent,
                        'referer': referer,
                        'user_agent': user_agent,
                        'session_id': session_id
                    }
                    
                    # Add log entry to corresponding session
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append(log_entry)
                    parsed_count += 1
                    
                except (ValueError, IndexError) as e:
                    # Skip lines that can't be parsed
                    if line_count <= 5:  # Only show first few errors
                        print(f"Debug: Parse error on line {line_count}: {e}")
                        print(f"Debug: Line: {line[:100]}...")
                    continue
                except Exception as e:
                    if line_count <= 5:  # Only show first few errors
                        print(f"Warning: Error processing log line {line_count}: {line[:100]}... Error: {e}")
                    continue
        
        print(f"Debug: Processed {line_count} lines, parsed {parsed_count} entries, found {len(sessions)} sessions")
        
        # Sort log entries in each session by timestamp
        for session_id in sessions:
            sessions[session_id].sort(key=lambda x: x['timestamp'])
        
        # Split sessions if there's a gap of more than session_timeout minutes
        final_sessions = {}
        for session_id, log_entries in sessions.items():
            if len(log_entries) == 0:
                continue
                
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
    
    def train(self, X_train, y_train, validation_split=0.2, sequence_name="Training"):
        """
        Train the model on the provided training data.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training labels
            validation_split (float): Fraction of data to use for validation
            sequence_name (str): Name of the training sequence for tracking
        """
        # Select only the features determined by feature selection
        X_train_selected = X_train[self.selected_features]
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        
        # Split data for validation if needed
        if validation_split > 0:
            from sklearn.model_selection import train_test_split
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_scaled, y_train, test_size=validation_split, random_state=42)
        else:
            X_train_final, y_train_final = X_train_scaled, y_train
            X_val, y_val = None, None
        
        # Train the model and track metrics
        train_scores = []
        val_scores = []
        train_losses = []
        val_losses = []
        
        # Train each base estimator separately to track metrics
        for name, estimator in self.model.estimators:
            print(f"Training {name}...")
            estimator.fit(X_train_final, y_train_final)
            
            # Calculate training accuracy
            train_acc = estimator.score(X_train_final, y_train_final)
            train_scores.append(train_acc)
            
            # Calculate validation metrics if validation data exists
            if X_val is not None and y_val is not None:
                val_acc = estimator.score(X_val, y_val)
                val_scores.append(val_acc)
                
                # For loss calculation (approximation)
                if hasattr(estimator, "predict_proba"):
                    y_val_pred = estimator.predict_proba(X_val)
                    from sklearn.metrics import log_loss
                    val_loss = log_loss(y_val, y_val_pred)
                    val_losses.append(val_loss)
                    
                    y_train_pred = estimator.predict_proba(X_train_final)
                    train_loss = log_loss(y_train_final, y_train_pred)
                    train_losses.append(train_loss)
        
        # Train the ensemble model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Store training history for subplot generation
        history = {
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        self.training_histories.append((sequence_name, history))
        
        return history
    
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

    def load_phase_data(self, dataset_base_path, phase='phase1', dataset_type='D1', data_type='train'):
        """
        Load web log data from a specific phase and dataset type.
        
        Args:
            dataset_base_path: Base path to the dataset
            phase: 'phase1' or 'phase2'
            dataset_type: 'D1' or 'D2'
            data_type: 'train' or 'test'
            
        Returns:
            tuple: (features_df, labels) - DataFrame with features and corresponding labels
        """
        all_features = []
        all_labels = []
        
        if phase == 'phase1':
            # Phase 1 structure: separate human and bot log files
            if dataset_type == 'D1':
                humans_dir = os.path.join(dataset_base_path, 'D1', 'data', 'web_logs', 'humans')
                bots_dir = os.path.join(dataset_base_path, 'D1', 'data', 'web_logs', 'bots')
            else:  # D2
                humans_dir = os.path.join(dataset_base_path, 'D2', 'data', 'web_logs', 'humans')
                bots_dir = os.path.join(dataset_base_path, 'D2', 'data', 'web_logs', 'bots')
            
            print(f"Loading Phase 1 {dataset_type} {data_type} data...")
            print(f"Humans directory: {humans_dir}")
            print(f"Bots directory: {bots_dir}")
            
            # Load human data
            if os.path.exists(humans_dir):
                human_files = [f for f in os.listdir(humans_dir) if f.endswith('.log')]
                print(f"Found {len(human_files)} human log files")
                # Split files for train/test
                if data_type == 'train':
                    human_files = human_files[:int(len(human_files) * 0.8)]
                else:
                    human_files = human_files[int(len(human_files) * 0.8):]
                print(f"Processing {len(human_files)} human files for {data_type}")
                
                for log_file in human_files:
                    log_path = os.path.join(humans_dir, log_file)
                    try:
                        print(f"Processing human log file: {log_file}")
                        sessions = self.extract_sessions(log_path)
                        print(f"Extracted {len(sessions)} sessions from {log_file}")
                        features_df = self.extract_features(sessions)
                        print(f"Extracted features for {len(features_df)} sessions from {log_file}")
                        if not features_df.empty:
                            all_features.append(features_df)
                            all_labels.extend([0] * len(features_df))  # 0 for human
                    except Exception as e:
                        print(f"Warning: Error processing human log file {log_file}: {e}")
            else:
                print(f"Humans directory does not exist: {humans_dir}")
            
            # Load bot data
            if os.path.exists(bots_dir):
                bot_files = [f for f in os.listdir(bots_dir) if f.endswith('.log')]
                print(f"Found {len(bot_files)} bot log files")
                # Split files for train/test - ensure at least one bot file for training
                if data_type == 'train':
                    if len(bot_files) == 1:
                        bot_files = bot_files  # Use all bot files for training
                    else:
                        bot_files = bot_files[:max(1, int(len(bot_files) * 0.8))]
                else:
                    if len(bot_files) == 1:
                        bot_files = []  # No bot files for test if only one exists
                    else:
                        bot_files = bot_files[int(len(bot_files) * 0.8):]
                print(f"Processing {len(bot_files)} bot files for {data_type}")
                
                for log_file in bot_files:
                    log_path = os.path.join(bots_dir, log_file)
                    try:
                        print(f"Processing bot log file: {log_file}")
                        sessions = self.extract_sessions(log_path)
                        print(f"Extracted {len(sessions)} sessions from {log_file}")
                        features_df = self.extract_features(sessions)
                        print(f"Extracted features for {len(features_df)} sessions from {log_file}")
                        if not features_df.empty:
                            all_features.append(features_df)
                            all_labels.extend([1] * len(features_df))  # 1 for bot
                    except Exception as e:
                        print(f"Warning: Error processing bot log file {log_file}: {e}")
            else:
                print(f"Bots directory does not exist: {bots_dir}")
        
        elif phase == 'phase2':
            # Phase 2 structure: separate human and bot log files
            if dataset_type == 'D1':
                humans_dir = os.path.join(dataset_base_path, 'D1', 'data', 'web_logs', 'humans')
                bots_dir = os.path.join(dataset_base_path, 'D1', 'data', 'web_logs', 'bots')
            else:  # D2
                humans_dir = os.path.join(dataset_base_path, 'D2', 'data', 'web_logs', 'humans')
                bots_dir = os.path.join(dataset_base_path, 'D2', 'data', 'web_logs', 'bots')
            
            print(f"Loading Phase 2 {dataset_type} {data_type} data...")
            print(f"Humans directory: {humans_dir}")
            print(f"Bots directory: {bots_dir}")
            
            # Load human data
            if os.path.exists(humans_dir):
                human_files = [f for f in os.listdir(humans_dir) if f.endswith('.log')]
                print(f"Found {len(human_files)} human log files")
                # Split files for train/test
                if data_type == 'train':
                    human_files = human_files[:int(len(human_files) * 0.8)]
                else:
                    human_files = human_files[int(len(human_files) * 0.8):]
                print(f"Processing {len(human_files)} human files for {data_type}")
                
                for log_file in human_files:
                    log_path = os.path.join(humans_dir, log_file)
                    try:
                        print(f"Processing human log file: {log_file}")
                        sessions = self.extract_sessions(log_path)
                        print(f"Extracted {len(sessions)} sessions from {log_file}")
                        features_df = self.extract_features(sessions)
                        print(f"Extracted features for {len(features_df)} sessions from {log_file}")
                        if not features_df.empty:
                            all_features.append(features_df)
                            all_labels.extend([0] * len(features_df))  # 0 for human
                    except Exception as e:
                        print(f"Warning: Error processing human log file {log_file}: {e}")
            else:
                print(f"Humans directory does not exist: {humans_dir}")
            
            # Load bot data
            if os.path.exists(bots_dir):
                bot_files = [f for f in os.listdir(bots_dir) if f.endswith('.log')]
                print(f"Found {len(bot_files)} bot log files")
                # Split files for train/test - ensure at least one bot file for training
                if data_type == 'train':
                    if len(bot_files) == 1:
                        bot_files = bot_files  # Use all bot files for training
                    else:
                        bot_files = bot_files[:max(1, int(len(bot_files) * 0.8))]
                else:
                    if len(bot_files) == 1:
                        bot_files = []  # No bot files for test if only one exists
                    else:
                        bot_files = bot_files[int(len(bot_files) * 0.8):]
                print(f"Processing {len(bot_files)} bot files for {data_type}")
                
                for log_file in bot_files:
                    log_path = os.path.join(bots_dir, log_file)
                    try:
                        print(f"Processing bot log file: {log_file}")
                        sessions = self.extract_sessions(log_path)
                        print(f"Extracted {len(sessions)} sessions from {log_file}")
                        features_df = self.extract_features(sessions)
                        print(f"Extracted features for {len(features_df)} sessions from {log_file}")
                        if not features_df.empty:
                            all_features.append(features_df)
                            all_labels.extend([1] * len(features_df))  # 1 for bot
                    except Exception as e:
                        print(f"Warning: Error processing bot log file {log_file}: {e}")
            else:
                print(f"Bots directory does not exist: {bots_dir}")
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            print(f"Loaded {len(combined_features)} sessions with {len(all_labels)} labels from {phase} {dataset_type} {data_type}")
            return combined_features, np.array(all_labels)
        else:
            print(f"No data found for {phase} {dataset_type} {data_type}")
            return pd.DataFrame(), np.array([])
    
    def train_sequentially(self, phase1_d1_data, phase1_d1_labels, phase1_d2_data, phase1_d2_labels,
                          phase2_d1_data=None, phase2_d1_labels=None, phase2_d2_data=None, phase2_d2_labels=None):
        """
        Train the model sequentially: first on phase1 data, then on phase2 data.
        
        Args:
            phase1_d1_data: Training data for phase1 D1
            phase1_d1_labels: Labels for phase1 D1 data
            phase1_d2_data: Training data for phase1 D2
            phase1_d2_labels: Labels for phase1 D2 data
            phase2_d1_data: Training data for phase2 D1 (optional)
            phase2_d1_labels: Labels for phase2 D1 data (optional)
            phase2_d2_data: Training data for phase2 D2 (optional)
            phase2_d2_labels: Labels for phase2 D2 data (optional)
        """
        print("=== SEQUENTIAL TRAINING: PHASE 1 → PHASE 2 ===")
        
        # === PHASE 1 TRAINING ===
        print("\n=== PHASE 1 TRAINING ===")
        
        # Initialize model for Phase 1
        self._initialize_model(phase='phase1')
        
        # Train on Phase 1 D1
        if not phase1_d1_data.empty:
            print(f"Training on Phase 1 D1: {len(phase1_d1_data)} sessions")
            self.train(phase1_d1_data, phase1_d1_labels)
            print("Phase 1 D1 training completed.")
        
        # Train on Phase 1 D2
        if not phase1_d2_data.empty:
            print(f"Training on Phase 1 D2: {len(phase1_d2_data)} sessions")
            self.train(phase1_d2_data, phase1_d2_labels)
            print("Phase 1 D2 training completed.")
        
        self.is_trained = True
        
        # === PHASE 2 TRAINING (INCREMENTAL) ===
        if phase2_d1_data is not None or phase2_d2_data is not None:
            print("\n=== PHASE 2 INCREMENTAL TRAINING ===")
            
            # Reinitialize model for Phase 2 (incremental learning)
            self._initialize_model(phase='phase2')
            
            # Train on Phase 2 D1
            if phase2_d1_data is not None and not phase2_d1_data.empty:
                print(f"Training on Phase 2 D1: {len(phase2_d1_data)} sessions")
                self.train(phase2_d1_data, phase2_d1_labels)
                print("Phase 2 D1 training completed.")
            
            # Train on Phase 2 D2
            if phase2_d2_data is not None and not phase2_d2_data.empty:
                print(f"Training on Phase 2 D2: {len(phase2_d2_data)} sessions")
                self.train(phase2_d2_data, phase2_d2_labels)
                print("Phase 2 D2 training completed.")
        
        print("Sequential training completed!")
    
    def evaluate_model(self, test_data, test_labels, dataset_name="Test"):
        """
        Evaluate the model on test data and return accuracy.
        
        Args:
            test_data: Test data
            test_labels: True labels for test data
            dataset_name: Name of the dataset for reporting
            
        Returns:
            float: Accuracy score
        """
        if test_data.empty or len(test_labels) == 0:
            print(f"No {dataset_name} data provided")
            return 0.0
        
        # Predict class labels
        predictions = self.predict_class(test_data)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == test_labels)
        print(f"{dataset_name} Accuracy: {accuracy:.4f}")
        return accuracy
    
    def save_model(self, model_path):
        """
        Save the trained model and scaler to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        # Get project root directory if model_path is not absolute
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, model_path)
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to load the model from
        """
        # Get project root directory if model_path is not absolute
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, model_path)
            
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.selected_features = model_data['selected_features']
        self.is_trained = model_data['is_trained']
        
    def generate_training_graphs(self):
        """
        Generate training graphs with subplots for each training sequence.
        Creates two figures: one for accuracy and one for loss.
        """
        if not self.training_histories:
            print("No training histories available for plotting.")
            return
        
        # Calculate project root for saving graphs
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        num_sequences = len(self.training_histories)
        
        # Create accuracy subplot figure
        fig_acc, axes_acc = plt.subplots(2, 2, figsize=(15, 10))
        fig_acc.suptitle('Web Log Detection - Training Accuracy by Sequence', fontsize=16)
        axes_acc = axes_acc.flatten()
        
        # Create loss subplot figure
        fig_loss, axes_loss = plt.subplots(2, 2, figsize=(15, 10))
        fig_loss.suptitle('Web Log Detection - Training Loss by Sequence', fontsize=16)
        axes_loss = axes_loss.flatten()
        
        for i, (sequence_name, history) in enumerate(self.training_histories):
            if i >= 4:  # Limit to 4 subplots
                break
                
            # Plot accuracy
            ax_acc = axes_acc[i]
            if history['train_scores']:
                epochs = range(1, len(history['train_scores']) + 1)
                ax_acc.plot(epochs, history['train_scores'], 'b-', label='Training Accuracy', linewidth=2)
                if history['val_scores']:
                    ax_acc.plot(epochs, history['val_scores'], 'r-', label='Validation Accuracy', linewidth=2)
                ax_acc.set_title(f'{sequence_name} - Accuracy')
                ax_acc.set_xlabel('Estimator')
                ax_acc.set_ylabel('Accuracy')
                ax_acc.legend()
                ax_acc.grid(True, alpha=0.3)
            
            # Plot loss
            ax_loss = axes_loss[i]
            if history['train_losses']:
                epochs = range(1, len(history['train_losses']) + 1)
                ax_loss.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
                if history['val_losses']:
                    ax_loss.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
                ax_loss.set_title(f'{sequence_name} - Loss')
                ax_loss.set_xlabel('Estimator')
                ax_loss.set_ylabel('Loss')
                ax_loss.legend()
                ax_loss.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_sequences, 4):
            axes_acc[i].set_visible(False)
            axes_loss[i].set_visible(False)
        
        # Adjust layout and save
        fig_acc.tight_layout()
        fig_loss.tight_layout()
        
        accuracy_path = os.path.join(results_dir, 'web_log_accuracy.png')
        loss_path = os.path.join(results_dir, 'web_log_loss.png')
        
        fig_acc.savefig(accuracy_path, dpi=300, bbox_inches='tight')
        fig_loss.savefig(loss_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig_acc)
        plt.close(fig_loss)
        
        print(f"Training graphs saved:")
        print(f"  Accuracy: {accuracy_path}")
        print(f"  Loss: {loss_path}")

# Example usage
if __name__ == "__main__":
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Dataset paths
    dataset_base_phase1 = os.path.join(project_root, 'dataset/phase1')
    dataset_base_phase2 = os.path.join(project_root, 'dataset/phase2')

    # Initialize the web log detector
    detector = WebLogDetectionBot()
    
    print("=== COMPREHENSIVE SEQUENTIAL TRAINING: ALL DATASETS ===")
    print("Training on all available datasets in sequential order...")
    
    # === PHASE 1 TRAINING (FIRST STAGE) ===
    print("\n" + "="*60)
    print("PHASE 1 TRAINING: D1 (Humans vs Moderate Bots)")
    print("="*60)
    
    # Load Phase 1 D1 training data
    phase1_d1_train_data, phase1_d1_train_labels = detector.load_phase_data(
        dataset_base_phase1, phase='phase1', dataset_type='D1', data_type='train'
    )
    
    if not phase1_d1_train_data.empty:
        print(f"Training on Phase 1 D1: {len(phase1_d1_train_data)} sessions")
        detector._initialize_model(phase='phase1')
        detector.train(phase1_d1_train_data, phase1_d1_train_labels, sequence_name="Phase 1 D1")
        print("Phase 1 D1 training completed.")
    else:
        print("No Phase 1 D1 training data found.")
    
    print("\n" + "="*60)
    print("PHASE 1 TRAINING: D2 (Humans vs Advanced Bots)")
    print("="*60)
    
    # Load Phase 1 D2 training data
    phase1_d2_train_data, phase1_d2_train_labels = detector.load_phase_data(
        dataset_base_phase1, phase='phase1', dataset_type='D2', data_type='train'
    )
    
    if not phase1_d2_train_data.empty:
        print(f"Training on Phase 1 D2: {len(phase1_d2_train_data)} sessions")
        detector.train(phase1_d2_train_data, phase1_d2_train_labels, sequence_name="Phase 1 D2")
        print("Phase 1 D2 training completed.")
    else:
        print("No Phase 1 D2 training data found.")
    
    detector.is_trained = True
    
    # === PHASE 2 INCREMENTAL TRAINING (SECOND STAGE) ===
    print("\n" + "="*60)
    print("PHASE 2 INCREMENTAL TRAINING: D1 (Humans vs Moderate & Advanced Bots)")
    print("="*60)
    
    # Load Phase 2 D1 training data
    phase2_d1_train_data, phase2_d1_train_labels = detector.load_phase_data(
        dataset_base_phase2, phase='phase2', dataset_type='D1', data_type='train'
    )
    
    if not phase2_d1_train_data.empty:
        print(f"Training on Phase 2 D1: {len(phase2_d1_train_data)} sessions")
        detector._initialize_model(phase='phase2')
        detector.train(phase2_d1_train_data, phase2_d1_train_labels, sequence_name="Phase 2 D1")
        print("Phase 2 D1 training completed.")
    else:
        print("No Phase 2 D1 training data found.")
    
    print("\n" + "="*60)
    print("PHASE 2 INCREMENTAL TRAINING: D2 (Humans vs Advanced Bots)")
    print("="*60)
    
    # Load Phase 2 D2 training data
    phase2_d2_train_data, phase2_d2_train_labels = detector.load_phase_data(
        dataset_base_phase2, phase='phase2', dataset_type='D2', data_type='train'
    )
    
    if not phase2_d2_train_data.empty:
        print(f"Training on Phase 2 D2: {len(phase2_d2_train_data)} sessions")
        detector.train(phase2_d2_train_data, phase2_d2_train_labels, sequence_name="Phase 2 D2")
        print("Phase 2 D2 training completed.")
    else:
        print("No Phase 2 D2 training data found.")
    
    # Save final model
    final_model_path = os.path.join(project_root, 'models/logs_model.pkl')
    detector.save_model(final_model_path)
    print(f"\nFinal comprehensive model saved to {final_model_path}")
    
    # Generate training graphs with all sequences
    detector.generate_training_graphs()
    
    # === COMPREHENSIVE TESTING PHASE ===
    print("\n" + "="*60)
    print("COMPREHENSIVE TESTING ON ALL DATASETS")
    print("="*60)
    
    # Test on Phase 1 D1
    print("\n--- Testing on Phase 1 D1 (Humans vs Moderate Bots) ---")
    phase1_d1_test_data, phase1_d1_test_labels = detector.load_phase_data(
        dataset_base_phase1, phase='phase1', dataset_type='D1', data_type='test'
    )
    
    if not phase1_d1_test_data.empty:
        detector.evaluate_model(phase1_d1_test_data, phase1_d1_test_labels, "Phase 1 D1 Test")
    else:
        print("No Phase 1 D1 test data found.")
    
    # Test on Phase 1 D2
    print("\n--- Testing on Phase 1 D2 (Humans vs Advanced Bots) ---")
    phase1_d2_test_data, phase1_d2_test_labels = detector.load_phase_data(
        dataset_base_phase1, phase='phase1', dataset_type='D2', data_type='test'
    )
    
    if not phase1_d2_test_data.empty:
        detector.evaluate_model(phase1_d2_test_data, phase1_d2_test_labels, "Phase 1 D2 Test")
    else:
        print("No Phase 1 D2 test data found.")
    
    # Test on Phase 2 D1
    print("\n--- Testing on Phase 2 D1 (Humans vs Moderate & Advanced Bots) ---")
    phase2_d1_test_data, phase2_d1_test_labels = detector.load_phase_data(
        dataset_base_phase2, phase='phase2', dataset_type='D1', data_type='test'
    )
    
    if not phase2_d1_test_data.empty:
        detector.evaluate_model(phase2_d1_test_data, phase2_d1_test_labels, "Phase 2 D1 Test")
    else:
        print("No Phase 2 D1 test data found.")
    
    # Test on Phase 2 D2
    print("\n--- Testing on Phase 2 D2 (Humans vs Advanced Bots) ---")
    phase2_d2_test_data, phase2_d2_test_labels = detector.load_phase_data(
        dataset_base_phase2, phase='phase2', dataset_type='D2', data_type='test'
    )
    
    if not phase2_d2_test_data.empty:
        detector.evaluate_model(phase2_d2_test_data, phase2_d2_test_labels, "Phase 2 D2 Test")
    else:
        print("No Phase 2 D2 test data found.")
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print("Training completed on all available datasets:")
    print("1. Phase 1 D1: Humans vs Moderate Bots")
    print("2. Phase 1 D2: Humans vs Advanced Bots") 
    print("3. Phase 2 D1: Humans vs Moderate & Advanced Bots")
    print("4. Phase 2 D2: Humans vs Advanced Bots")
    print(f"\nFinal model saved as: {final_model_path}")
    print("="*60)