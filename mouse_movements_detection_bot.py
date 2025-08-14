import numpy as np
import json
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from keras.optimizers import Adam
from collections import defaultdict

class MouseMovementDetectionBot:
    def __init__(self):
        self.model = None
        self.input_shape = (480, 1320, 1)  # Cropped from 1080x1920 with 300px offset
        self._build_model()
        
    def _build_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=self.input_shape))
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
        self.model.add(Flatten())
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    def _preprocess_mouse_movements(self, mouse_data, phase='phase1'):
        matrices = []
        
        if phase == 'phase1':
            # Extract mouse movements and timestamps
            timestamps = mouse_data.get('mousemove_times', [])
            coordinates = mouse_data.get('mousemove_total_behaviour', [])
            
            # Group by page (in phase1, we don't have URL info, so we'll use time gaps to estimate page changes)
            # A gap of more than 5 seconds might indicate a page change
            page_groups = []
            current_group = []
            
            for i in range(len(timestamps)):
                try:
                    # Handle timestamp parsing with error checking
                    current_time = float(str(timestamps[i]).replace(',', '.'))
                    if i > 0:
                        prev_time = float(str(timestamps[i-1]).replace(',', '.'))
                        if current_time - prev_time > 5000:  # 5 seconds gap
                            if current_group:
                                page_groups.append(current_group)
                                current_group = []
                    
                    if i < len(coordinates):
                        # Clean coordinate data - handle various formats
                        coord = coordinates[i]
                        if isinstance(coord, str):
                            # Parse coordinate string like "(x,y)" or "x,y"
                            coord = coord.strip('()')
                            if ',' in coord:
                                try:
                                    x, y = coord.split(',')
                                    x = float(x.strip())
                                    y = float(y.strip())
                                    coord = (x, y)
                                except ValueError:
                                    continue  # Skip invalid coordinates
                        elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            try:
                                x = float(str(coord[0]).replace(',', '.'))
                                y = float(str(coord[1]).replace(',', '.'))
                                coord = (x, y)
                            except (ValueError, IndexError):
                                continue  # Skip invalid coordinates
                        else:
                            continue  # Skip invalid coordinate formats
                        
                        current_group.append((coord, current_time))
                except (ValueError, TypeError, IndexError) as e:
                    # Skip problematic entries
                    continue
            
            if current_group:
                page_groups.append(current_group)
            
        else:  # phase2
            # Extract mouse movements, timestamps, and URLs
            timestamps = mouse_data.get('mousemove_times', [])
            coordinates = mouse_data.get('mousemove_total_behaviour', [])
            urls = mouse_data.get('mousemove_visited_urls', [])
            
            # Group by URL
            url_to_movements = defaultdict(list)
            
            for i in range(min(len(timestamps), len(coordinates), len(urls))):
                url = urls[i]
                url_to_movements[url].append((coordinates[i], timestamps[i]))
            
            page_groups = list(url_to_movements.values())
        
        # Convert each page group to a matrix
        for page_group in page_groups:
            if len(page_group) < 2:  # Need at least 2 points to calculate dt
                continue
            
            # Create a matrix of zeros with the input shape
            matrix = np.zeros(self.input_shape[:2], dtype=np.float32)
            
            # Fill the matrix with dt values
            for i in range(len(page_group) - 1):
                coord_i, time_i = page_group[i]
                _, time_i_plus_1 = page_group[i + 1]
                
                # Extract x, y coordinates
                try:
                    x, y = map(int, coord_i)
                    
                    # Apply center cropping (300px offset from each edge)
                    x_cropped = max(0, min(x - 300, self.input_shape[1] - 1))
                    y_cropped = max(0, min(y - 300, self.input_shape[0] - 1))
                    
                    # Calculate dt (time spent at this point)
                    dt = float(time_i_plus_1) - float(time_i)
                    
                    # Set the value in the matrix
                    matrix[y_cropped, x_cropped] = dt
                except (ValueError, IndexError):
                    # Skip invalid coordinates
                    continue
            
            # Reshape to match input shape (add channel dimension)
            matrix = matrix.reshape(self.input_shape)
            
            matrices.append(matrix)
        
        return matrices
    
    def train(self, train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2):
        # Convert labels to one-hot encoding
        train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=2)
        self.model.fit(train_data, train_labels_one_hot, epochs=epochs, batch_size=batch_size,validation_split=validation_split)
    
    def predict(self, matrices):
        if not matrices:
            return []

        matrices_array = np.array(matrices)
        predictions = self.model.predict(matrices_array)
        return np.argmax(predictions, axis=1).tolist()
    
    def get_mouse_movement_score(self, mouse_data, phase='phase1'):
        # Preprocess mouse movements into matrices
        matrices = self._preprocess_mouse_movements(mouse_data, phase)
        
        if not matrices:
            return 0.5  # Neutral score if no matrices
        
        # Get predictions for each matrix
        predictions = self.predict(matrices)
        
        if not predictions:
            return 0.5  # Neutral score if no predictions
        
        # Calculate score using majority voting
        bot_count = sum(predictions)
        total_count = len(predictions)
        
        # Return the proportion of matrices classified as bot
        return bot_count / total_count
    
    def load_model(self, model_path):
        """
        Load a pre-trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(model_path)
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        self.model.save(model_path)

    def train_sequentially(self, phase1_data, phase1_labels, phase2_data=None, phase2_labels=None, 
                          phase1_epochs=10, phase2_epochs=5, batch_size=32, validation_split=0.2):
        """
        Train the model sequentially: first on phase1 data, then on phase2 data.
        
        Args:
            phase1_data: Training data for phase1
            phase1_labels: Labels for phase1 data
            phase2_data: Training data for phase2 (optional)
            phase2_labels: Labels for phase2 data (optional)
            phase1_epochs: Number of epochs for phase1 training
            phase2_epochs: Number of epochs for phase2 training
            batch_size: Batch size for training
            validation_split: Validation split ratio
        """
        print("=== PHASE 1 TRAINING ===")
        if phase1_data is not None and len(phase1_data) > 0:
            # Convert labels to one-hot encoding
            phase1_labels_one_hot = tf.keras.utils.to_categorical(phase1_labels, num_classes=2)
            self.model.fit(phase1_data, phase1_labels_one_hot, 
                          epochs=phase1_epochs, batch_size=batch_size, 
                          validation_split=validation_split)
            print(f"Phase 1 training completed with {len(phase1_data)} samples")
        else:
            print("No Phase 1 data provided, skipping Phase 1 training")
        
        print("\n=== PHASE 2 INCREMENTAL TRAINING ===")
        if phase2_data is not None and len(phase2_data) > 0:
            # Convert labels to one-hot encoding
            phase2_labels_one_hot = tf.keras.utils.to_categorical(phase2_labels, num_classes=2)
            self.model.fit(phase2_data, phase2_labels_one_hot, 
                          epochs=phase2_epochs, batch_size=batch_size, 
                          validation_split=validation_split)
            print(f"Phase 2 incremental training completed with {len(phase2_data)} samples")
        else:
            print("No Phase 2 data provided, skipping Phase 2 training")
        
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
        if test_data is None or test_data.size == 0:
            print(f"No {dataset_name} data provided")
            return 0.0
        
        # Convert labels to one-hot encoding
        test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=2)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(test_data, test_labels_one_hot, verbose=0)
        print(f"{dataset_name} Accuracy: {accuracy:.4f}")
        return accuracy

    def load_phase_data(self, dataset_base_path, phase='phase1', dataset_type='D1', data_type='train'):
        """
        Load data from a specific phase and dataset type.
        
        Args:
            dataset_base_path: Base path to the dataset
            phase: 'phase1' or 'phase2'
            dataset_type: 'D1' or 'D2'
            data_type: 'train' or 'test'
            
        Returns:
            tuple: (matrices, labels) - lists of matrices and corresponding labels
        """
        matrices = []
        labels = []
        
        if phase == 'phase1':
            # Phase 1 structure: dataset_base_path/D1/annotations/humans_and_moderate_bots/train
            if dataset_type == 'D1':
                annotations_path = os.path.join(dataset_base_path, 'D1', 'annotations', 'humans_and_moderate_bots', data_type)
                data_base_path = os.path.join(dataset_base_path, 'D1', 'data', 'mouse_movements', 'humans_and_moderate_bots')
            else:  # D2
                annotations_path = os.path.join(dataset_base_path, 'D2', 'annotations', 'humans_and_advanced_bots', data_type)
                data_base_path = os.path.join(dataset_base_path, 'D2', 'data', 'mouse_movements', 'humans_and_advanced_bots')
            
            print(f"Loading Phase 1 {dataset_type} {data_type} data...")
            
            # Load annotation file
            if os.path.exists(annotations_path):
                with open(annotations_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        try:
                            session_id, label_str = line.split()
                            label = 0 if label_str == 'human' else 1
                            
                            # Load corresponding JSON file from session directory
                            json_file_path = os.path.join(data_base_path, session_id, 'mouse_movements.json')
                            if os.path.exists(json_file_path):
                                session_matrices = self._preprocess_mouse_movement_data(json_file_path, session_id)
                                if session_matrices:
                                    matrices.extend(session_matrices)
                                    labels.extend([label] * len(session_matrices))
                        except ValueError as e:
                            print(f"Warning: Invalid line format in {annotations_path}: {line}")
                            continue
                        except Exception as e:
                            print(f"Warning: Error processing line in {annotations_path}: {e}")
                            continue
            else:
                print(f"Warning: Annotation file not found: {annotations_path}")
        
        elif phase == 'phase2':
            # Phase 2 structure: dataset_base_path/D1/data/mouse_movements/humans/mouse_movements_humans.json
            if dataset_type == 'D1':
                human_file = os.path.join(dataset_base_path, 'D1', 'data', 'mouse_movements', 'humans', 'mouse_movements_humans.json')
                bot_file = os.path.join(dataset_base_path, 'D1', 'data', 'mouse_movements', 'bots', 'mouse_movements_moderate_bots.json')
            else:  # D2
                human_file = os.path.join(dataset_base_path, 'D2', 'data', 'mouse_movements', 'humans', 'mouse_movements_humans.json')
                bot_file = os.path.join(dataset_base_path, 'D2', 'data', 'mouse_movements', 'bot', 'mouse_movements_advanced_bots.json')
            
            print(f"Loading Phase 2 {dataset_type} {data_type} data...")
            
            # Load human data
            human_matrices, human_labels = self._load_phase2_raw_data(human_file, 0, data_type)
            matrices.extend(human_matrices)
            labels.extend(human_labels)
            
            # Load bot data
            bot_matrices, bot_labels = self._load_phase2_raw_data(bot_file, 1, data_type)
            matrices.extend(bot_matrices)
            labels.extend(bot_labels)
        
        print(f"Loaded {len(matrices)} matrices with {len(labels)} labels from {phase} {dataset_type} {data_type}")
        return matrices, labels

    def _load_phase2_json_data(self, json_file_path, label, data_type='train'):
        """
        Load and preprocess Phase 2 JSON data (URL sequences).
        
        Args:
            json_file_path: Path to the JSON file
            label: Label for this data (0 for human, 1 for bot)
            data_type: 'train' or 'test'
            
        Returns:
            tuple: (matrices, labels) - lists of matrices and corresponding labels
        """
        matrices = []
        labels = []
        
        try:
            # Read the JSON file line by line to handle large files
            with open(json_file_path, 'r') as f:
                lines = f.readlines()
            
            # Split data into train/test if needed
            if data_type == 'train':
                # Use 80% for training
                split_index = int(len(lines) * 0.8)
                lines_to_process = lines[:split_index]
            else:  # test
                # Use 20% for testing
                split_index = int(len(lines) * 0.8)
                lines_to_process = lines[split_index:]
            
            print(f"Processing {len(lines_to_process)} lines for {data_type} from {len(lines)} total lines")
            
            processed_count = 0
            for i, line in enumerate(lines_to_process):
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    
                    # Extract URL sequence and other data
                    url_sequence = data.get('url_sequence', '')
                    unique_id = data.get('unique_id', '')
                    
                    # Preprocess URL sequence into matrix
                    session_matrices = self._preprocess_url_sequence(url_sequence, unique_id)
                    
                    if session_matrices:
                        matrices.extend(session_matrices)
                        labels.extend([label] * len(session_matrices))
                        processed_count += 1
                    
                    # Debug: print first few lines
                    if i < 3:
                        print(f"  Line {i}: URL sequence length = {len(url_sequence)}, Matrices generated = {len(session_matrices)}")
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON line {i}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {i}: {e}")
                    continue
            
            print(f"Successfully processed {processed_count} out of {len(lines_to_process)} lines")
                    
        except Exception as e:
            print(f"Error reading file {json_file_path}: {e}")
        
        return matrices, labels

    def _preprocess_url_sequence(self, url_sequence, unique_id):
        """
        Preprocess URL sequence data into matrices for Phase 2.
        
        Args:
            url_sequence: String containing URL sequence
            unique_id: Unique identifier for the session
            
        Returns:
            list: List of matrices
        """
        matrices = []
        
        if not url_sequence:
            return matrices
        
        try:
            # Split URL sequence into individual URLs
            urls = url_sequence.split('][')
            
            # Clean URLs (remove brackets and quotes)
            cleaned_urls = []
            for url in urls:
                url = url.strip('[]"')
                if url and url.startswith('http'):
                    cleaned_urls.append(url)
            
            if len(cleaned_urls) < 2:
                return matrices
            
            # Create a matrix representation of URL patterns
            matrix = np.zeros(self.input_shape[:2], dtype=np.float32)
            
            # Analyze URL patterns and transitions
            for i in range(len(cleaned_urls) - 1):
                current_url = cleaned_urls[i]
                next_url = cleaned_urls[i + 1]
                
                # Create a hash-based position in the matrix
                url_hash = hash(current_url) % (self.input_shape[0] * self.input_shape[1])
                row = url_hash // self.input_shape[1]
                col = url_hash % self.input_shape[1]
                
                # Ensure indices are within bounds
                row = max(0, min(row, self.input_shape[0] - 1))
                col = max(0, min(col, self.input_shape[1] - 1))
                
                # Set value based on transition pattern
                transition_value = 1.0 if current_url != next_url else 0.5
                matrix[row, col] = transition_value
            
            # Add some additional features based on URL patterns
            # Count different domains
            domains = set()
            for url in cleaned_urls:
                try:
                    domain = url.split('/')[2] if len(url.split('/')) > 2 else url
                    domains.add(domain)
                except:
                    continue
            
            # Add domain diversity as a feature
            domain_diversity = len(domains) / len(cleaned_urls) if cleaned_urls else 0
            diversity_row = int(domain_diversity * (self.input_shape[0] - 1))
            diversity_col = int(domain_diversity * (self.input_shape[1] - 1))
            matrix[diversity_row, diversity_col] = domain_diversity
            
            # Add sequence length as a feature
            seq_length_norm = min(len(cleaned_urls) / 100.0, 1.0)  # Normalize to 0-1
            length_row = int(seq_length_norm * (self.input_shape[0] - 1))
            length_col = int((1 - seq_length_norm) * (self.input_shape[1] - 1))
            matrix[length_row, length_col] = seq_length_norm
            
            # Reshape to match input shape (add channel dimension)
            matrix = matrix.reshape(self.input_shape)
            matrices.append(matrix)
            
        except Exception as e:
            print(f"Error preprocessing URL sequence for {unique_id}: {e}")
        
        return matrices

    def _load_phase2_raw_data(self, file_path, label, data_type='train'):
        """
        Load and preprocess Phase 2 raw URL sequence data.
        
        Args:
            file_path: Path to the file containing URL sequences
            label: Label for this data (0 for human, 1 for bot)
            data_type: 'train' or 'test'
            
        Returns:
            tuple: (matrices, labels) - lists of matrices and corresponding labels
        """
        matrices = []
        labels = []
        
        try:
            # Read the file line by line to handle large files
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Split data into train/test if needed
            if data_type == 'train':
                # Use 80% for training
                split_index = int(len(lines) * 0.8)
                lines_to_process = lines[:split_index]
            else:  # test
                # Use 20% for testing
                split_index = int(len(lines) * 0.8)
                lines_to_process = lines[split_index:]
            
            print(f"Processing {len(lines_to_process)} lines for {data_type} from {len(lines)} total lines")
            
            processed_count = 0
            for i, line in enumerate(lines_to_process):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Extract URL sequence from the line
                    # The line contains a long URL sequence, we need to parse it
                    url_sequence = line
                    unique_id = f"phase2_{i}"
                    
                    # Preprocess URL sequence into matrix
                    session_matrices = self._preprocess_url_sequence(url_sequence, unique_id)
                    
                    if session_matrices:
                        matrices.extend(session_matrices)
                        labels.extend([label] * len(session_matrices))
                        processed_count += 1
                    
                    # Debug: print first few lines
                    if i < 3:
                        print(f"  Line {i}: URL sequence length = {len(url_sequence)}, Matrices generated = {len(session_matrices)}")
                    
                except Exception as e:
                    print(f"Warning: Error processing line {i}: {e}")
                    continue
            
            print(f"Successfully processed {processed_count} out of {len(lines_to_process)} lines")
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
        
        return matrices, labels

    def _preprocess_mouse_movement_data(self, json_file_path, session_id):
        """
        Preprocess mouse movement data from Phase 1 JSON files.
        
        Args:
            json_file_path: Path to the JSON file
            session_id: Session identifier
            
        Returns:
            list: List of matrices
        """
        matrices = []
        
        try:
            with open(json_file_path, 'r') as f:
                mouse_data = json.load(f)
            
            # Preprocess mouse movements into matrices
            session_matrices = self._preprocess_mouse_movements(mouse_data, phase='phase1')
            matrices.extend(session_matrices)
            
        except Exception as e:
            print(f"Error preprocessing mouse movement data for {session_id}: {e}")
        
        return matrices

    def load_all_training_data(self, dataset_base_phase1, dataset_base_phase2):
        """
        Load all training data from both phases and all dataset types.
        
        Returns:
            dict: Dictionary containing all training data organized by phase and dataset
        """
        training_data = {
            'phase1': {
                'D1': {'matrices': [], 'labels': []},
                'D2': {'matrices': [], 'labels': []}
            },
            'phase2': {
                'D1': {'matrices': [], 'labels': []},
                'D2': {'matrices': [], 'labels': []}
            }
        }
        
        # Load Phase 1 training data
        print("=== Loading Phase 1 Training Data ===")
        for dataset_type in ['D1', 'D2']:
            matrices, labels = self.load_phase_data(dataset_base_phase1, 'phase1', dataset_type, 'train')
            training_data['phase1'][dataset_type]['matrices'] = matrices
            training_data['phase1'][dataset_type]['labels'] = labels
        
        # Load Phase 2 training data
        print("\n=== Loading Phase 2 Training Data ===")
        for dataset_type in ['D1', 'D2']:
            matrices, labels = self.load_phase_data(dataset_base_phase2, 'phase2', dataset_type, 'train')
            training_data['phase2'][dataset_type]['matrices'] = matrices
            training_data['phase2'][dataset_type]['labels'] = labels
        
        return training_data

    def load_all_testing_data(self, dataset_base_phase1, dataset_base_phase2):
        """
        Load all testing data from both phases and all dataset types.
        
        Returns:
            dict: Dictionary containing all testing data organized by phase and dataset
        """
        testing_data = {
            'phase1': {
                'D1': {'matrices': [], 'labels': []},
                'D2': {'matrices': [], 'labels': []}
            },
            'phase2': {
                'D1': {'matrices': [], 'labels': []},
                'D2': {'matrices': [], 'labels': []}
            }
        }
        
        # Load Phase 1 testing data
        print("=== Loading Phase 1 Testing Data ===")
        for dataset_type in ['D1', 'D2']:
            matrices, labels = self.load_phase_data(dataset_base_phase1, 'phase1', dataset_type, 'test')
            testing_data['phase1'][dataset_type]['matrices'] = matrices
            testing_data['phase1'][dataset_type]['labels'] = labels
        
        # Load Phase 2 testing data
        print("\n=== Loading Phase 2 Testing Data ===")
        for dataset_type in ['D1', 'D2']:
            matrices, labels = self.load_phase_data(dataset_base_phase2, 'phase2', dataset_type, 'test')
            testing_data['phase2'][dataset_type]['matrices'] = matrices
            testing_data['phase2'][dataset_type]['labels'] = labels
        
        return testing_data

    def process_session_data(self, session_id, data_dir, phase='phase1', dataset_type='D1'):
        """
        Process mouse movement data for a specific session.
        
        Args:
            session_id (str): PHP session ID
            data_dir (str): Directory containing the dataset (should point to phase1 folder)
            phase (str): 'phase1' or 'phase2' to specify which dataset format to use
            dataset_type (str): 'D1' for humans_and_moderate_bots, 'D2' for humans_and_advanced_bots
            
        Returns:
            float: Bot detection score for the session
        """
        mouse_data = None
        
        if phase == 'phase1':
            # In phase1, each session has its own folder with a mouse_movements.json file
            session_dir = os.path.join(data_dir, dataset_type, 'data', 'mouse_movements', 'humans_and_moderate_bots', session_id)
            json_path = os.path.join(session_dir, 'mouse_movements.json')
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    mouse_data = json.load(f)
        else:  # phase2
            # In phase2, data is in MongoDB exported JSON collections
            json_path = os.path.join(data_dir, f'{session_id}.json')
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    mouse_data = json.load(f)
        
        if mouse_data:
            return self.get_mouse_movement_score(mouse_data, phase)
        else:
            return 0.5  # Neutral score if no data found

# Example usage
if __name__ == "__main__":
    # Dataset paths
    dataset_base_phase1 = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase1'
    dataset_base_phase2 = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase2'

    # Initialize the mouse movement detector
    detector = MouseMovementDetectionBot()

    print("=== COMPREHENSIVE SEQUENTIAL TRAINING: ALL DATASETS ===")
    print("Training on all available datasets in sequential order...")

    # === PHASE 1 TRAINING (FIRST STAGE) ===
    print("\n" + "="*60)
    print("PHASE 1 TRAINING: D1 (Humans vs Moderate Bots)")
    print("="*60)
    
    # Load Phase 1 D1 training data
    phase1_d1_train_matrices, phase1_d1_train_labels = detector.load_phase_data(
        dataset_base_phase1, phase='phase1', dataset_type='D1', data_type='train'
    )
    
    if phase1_d1_train_matrices:
        phase1_d1_train_data = np.array(phase1_d1_train_matrices)
        phase1_d1_train_labels = np.array(phase1_d1_train_labels)
        
        print(f"Training on Phase 1 D1: {len(phase1_d1_train_matrices)} matrices")
        detector.train(phase1_d1_train_data, phase1_d1_train_labels, epochs=10, batch_size=32)
        print("Phase 1 D1 training completed.")
    else:
        print("No Phase 1 D1 training data found.")

    print("\n" + "="*60)
    print("PHASE 1 TRAINING: D2 (Humans vs Advanced Bots)")
    print("="*60)
    
    # Load Phase 1 D2 training data
    phase1_d2_train_matrices, phase1_d2_train_labels = detector.load_phase_data(
        dataset_base_phase1, phase='phase1', dataset_type='D2', data_type='train'
    )
    
    if phase1_d2_train_matrices:
        phase1_d2_train_data = np.array(phase1_d2_train_matrices)
        phase1_d2_train_labels = np.array(phase1_d2_train_labels)
        
        print(f"Training on Phase 1 D2: {len(phase1_d2_train_matrices)} matrices")
        detector.train(phase1_d2_train_data, phase1_d2_train_labels, epochs=10, batch_size=32)
        print("Phase 1 D2 training completed.")
    else:
        print("No Phase 1 D2 training data found.")

    # === PHASE 2 INCREMENTAL TRAINING (SECOND STAGE) ===
    print("\n" + "="*60)
    print("PHASE 2 INCREMENTAL TRAINING: D1 (Humans vs Moderate & Advanced Bots)")
    print("="*60)
    
    # Load Phase 2 D1 training data
    phase2_d1_train_matrices, phase2_d1_train_labels = detector.load_phase_data(
        dataset_base_phase2, phase='phase2', dataset_type='D1', data_type='train'
    )
    
    if phase2_d1_train_matrices:
        phase2_d1_train_data = np.array(phase2_d1_train_matrices)
        phase2_d1_train_labels = np.array(phase2_d1_train_labels)
        
        print(f"Training on Phase 2 D1: {len(phase2_d1_train_matrices)} matrices")
        detector.train(phase2_d1_train_data, phase2_d1_train_labels, epochs=5, batch_size=32)
        print("Phase 2 D1 training completed.")
    else:
        print("No Phase 2 D1 training data found.")

    print("\n" + "="*60)
    print("PHASE 2 INCREMENTAL TRAINING: D2 (Humans vs Advanced Bots)")
    print("="*60)
    
    # Load Phase 2 D2 training data
    phase2_d2_train_matrices, phase2_d2_train_labels = detector.load_phase_data(
        dataset_base_phase2, phase='phase2', dataset_type='D2', data_type='train'
    )
    
    if phase2_d2_train_matrices:
        phase2_d2_train_data = np.array(phase2_d2_train_matrices)
        phase2_d2_train_labels = np.array(phase2_d2_train_labels)
        
        print(f"Training on Phase 2 D2: {len(phase2_d2_train_matrices)} matrices")
        detector.train(phase2_d2_train_data, phase2_d2_train_labels, epochs=5, batch_size=32)
        print("Phase 2 D2 training completed.")
    else:
        print("No Phase 2 D2 training data found.")

    # Save final comprehensive model (single file)
    final_model_path = 'mouse_movement_detector_comprehensive.h5'
    detector.save_model(final_model_path)
    print(f"\nFinal comprehensive model saved to {final_model_path}")

    # === COMPREHENSIVE TESTING PHASE ===
    print("\n" + "="*60)
    print("COMPREHENSIVE TESTING ON ALL DATASETS")
    print("="*60)

    # Test on Phase 1 D1
    print("\n--- Testing on Phase 1 D1 (Humans vs Moderate Bots) ---")
    phase1_d1_test_matrices, phase1_d1_test_labels = detector.load_phase_data(
        dataset_base_phase1, phase='phase1', dataset_type='D1', data_type='test'
    )
    
    if phase1_d1_test_matrices:
        phase1_d1_test_data = np.array(phase1_d1_test_matrices)
        phase1_d1_test_labels = np.array(phase1_d1_test_labels)
        detector.evaluate_model(phase1_d1_test_data, phase1_d1_test_labels, "Phase 1 D1 Test")
    else:
        print("No Phase 1 D1 test data found.")

    # Test on Phase 1 D2
    print("\n--- Testing on Phase 1 D2 (Humans vs Advanced Bots) ---")
    phase1_d2_test_matrices, phase1_d2_test_labels = detector.load_phase_data(
        dataset_base_phase1, phase='phase1', dataset_type='D2', data_type='test'
    )
    
    if phase1_d2_test_matrices:
        phase1_d2_test_data = np.array(phase1_d2_test_matrices)
        phase1_d2_test_labels = np.array(phase1_d2_test_labels)
        detector.evaluate_model(phase1_d2_test_data, phase1_d2_test_labels, "Phase 1 D2 Test")
    else:
        print("No Phase 1 D2 test data found.")

    # Test on Phase 2 D1
    print("\n--- Testing on Phase 2 D1 (Humans vs Moderate & Advanced Bots) ---")
    phase2_d1_test_matrices, phase2_d1_test_labels = detector.load_phase_data(
        dataset_base_phase2, phase='phase2', dataset_type='D1', data_type='test'
    )
    
    if phase2_d1_test_matrices:
        phase2_d1_test_data = np.array(phase2_d1_test_matrices)
        phase2_d1_test_labels = np.array(phase2_d1_test_labels)
        detector.evaluate_model(phase2_d1_test_data, phase2_d1_test_labels, "Phase 2 D1 Test")
    else:
        print("No Phase 2 D1 test data found.")

    # Test on Phase 2 D2
    print("\n--- Testing on Phase 2 D2 (Humans vs Advanced Bots) ---")
    phase2_d2_test_matrices, phase2_d2_test_labels = detector.load_phase_data(
        dataset_base_phase2, phase='phase2', dataset_type='D2', data_type='test'
    )
    
    if phase2_d2_test_matrices:
        phase2_d2_test_data = np.array(phase2_d2_test_matrices)
        phase2_d2_test_labels = np.array(phase2_d2_test_labels)
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