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
        """
        Initialize the Mouse Movement Detection Bot with the CNN model architecture.
        """
        self.model = None
        self.input_shape = (480, 1320, 1)  # Cropped from 1080x1920 with 300px offset
        self._build_model()
        
    def _build_model(self):
        """
        Build the CNN model with the architecture specified in the requirements.
        
        Architecture:
        - InputLayer: Output Shape (480, 1320, 1)
        - Conv (1): Kernel size/stride 3x3/2, Output Shape (239, 659, 64), Activation ReLU
        - Conv (2): Kernel size/stride 3x3/2, Output Shape (119, 329, 64), Activation ReLU
        - M-Pool (1): Kernel size/stride 4x4/4, Output Shape (29, 82, 64)
        - Conv (3): Kernel size/stride 3x3/2, Output Shape (14, 41, 64), Activation ReLU
        - M-Pool (2): Kernel size/stride 4x4/4, Output Shape (3, 10, 64)
        - Flatten: Output Shape (1920)
        - Dense: Output Shape (2), Activation Softmax
        """
        self.model = Sequential()
        
        # Input layer - use Input instead of InputLayer
        self.model.add(Input(shape=self.input_shape))
        
        # Conv (1): 3x3 kernel, stride 2, 64 filters, ReLU activation
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        
        # Conv (2): 3x3 kernel, stride 2, 64 filters, ReLU activation
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        
        # M-Pool (1): 4x4 kernel, stride 4
        self.model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
        
        # Conv (3): 3x3 kernel, stride 2, 64 filters, ReLU activation
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        
        # M-Pool (2): 4x4 kernel, stride 4
        self.model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
        
        # Flatten
        self.model.add(Flatten())
        
        # Dense output layer: 2 units (bot or human), softmax activation
        self.model.add(Dense(2, activation='softmax'))
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _preprocess_mouse_movements(self, mouse_data, phase='phase1'):
        """
        Preprocess mouse movement data into matrices for classification.
        
        Args:
            mouse_data (dict): Mouse movement data in the format specified by the dataset
            phase (str): 'phase1' or 'phase2' to specify which dataset format to use
            
        Returns:
            list: List of matrices representing mouse movements per page
        """
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
        """
        Train the CNN model on preprocessed mouse movement matrices.
        
        Args:
            train_data (numpy.ndarray): Training data matrices
            train_labels (numpy.ndarray): Training labels (0 for human, 1 for bot)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
        """
        # Convert labels to one-hot encoding
        train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=2)
        
        # Train the model
        self.model.fit(
            train_data,
            train_labels_one_hot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
    
    def predict(self, matrices):
        """
        Predict whether each matrix represents a bot or human.
        
        Args:
            matrices (list): List of preprocessed mouse movement matrices
            
        Returns:
            list: List of predictions (0 for human, 1 for bot) for each matrix
        """
        if not matrices:
            return []
        
        # Convert list to numpy array
        matrices_array = np.array(matrices)
        
        # Get predictions
        predictions = self.model.predict(matrices_array)
        
        # Return the class with highest probability for each matrix
        return np.argmax(predictions, axis=1).tolist()
    
    def get_mouse_movement_score(self, mouse_data, phase='phase1'):
        """
        Process mouse movement data and return a bot detection score.
        Uses majority voting across all matrices within a session.
        
        Args:
            mouse_data (dict): Mouse movement data in the format specified by the dataset
            phase (str): 'phase1' or 'phase2' to specify which dataset format to use
            
        Returns:
            float: Score between 0 and 1 indicating likelihood of being a bot
                  (0 = human, 1 = bot)
        """
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
    # Point to the base 'phase1' folder
    dataset_base_phase1 = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase1'
    # Point to the base 'phase2' folder if needed
    '''dataset_base_phase2 = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase2'''

    # Initialize the mouse movement detector
    detector = MouseMovementDetectionBot()

    # --- Prepare Training Data for D1 (Humans vs Moderate Bots) ---
    train_matrices_list = []
    train_labels_list = []

    # Path to D1 annotations train file
    annotations_d1_train = os.path.join(dataset_base_phase1, 'annotations', 'humans_and_moderate_bots', 'train')

    with open(annotations_d1_train, 'r') as f:
        for line in f:
            session_id, label_str = line.strip().split()
            
            # Map string label to numerical label (0 for human, 1 for bot)
            label = 0 if label_str == 'human' else 1 # Assuming 'moderate bot' or 'advanced bot' are 1
            
            # Load mouse movement data for this session
            mouse_data_path = os.path.join(dataset_base_phase1, 'data', 'mouse_movements', 'humans_and_moderate_bots', session_id, 'mouse_movements.json')
            
            if os.path.exists(mouse_data_path):
                with open(mouse_data_path, 'r') as mouse_f:
                    mouse_data = json.load(mouse_f)
                    
                # Preprocess mouse movements into matrices
                matrices = detector._preprocess_mouse_movements(mouse_data, phase='phase1')
                
                # Extend the list with new matrices and their corresponding labels
                train_matrices_list.extend(matrices)
                train_labels_list.extend([label] * len(matrices))
            else:
                print(f"Warning: Mouse movement data not found for session {session_id} at {mouse_data_path}")

    # Convert lists to numpy arrays for training
    train_data_array = np.array(train_matrices_list)
    train_labels_array = np.array(train_labels_list)
    
    if train_data_array.size > 0:
        print(f"Training on {len(train_matrices_list)} matrices with {len(train_labels_list)} labels.")
        # Complete the detector.train command
        detector.train(train_data_array, train_labels_array, epochs=10, batch_size=32)
    else:
        print("No training data generated. Skipping model training.")

    # Example: Process sessions for D1 (humans vs moderate bots)
    # This part would typically use the 'test' dataset, but using 'train' for demonstration consistency
    annotations_d1_test_demo = os.path.join(dataset_base_phase1, 'annotations', 'humans_and_moderate_bots', 'test') # Changed to test for more realistic demo
    print(f"\n--- Testing on D1 (Humans vs Moderate Bots) ---")
    with open(annotations_d1_test_demo, 'r') as f:
        for line in f:
            session_id, label = line.strip().split()
            score = detector.process_session_data(session_id, dataset_base_phase1, phase='phase1', dataset_type='D1')
            print(f"Session {session_id} (Actual: {label}): Bot score = {score:.4f}")
    
    # Example: Save the trained model
    detector.save_model('mouse_movement_detector.h5')
    print("\nModel saved to mouse_movement_detector.h5")