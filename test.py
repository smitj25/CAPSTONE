#!/usr/bin/env python3
"""
Test Bot Detection Models on Log Files

This script tests the trained bot detection models on the 4 JSON log files
from the login_page/src/logs directory.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import sys

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_log_detection_bot import WebLogDetectionBot
from mouse_movements_detection_bot import MouseMovementDetectionBot
from fusion import BotDetectionFusion

class LogFileTester:
    """
    Test bot detection models on log files from the login page application.
    """
    
    def __init__(self, 
                 logs_dir: str = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/login_page/src/logs',
                 web_log_model_path: str = 'models/web_log_detector_comprehensive.pkl',
                 mouse_movement_model_path: str = 'models/mouse_movement_detector_comprehensive.h5'):
        """
        Initialize the log file tester.
        
        Args:
            logs_dir: Directory containing the log files
            web_log_model_path: Path to the trained web log detection model
            mouse_movement_model_path: Path to the trained mouse movement detection model
        """
        self.logs_dir = logs_dir
        self.web_log_model_path = web_log_model_path
        self.mouse_movement_model_path = mouse_movement_model_path
        
        # Initialize detection modules
        self.web_log_detector = WebLogDetectionBot()
        self.mouse_movement_detector = MouseMovementDetectionBot()
        self.fusion = BotDetectionFusion()
        
        # Load trained models
        self._load_models()
    
    def _load_models(self):
        """Load the trained models."""
        try:
            if os.path.exists(self.web_log_model_path):
                self.web_log_detector.load_model(self.web_log_model_path)
                print(f"✅ Loaded web log model from {self.web_log_model_path}")
            else:
                print(f"❌ Web log model not found at {self.web_log_model_path}")
                return False
                
            if os.path.exists(self.mouse_movement_model_path):
                self.mouse_movement_detector.load_model(self.mouse_movement_model_path)
                print(f"✅ Loaded mouse movement model from {self.mouse_movement_model_path}")
            else:
                print(f"❌ Mouse movement model not found at {self.mouse_movement_model_path}")
                return False
                
            # Initialize fusion with loaded models
            self.fusion = BotDetectionFusion(
                web_log_model_path=self.web_log_model_path,
                mouse_movement_model_path=self.mouse_movement_model_path
            )
            print("✅ Fusion module initialized")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def load_log_file(self, filename: str) -> Dict[str, Any]:
        """Load a JSON log file."""
        filepath = os.path.join(self.logs_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded {filename} ({len(data) if isinstance(data, list) else 'object'} entries)")
            return data
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def preprocess_mouse_movements(self, movements_data: List[str]) -> np.ndarray:
        """
        Preprocess mouse movement data from the log format to matrix format.
        
        Args:
            movements_data: List of mouse movement strings like ["m(218,4)", "m(366,132)", ...]
            
        Returns:
            Preprocessed matrix for the CNN model
        """
        try:
            # Extract coordinates from movement strings
            coordinates = []
            for movement in movements_data:
                if movement.startswith('m(') and movement.endswith(')'):
                    coords = movement[2:-1].split(',')
                    if len(coords) == 2:
                        x, y = int(coords[0]), int(coords[1])
                        coordinates.append([x, y])
            
            if len(coordinates) < 2:
                print("⚠️  Not enough mouse movement coordinates")
                return None
            
            # Convert to numpy array
            coords_array = np.array(coordinates)
            
            # Create a matrix with the expected input shape (480, 1320, 1)
            matrix = np.zeros((480, 1320))
            
            # Normalize coordinates to fit in the matrix
            if coords_array.shape[0] > 0:
                x_coords = coords_array[:, 0]
                y_coords = coords_array[:, 1]
                
                # Normalize to fit the matrix dimensions
                x_norm = ((x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * 1319).astype(int)
                y_norm = ((y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * 479).astype(int)
                
                # Populate matrix with movement intensity
                for i in range(len(x_norm)):
                    if 0 <= x_norm[i] < 1320 and 0 <= y_norm[i] < 480:
                        matrix[y_norm[i], x_norm[i]] += 1
                
                # Normalize the matrix
                if matrix.max() > 0:
                    matrix = matrix / matrix.max()
            
            # Reshape for CNN input (add channel dimension)
            matrix = matrix.reshape(1, 480, 1320, 1)
            return matrix
            
        except Exception as e:
            print(f"❌ Error preprocessing mouse movements: {e}")
            return None
    
    def preprocess_web_logs(self, web_logs: List[Dict]) -> Dict[str, Any]:
        """
        Preprocess web log data for the ensemble classifier.
        
        Args:
            web_logs: List of web log entries
            
        Returns:
            Dictionary with extracted features
        """
        try:
            if not web_logs:
                return None
            
            # Extract basic features
            total_requests = len(web_logs)
            unique_methods = set(log.get('method', '') for log in web_logs)
            total_bytes = sum(log.get('bytes', 0) for log in web_logs)
            
            # Count status codes
            status_3xx = sum(1 for log in web_logs if 300 <= log.get('status', 0) < 400)
            status_4xx = sum(1 for log in web_logs if 400 <= log.get('status', 0) < 500)
            
            # Calculate percentages
            pct_3xx = (status_3xx / total_requests) * 100 if total_requests > 0 else 0
            pct_4xx = (status_4xx / total_requests) * 100 if total_requests > 0 else 0
            
            # Extract features similar to what the web log detector expects
            features = {
                'total_requests': total_requests,
                'total_bytes': total_bytes,
                'unique_methods': len(unique_methods),
                'pct_3xx': pct_3xx,
                'pct_4xx': pct_4xx,
                'avg_bytes_per_request': total_bytes / total_requests if total_requests > 0 else 0
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Error preprocessing web logs: {e}")
            return None
    
    def test_mouse_movements_file(self, filename: str = 'mouse_movements.json'):
        """Test mouse movement detection on the mouse movements log file."""
        print(f"\n{'='*60}")
        print(f"TESTING MOUSE MOVEMENTS: {filename}")
        print(f"{'='*60}")
        
        data = self.load_log_file(filename)
        if not data:
            return
        
        results = []
        
        for i, session_data in enumerate(data):
            session_id = session_data.get('sessionId', f'session_{i}')
            movements = session_data.get('movements', [])
            
            print(f"\n📊 Session {i+1}: {session_id}")
            print(f"   Mouse movements: {len(movements)}")
            
            if len(movements) < 10:
                print("   ⚠️  Too few movements for reliable detection")
                continue
            
            # Preprocess mouse movements
            matrix = self.preprocess_mouse_movements(movements)
            if matrix is None:
                continue
            
            # Predict using mouse movement model
            try:
                prediction = self.mouse_movement_detector.model.predict(matrix, verbose=0)
                score = float(prediction[0][0])
                is_bot = score > 0.5
                
                results.append({
                    'session_id': session_id,
                    'score': score,
                    'is_bot': is_bot,
                    'movements_count': len(movements)
                })
                
                print(f"   🎯 Bot Score: {score:.4f}")
                print(f"   🤖 Classification: {'BOT' if is_bot else 'HUMAN'}")
                
            except Exception as e:
                print(f"   ❌ Prediction error: {e}")
        
        return results
    
    def test_web_logs_file(self, filename: str = 'web_logs.json'):
        """Test web log detection on the web logs file."""
        print(f"\n{'='*60}")
        print(f"TESTING WEB LOGS: {filename}")
        print(f"{'='*60}")
        
        data = self.load_log_file(filename)
        if not data:
            return
        
        # Group by session ID
        sessions = {}
        for log_entry in data:
            session_id = log_entry.get('sessionId', 'unknown')
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(log_entry)
        
        results = []
        
        for session_id, session_logs in sessions.items():
            print(f"\n📊 Session: {session_id}")
            print(f"   Log entries: {len(session_logs)}")
            
            # Preprocess web logs
            features = self.preprocess_web_logs(session_logs)
            if features is None:
                continue
            
            # Convert features to format expected by the model (19 features)
            # Create a feature vector with all expected features, using defaults for missing ones
            feature_vector = np.array([
                features['total_requests'],                    # total_requests
                features['total_bytes'],                       # total_session_bytes
                features.get('http_get_requests', 0),          # http_get_requests
                features.get('http_post_requests', 0),         # http_post_requests
                features.get('http_head_requests', 0),         # http_head_requests
                features['pct_3xx'],                          # percent_http_3xx_requests
                features['pct_4xx'],                          # percent_http_4xx_requests
                features.get('percent_image_requests', 0),     # percent_image_requests
                features.get('percent_css_requests', 0),       # percent_css_requests
                features.get('percent_js_requests', 0),        # percent_js_requests
                features.get('html_to_image_ratio', 0),        # html_to_image_ratio
                features.get('depth_sd', 0),                   # depth_sd
                features.get('max_requests_per_page', 0),      # max_requests_per_page
                features.get('avg_requests_per_page', 0),      # avg_requests_per_page
                features.get('max_consecutive_sequential_requests', 0),  # max_consecutive_sequential_requests
                features.get('percent_consecutive_sequential_requests', 0),  # percent_consecutive_sequential_requests
                features.get('session_time', 0),               # session_time
                features.get('browse_speed', 0),               # browse_speed
                features.get('sd_inter_request_times', 0)      # sd_inter_request_times
            ]).reshape(1, -1)
            
            # Predict using web log model
            try:
                # Access the trained model
                if hasattr(self.web_log_detector, 'model') and self.web_log_detector.model is not None:
                    prediction = self.web_log_detector.model.predict_proba(feature_vector)
                else:
                    print("   ❌ No trained model found")
                    continue
                score = float(prediction[0][1])  # Probability of being a bot
                is_bot = score > 0.5
                
                results.append({
                    'session_id': session_id,
                    'score': score,
                    'is_bot': is_bot,
                    'log_entries': len(session_logs)
                })
                
                print(f"   🎯 Bot Score: {score:.4f}")
                print(f"   🤖 Classification: {'BOT' if is_bot else 'HUMAN'}")
                print(f"   📈 Features: {features}")
                
            except Exception as e:
                print(f"   ❌ Prediction error: {e}")
        
        return results
    
    def test_login_attempts_file(self, filename: str = 'login_attempts.json'):
        """Test both models on login attempts file."""
        print(f"\n{'='*60}")
        print(f"TESTING LOGIN ATTEMPTS: {filename}")
        print(f"{'='*60}")
        
        data = self.load_log_file(filename)
        if not data:
            return
        
        results = []
        
        for i, attempt in enumerate(data):
            print(f"\n📊 Login Attempt {i+1}")
            
            # Extract mouse movements
            mouse_data = attempt.get('details', {}).get('mouseMovementData', [])
            timestamp = attempt.get('timestamp', 'unknown')
            success = attempt.get('details', {}).get('success', False)
            
            print(f"   Timestamp: {timestamp}")
            print(f"   Success: {success}")
            print(f"   Mouse movements: {len(mouse_data)}")
            
            if len(mouse_data) < 10:
                print("   ⚠️  Too few movements for reliable detection")
                continue
            
            # Test mouse movement detection
            matrix = self.preprocess_mouse_movements(mouse_data)
            if matrix is None:
                continue
            
            try:
                mouse_prediction = self.mouse_movement_detector.model.predict(matrix, verbose=0)
                mouse_score = float(mouse_prediction[0][0])
                
                # For web logs, we'll use a simple heuristic based on the attempt data
                # In a real scenario, you'd have actual web logs for this session
                web_score = 0.3  # Default score for login attempts
                
                # Use fusion to combine scores
                fusion_score = self.fusion.fuse_scores(mouse_score, web_score)
                final_classification = self.fusion.classify_session(mouse_score, web_score)
                
                results.append({
                    'attempt_id': i+1,
                    'timestamp': timestamp,
                    'success': success,
                    'mouse_score': mouse_score,
                    'web_score': web_score,
                    'fusion_score': fusion_score,
                    'final_classification': final_classification
                })
                
                print(f"   🎯 Mouse Score: {mouse_score:.4f}")
                print(f"   🌐 Web Score: {web_score:.4f}")
                print(f"   🔗 Fusion Score: {fusion_score:.4f}")
                is_bot, final_score = final_classification
                print(f"   🤖 Final Classification: {'BOT' if is_bot else 'HUMAN'} (Score: {final_score:.4f})")
                
            except Exception as e:
                print(f"   ❌ Prediction error: {e}")
        
        return results
    
    def test_events_file(self, filename: str = 'events.json'):
        """Test the events file (combined data)."""
        print(f"\n{'='*60}")
        print(f"TESTING EVENTS: {filename}")
        print(f"{'='*60}")
        
        data = self.load_log_file(filename)
        if not data:
            return
        
        # Check if it's the combined format
        if isinstance(data, dict):
            mouse_movements = data.get('mouseMovements', [])
            web_logs = data.get('webLogs', [])
            
            print(f"   Mouse movements: {len(mouse_movements)}")
            print(f"   Web logs: {len(web_logs)}")
            
            if len(mouse_movements) == 0:
                print("   ⚠️  No mouse movements in events file")
                return
            
            # Test mouse movements
            matrix = self.preprocess_mouse_movements(mouse_movements)
            if matrix is not None:
                try:
                    mouse_prediction = self.mouse_movement_detector.model.predict(matrix, verbose=0)
                    mouse_score = float(mouse_prediction[0][0])
                    print(f"   🎯 Mouse Score: {mouse_score:.4f}")
                except Exception as e:
                    print(f"   ❌ Mouse prediction error: {e}")
            
            if len(web_logs) > 0:
                features = self.preprocess_web_logs(web_logs)
                if features is not None:
                    # Create a feature vector with all expected features (19 features)
                    feature_vector = np.array([
                        features['total_requests'],                    # total_requests
                        features['total_bytes'],                       # total_session_bytes
                        features.get('http_get_requests', 0),          # http_get_requests
                        features.get('http_post_requests', 0),         # http_post_requests
                        features.get('http_head_requests', 0),         # http_head_requests
                        features['pct_3xx'],                          # percent_http_3xx_requests
                        features['pct_4xx'],                          # percent_http_4xx_requests
                        features.get('percent_image_requests', 0),     # percent_image_requests
                        features.get('percent_css_requests', 0),       # percent_css_requests
                        features.get('percent_js_requests', 0),        # percent_js_requests
                        features.get('html_to_image_ratio', 0),        # html_to_image_ratio
                        features.get('depth_sd', 0),                   # depth_sd
                        features.get('max_requests_per_page', 0),      # max_requests_per_page
                        features.get('avg_requests_per_page', 0),      # avg_requests_per_page
                        features.get('max_consecutive_sequential_requests', 0),  # max_consecutive_sequential_requests
                        features.get('percent_consecutive_sequential_requests', 0),  # percent_consecutive_sequential_requests
                        features.get('session_time', 0),               # session_time
                        features.get('browse_speed', 0),               # browse_speed
                        features.get('sd_inter_request_times', 0)      # sd_inter_request_times
                    ]).reshape(1, -1)
                    
                    try:
                        # Access the trained model
                        if hasattr(self.web_log_detector, 'model') and self.web_log_detector.model is not None:
                            web_prediction = self.web_log_detector.model.predict_proba(feature_vector)
                            web_score = float(web_prediction[0][1])
                            print(f"   🌐 Web Score: {web_score:.4f}")
                        else:
                            print("   ❌ No trained model found")
                    except Exception as e:
                        print(f"   ❌ Web prediction error: {e}")
    
    def run_all_tests(self):
        """Run tests on all 4 log files."""
        print("🚀 STARTING BOT DETECTION TESTS ON LOG FILES")
        print("="*80)
        
        # Test each file
        mouse_results = self.test_mouse_movements_file()
        web_results = self.test_web_logs_file()
        login_results = self.test_login_attempts_file()
        events_results = self.test_events_file()
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 TEST SUMMARY")
        print(f"{'='*80}")
        
        if mouse_results:
            bot_count = sum(1 for r in mouse_results if r['is_bot'])
            print(f"Mouse Movements: {len(mouse_results)} sessions, {bot_count} detected as bots")
        
        if web_results:
            bot_count = sum(1 for r in web_results if r['is_bot'])
            print(f"Web Logs: {len(web_results)} sessions, {bot_count} detected as bots")
        
        if login_results:
            bot_count = sum(1 for r in login_results if r['final_classification'][0] == True)
            print(f"Login Attempts: {len(login_results)} attempts, {bot_count} detected as bots")
        
        print(f"\n✅ All tests completed!")

def main():
    """Main function to run the log file tests."""
    # Initialize tester
    tester = LogFileTester()
    
    # Run all tests
    tester.run_all_tests()

if __name__ == "__main__":
    main()
