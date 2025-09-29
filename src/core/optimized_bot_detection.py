#!/usr/bin/env python3
"""
Optimized Bot Detection Models for Fast Processing

This script provides a faster version of bot detection with:
- Model caching and singleton pattern
- Optimized preprocessing
- Reduced computational overhead
- Command line arguments for session-specific processing
"""

import os
import json
import numpy as np
import argparse
import time
import sys
from typing import Dict, List, Any
import threading
import warnings

# Suppress sklearn version compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()

class OptimizedBotDetector:
    """
    Optimized bot detection with model caching and fast processing.
    """
    
    def __init__(self):
        self.models_loaded = False
        self.web_log_detector = None
        self.mouse_movement_detector = None
        self.fusion = None
        
    def load_models_once(self):
        """Load models only once and cache them globally."""
        global _model_cache
        
        with _model_lock:
            if _model_cache.get('loaded', False):
                self.web_log_detector = _model_cache['web_log']
                self.mouse_movement_detector = _model_cache['mouse']
                self.fusion = _model_cache['fusion']
                self.models_loaded = True
                return True
            
            try:
                print("Loading models (this may take a few seconds)...")
                start_time = time.time()
                
                # Import modules (clear cache to ensure latest version)
                import sys
                modules_to_clear = ['web_log_detection_bot', 'mouse_movements_detection_bot', 'fusion']
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                from web_logs import WebLogDetectionBot
                from mouse_movements import MouseMovementDetectionBot
                from fusion import BotDetectionFusion
                
                # Initialize detectors
                self.web_log_detector = WebLogDetectionBot()
                self.mouse_movement_detector = MouseMovementDetectionBot()
                
                # Load models
                web_log_model_path = 'models/logs_model.pkl'
                mouse_movement_model_path = 'models/mouse_model.h5'
                
                if os.path.exists(web_log_model_path):
                    self.web_log_detector.load_model(web_log_model_path)
                    print(f"[SUCCESS] Web log model loaded")
                else:
                    print(f"[ERROR] Web log model not found at {web_log_model_path}")
                    return False
                    
                if os.path.exists(mouse_movement_model_path):
                    self.mouse_movement_detector.load_model(mouse_movement_model_path)
                    print(f"[SUCCESS] Mouse movement model loaded")
                else:
                    print(f"[ERROR] Mouse movement model not found at {mouse_movement_model_path}")
                    return False
                
                # Initialize fusion
                self.fusion = BotDetectionFusion(
                    web_log_model_path=web_log_model_path,
                    mouse_movement_model_path=mouse_movement_model_path
                )
                print(f"[SUCCESS] Fusion module initialized")
                
                # Cache models globally
                _model_cache['web_log'] = self.web_log_detector
                _model_cache['mouse'] = self.mouse_movement_detector
                _model_cache['fusion'] = self.fusion
                _model_cache['loaded'] = True
                
                load_time = time.time() - start_time
                print(f"[SUCCESS] All models loaded in {load_time:.2f}s")
                
                self.models_loaded = True
                return True
                
            except Exception as e:
                print(f"[ERROR] Error loading models: {e}")
                return False
    
    def fast_preprocess_mouse_movements(self, movements_data) -> np.ndarray:
        """
        Fast preprocessing of mouse movements with caching.
        
        Args:
            movements_data: List of movement objects or strings
            
        Returns:
            Preprocessed matrix for the CNN model
        """
        try:
            # Extract coordinates efficiently
            coordinates = []
            
            for movement in movements_data:
                if isinstance(movement, str):
                    # Handle string format like "m(x,y)"
                    if movement.startswith('m(') and movement.endswith(')'):
                        coords = movement[2:-1].split(',')
                        if len(coords) == 2:
                            try:
                                x, y = int(coords[0]), int(coords[1])
                                coordinates.append([x, y])
                            except ValueError:
                                continue
                elif isinstance(movement, dict):
                    # Handle object format like {"x": 100, "y": 200}
                    if 'x' in movement and 'y' in movement:
                        try:
                            x, y = int(movement['x']), int(movement['y'])
                            coordinates.append([x, y])
                        except (ValueError, TypeError):
                            continue
            
            if len(coordinates) < 5:  # Minimum required movements
                return None
            
            # Convert to numpy array
            coords_array = np.array(coordinates, dtype=np.float32)
            
            # Use original matrix size for compatibility (480x1320)
            matrix = np.zeros((480, 1320), dtype=np.float32)
            
            if coords_array.shape[0] > 0:
                x_coords = coords_array[:, 0]
                y_coords = coords_array[:, 1]
                
                # Normalize to original matrix size
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                if x_max > x_min and y_max > y_min:
                    x_norm = ((x_coords - x_min) / (x_max - x_min) * 1319).astype(int)
                    y_norm = ((y_coords - y_min) / (y_max - y_min) * 479).astype(int)
                    
                    # Populate matrix efficiently
                    for i in range(len(x_norm)):
                        if 0 <= x_norm[i] < 1320 and 0 <= y_norm[i] < 480:
                            matrix[y_norm[i], x_norm[i]] += 1
                    
                    # Normalize
                    max_val = matrix.max()
                    if max_val > 0:
                        matrix = matrix / max_val
            
            # Reshape for CNN input
            matrix = matrix.reshape(1, 480, 1320, 1)
            return matrix
            
        except Exception as e:
            print(f"Error in fast mouse preprocessing: {e}")
            return None
    
    def fast_preprocess_web_logs(self, web_logs: List[Dict]) -> Dict[str, Any]:
        """
        Fast preprocessing of web logs with minimal computation.
        
        Args:
            web_logs: List of web log entries
            
        Returns:
            Dictionary with extracted features
        """
        try:
            if not web_logs:
                return None
            
            # Use numpy for faster calculations
            total_requests = len(web_logs)
            status_codes = np.array([log.get('status', 0) for log in web_logs])
            bytes_array = np.array([log.get('bytes', 0) for log in web_logs])
            
            # Fast calculations
            total_bytes = np.sum(bytes_array)
            status_3xx = np.sum((status_codes >= 300) & (status_codes < 400))
            status_4xx = np.sum((status_codes >= 400) & (status_codes < 500))
            
            # Calculate percentages
            pct_3xx = (status_3xx / total_requests) * 100 if total_requests > 0 else 0
            pct_4xx = (status_4xx / total_requests) * 100 if total_requests > 0 else 0
            
            features = {
                'total_requests': total_requests,
                'total_bytes': total_bytes,
                'unique_methods': len(set(log.get('method', '') for log in web_logs)),
                'pct_3xx': pct_3xx,
                'pct_4xx': pct_4xx,
                'avg_bytes_per_request': total_bytes / total_requests if total_requests > 0 else 0
            }
            
            return features
            
        except Exception as e:
            print(f"Error in fast web log preprocessing: {e}")
            return None
    
    def detect_bot_fast(self, session_id: str = None) -> Dict[str, Any]:
        """
        Fast bot detection for a specific session.
        
        Args:
            session_id: Optional session ID to filter data
            
        Returns:
            Detection results
        """
        start_time = time.time()
        
        if not self.models_loaded:
            if not self.load_models_once():
                return {'error': 'Failed to load models'}
        
        logs_dir = 'login_page/src/logs'
        results = {
            'mouseMovements': [],
            'webLogs': None,
            'loginAttempts': [],
            'summary': {},
            'processingTime': 0,
            'behavior': None
        }
        
        try:
            # Load and process mouse movements
            mouse_file = os.path.join(logs_dir, 'mouse_movements.json')
            if os.path.exists(mouse_file):
                with open(mouse_file, 'r') as f:
                    mouse_data = json.load(f)
                
                for i, session_data in enumerate(mouse_data):
                    if session_id and session_data.get('sessionId') != session_id:
                        continue
                        
                    movements = session_data.get('movements', [])
                    if len(movements) < 5:
                        continue
                    
                    # Process movements directly
                    matrix = self.fast_preprocess_mouse_movements(movements)
                    
                    if matrix is not None:
                        try:
                            prediction = self.mouse_movement_detector.model.predict(matrix, verbose=0)
                            score = float(prediction[0][0])
                            
                            results['mouseMovements'].append({
                                'sessionId': session_data.get('sessionId', f'session_{i}'),
                                'movements': len(movements),
                                'botScore': score,
                                'classification': 'BOT' if score > 0.45 else 'HUMAN'
                            })
                        except Exception as e:
                            print(f"Mouse prediction error: {e}")
            
            # Load and process web logs
            web_log_file = os.path.join(logs_dir, 'web_logs.json')
            if os.path.exists(web_log_file):
                with open(web_log_file, 'r') as f:
                    web_logs_data = json.load(f)
                
                # Filter by session if specified
                if session_id:
                    web_logs_data = [log for log in web_logs_data if log.get('sessionId') == session_id]
                
                if web_logs_data:
                    features = self.fast_preprocess_web_logs(web_logs_data)
                    if features is not None:
                        # Create feature vector with all 19 features
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
                            prediction = self.web_log_detector.model.predict_proba(feature_vector)
                            score = float(prediction[0][1])
                            
                            results['webLogs'] = {
                                'sessionId': session_id or 'unknown',
                                'entries': len(web_logs_data),
                                'botScore': score,
                                'classification': 'BOT' if score > 0.45 else 'HUMAN'
                            }
                        except Exception as e:
                            print(f"Web log prediction error: {e}")
            
            # Load behavior signals if present and derive penalties
            behavior_file = os.path.join(logs_dir, 'behavior.json')
            behavior_penalty = 0.0
            if os.path.exists(behavior_file):
                with open(behavior_file, 'r') as f:
                    behavior_data = json.load(f)
                # Filter for session
                if session_id:
                    behavior_data = [b for b in behavior_data if b.get('sessionId') == session_id]
                if behavior_data:
                    # Use latest snapshot
                    b = behavior_data[-1]
                    avg_key = b.get('avgKeyInterval') or 0
                    untrusted_ratio = b.get('untrustedClickRatio') or 0
                    scroll_var = b.get('scrollVariance') or 0
                    focus_blur = b.get('focusBlurCount') or 0

                    # Heuristics: high untrusted clicks strongly penalize, ultra-low key interval penalize, extremely low scroll variance penalize
                    if untrusted_ratio > 0.2:
                        behavior_penalty += min(0.25, untrusted_ratio)
                    if avg_key > 0 and avg_key < 35:  # very fast typing (ms)
                        behavior_penalty += 0.15
                    if scroll_var is not None and scroll_var < 5:
                        behavior_penalty += 0.1
                    if focus_blur > 4:
                        behavior_penalty += 0.1

                    results['behavior'] = {
                        'avgKeyInterval': avg_key,
                        'untrustedClickRatio': untrusted_ratio,
                        'scrollVariance': scroll_var,
                        'focusBlurCount': focus_blur,
                        'penalty': round(behavior_penalty, 4)
                    }

            # Load honeypot data if present
            honeypot_file = os.path.join(logs_dir, 'honeypot.json')
            honeypot_data = None
            if os.path.exists(honeypot_file):
                with open(honeypot_file, 'r') as f:
                    honeypot_data = json.load(f)
                # Filter for session
                if session_id:
                    honeypot_data = [h for h in honeypot_data if h.get('sessionId') == session_id]
                if honeypot_data:
                    # Use latest honeypot data
                    honeypot_data = honeypot_data[-1]
                    results['honeypot'] = honeypot_data

            # Process login attempts
            login_file = os.path.join(logs_dir, 'login_attempts.json')
            if os.path.exists(login_file):
                with open(login_file, 'r') as f:
                    login_data = json.load(f)
                
                for i, attempt in enumerate(login_data):
                    if session_id and attempt.get('sessionId') != session_id:
                        continue
                    
                    mouse_data = attempt.get('details', {}).get('mouseMovementData', [])
                    if len(mouse_data) < 5:
                        continue
                    
                    # Fast mouse movement processing
                    matrix = self.fast_preprocess_mouse_movements(mouse_data)
                    
                    if matrix is not None:
                        try:
                            mouse_prediction = self.mouse_movement_detector.model.predict(matrix, verbose=0)
                            mouse_score = float(mouse_prediction[0][0])
                            web_score = 0.3  # Default score
                            
                            # Use fusion for final classification
                            # Apply behavior penalty by increasing mouse_score
                            penalized_mouse = min(1.0, mouse_score + behavior_penalty)
                            
                            # Calculate honeypot score
                            honeypot_score = 0.0
                            if honeypot_data:
                                honeypot_score = self.fusion.calculate_honeypot_score(honeypot_data)
                            
                            fusion_score = self.fusion.fuse_scores(penalized_mouse, web_score, honeypot_score)
                            final_classification = self.fusion.classify_session(penalized_mouse, web_score, honeypot_score)
                            
                            results['loginAttempts'].append({
                                'attemptNumber': i + 1,
                                'mouseScore': penalized_mouse,
                                'webScore': web_score,
                                'honeypotScore': honeypot_score,
                                'fusionScore': fusion_score,
                                'classification': 'BOT' if final_classification[0] else 'HUMAN'
                            })
                        except Exception as e:
                            print(f"Login attempt prediction error: {e}")
            
            # Calculate summary
            if results['mouseMovements']:
                bot_count = sum(1 for r in results['mouseMovements'] if r['classification'] == 'BOT')
                results['summary']['mouseMovements'] = {
                    'sessions': len(results['mouseMovements']),
                    'botsDetected': bot_count
                }
            
            if results['webLogs']:
                results['summary']['webLogs'] = {
                    'sessions': 1,
                    'botsDetected': 1 if results['webLogs']['classification'] == 'BOT' else 0
                }
            
            if results['loginAttempts']:
                bot_count = sum(1 for r in results['loginAttempts'] if r['classification'] == 'BOT')
                results['summary']['loginAttempts'] = {
                    'attempts': len(results['loginAttempts']),
                    'botsDetected': bot_count
                }
            
            processing_time = time.time() - start_time
            results['processingTime'] = processing_time
            print(f"Processing time: {processing_time:.3f}s")
            
            return results
            
        except Exception as e:
            print(f"Error in fast bot detection: {e}")
            return {'error': str(e)}

def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description='Optimized Bot Detection')
    parser.add_argument('--fast', action='store_true', help='Enable fast mode')
    parser.add_argument('--session-id', type=str, help='Process specific session ID')
    parser.add_argument('--timeout', type=int, default=15, help='Processing timeout in seconds')
    
    args = parser.parse_args()
    
    # Set timeout (Windows compatible)
    if args.timeout > 0:
        import signal
        import platform
        
        if platform.system() != 'Windows':
            def timeout_handler(signum, frame):
                print("Processing timeout reached")
                sys.exit(1)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(args.timeout)
        else:
            print(f"Timeout set to {args.timeout}s (Windows - no signal alarm)")
    
    try:
        detector = OptimizedBotDetector()
        results = detector.detect_bot_fast(args.session_id)
        
        # Print results in the expected format
        if 'error' in results:
            print(f"ERROR: {results['error']}")
            sys.exit(1)
        
        # Print mouse movement results
        for session in results['mouseMovements']:
            print(f"Session {session['sessionId']}: Mouse movements: {session['movements']}")
            print(f"Bot Score: {session['botScore']:.4f}")
            print(f"Classification: {session['classification']}")
        
        # Print web log results
        if results['webLogs']:
            print(f"Session: {results['webLogs']['sessionId']}")
            print(f"Log entries: {results['webLogs']['entries']}")
            print(f"Bot Score: {results['webLogs']['botScore']:.4f}")
            print(f"Classification: {results['webLogs']['classification']}")
        
        # Print login attempt results
        for attempt in results['loginAttempts']:
            print(f"Login Attempt {attempt['attemptNumber']}")
            print(f"Mouse Score: {attempt['mouseScore']:.4f}")
            print(f"Web Score: {attempt['webScore']:.4f}")
            print(f"Honeypot Score: {attempt.get('honeypotScore', 0.0):.4f}")
            print(f"Fusion Score: {attempt['fusionScore']:.4f}")
            print(f"Final Classification: {attempt['classification']}")
        
        # Print summary
        if results['summary'].get('mouseMovements'):
            summary = results['summary']['mouseMovements']
            print(f"Mouse Movements: {summary['sessions']} sessions, {summary['botsDetected']} detected as bots")
        
        if results['summary'].get('webLogs'):
            summary = results['summary']['webLogs']
            print(f"Web Logs: {summary['sessions']} sessions, {summary['botsDetected']} detected as bots")
        
        if results['summary'].get('loginAttempts'):
            summary = results['summary']['loginAttempts']
            print(f"Login Attempts: {summary['attempts']} attempts, {summary['botsDetected']} detected as bots")
        
        print(f"Processing time: {results['processingTime']:.3f}s")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
