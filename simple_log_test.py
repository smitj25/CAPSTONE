#!/usr/bin/env python3
"""
Simple Log File Test

A simplified version that tests the bot detection models on the log files
with better error handling and simpler preprocessing.
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

class SimpleLogTester:
    """
    Simple test bot detection models on log files.
    """
    
    def __init__(self, 
                 logs_dir: str = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/login_page/src/logs',
                 web_log_model_path: str = 'models/web_log_detector_comprehensive.pkl',
                 mouse_movement_model_path: str = 'models/mouse_movement_detector_comprehensive.h5'):
        """
        Initialize the simple log tester.
        """
        self.logs_dir = logs_dir
        self.web_log_model_path = web_log_model_path
        self.mouse_movement_model_path = mouse_movement_model_path
        
        # Initialize detection modules
        self.web_log_detector = WebLogDetectionBot()
        self.mouse_movement_detector = MouseMovementDetectionBot()
        
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
    
    def analyze_web_logs(self, filename: str = 'web_logs.json'):
        """Analyze web logs using simple heuristics."""
        print(f"\n{'='*60}")
        print(f"ANALYZING WEB LOGS: {filename}")
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
            
            # Analyze patterns
            methods = [log.get('method', '') for log in session_logs]
            status_codes = [log.get('status', 0) for log in session_logs]
            bytes_sent = [log.get('bytes', 0) for log in session_logs]
            
            # Calculate features
            total_requests = len(session_logs)
            unique_methods = len(set(methods))
            avg_bytes = np.mean(bytes_sent) if bytes_sent else 0
            error_rate = sum(1 for s in status_codes if s >= 400) / total_requests if total_requests > 0 else 0
            
            # Simple bot detection heuristics
            bot_indicators = []
            
            if total_requests > 50:  # High request rate
                bot_indicators.append("High request rate")
            
            if unique_methods < 2:  # Limited HTTP methods
                bot_indicators.append("Limited HTTP methods")
            
            if avg_bytes < 100:  # Small response sizes
                bot_indicators.append("Small response sizes")
            
            if error_rate > 0.1:  # High error rate
                bot_indicators.append("High error rate")
            
            # Check for rapid successive requests (bot-like behavior)
            timestamps = [log.get('timestamp', '') for log in session_logs]
            rapid_requests = 0
            for i in range(1, len(timestamps)):
                try:
                    t1 = pd.to_datetime(timestamps[i-1])
                    t2 = pd.to_datetime(timestamps[i])
                    if (t2 - t1).total_seconds() < 0.1:  # Less than 100ms between requests
                        rapid_requests += 1
                except:
                    pass
            
            if rapid_requests > len(session_logs) * 0.3:  # More than 30% rapid requests
                bot_indicators.append("Rapid successive requests")
            
            # Determine classification
            is_likely_bot = len(bot_indicators) >= 2
            
            results.append({
                'session_id': session_id,
                'total_requests': total_requests,
                'unique_methods': unique_methods,
                'avg_bytes': avg_bytes,
                'error_rate': error_rate,
                'rapid_requests': rapid_requests,
                'bot_indicators': bot_indicators,
                'is_likely_bot': is_likely_bot
            })
            
            print(f"   📈 Total requests: {total_requests}")
            print(f"   🔧 Unique methods: {unique_methods}")
            print(f"   📦 Avg bytes: {avg_bytes:.1f}")
            print(f"   ❌ Error rate: {error_rate:.2%}")
            print(f"   ⚡ Rapid requests: {rapid_requests}")
            print(f"   🚨 Bot indicators: {', '.join(bot_indicators) if bot_indicators else 'None'}")
            print(f"   🤖 Classification: {'LIKELY BOT' if is_likely_bot else 'LIKELY HUMAN'}")
        
        return results
    
    def analyze_mouse_movements(self, filename: str = 'mouse_movements.json'):
        """Analyze mouse movements using simple heuristics."""
        print(f"\n{'='*60}")
        print(f"ANALYZING MOUSE MOVEMENTS: {filename}")
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
                print("   ⚠️  Too few movements for analysis")
                continue
            
            # Extract coordinates
            coordinates = []
            for movement in movements:
                if movement.startswith('m(') and movement.endswith(')'):
                    coords = movement[2:-1].split(',')
                    if len(coords) == 2:
                        try:
                            x, y = int(coords[0]), int(coords[1])
                            coordinates.append([x, y])
                        except ValueError:
                            continue
            
            if len(coordinates) < 5:
                print("   ⚠️  Not enough valid coordinates")
                continue
            
            coords_array = np.array(coordinates)
            
            # Calculate movement patterns
            total_distance = 0
            straight_line_distance = np.sqrt((coords_array[-1][0] - coords_array[0][0])**2 + 
                                           (coords_array[-1][1] - coords_array[0][1])**2)
            
            for i in range(1, len(coords_array)):
                dx = coords_array[i][0] - coords_array[i-1][0]
                dy = coords_array[i][1] - coords_array[i-1][1]
                total_distance += np.sqrt(dx**2 + dy**2)
            
            # Calculate efficiency (straight line distance / total distance)
            efficiency = straight_line_distance / total_distance if total_distance > 0 else 0
            
            # Calculate movement variability
            distances = []
            for i in range(1, len(coords_array)):
                dx = coords_array[i][0] - coords_array[i-1][0]
                dy = coords_array[i][1] - coords_array[i-1][1]
                distances.append(np.sqrt(dx**2 + dy**2))
            
            distance_std = np.std(distances) if distances else 0
            distance_mean = np.mean(distances) if distances else 0
            distance_cv = distance_std / distance_mean if distance_mean > 0 else 0  # Coefficient of variation
            
            # Bot detection heuristics
            bot_indicators = []
            
            if efficiency > 0.9:  # Very straight movement
                bot_indicators.append("Very straight movement")
            
            if distance_cv < 0.1:  # Very consistent movement speed
                bot_indicators.append("Consistent movement speed")
            
            if len(coordinates) > 200:  # Too many movements
                bot_indicators.append("Excessive movements")
            
            if distance_mean < 5:  # Very small movements
                bot_indicators.append("Very small movements")
            
            # Check for grid-like patterns (bot-like)
            x_coords = coords_array[:, 0]
            y_coords = coords_array[:, 1]
            
            x_unique = len(set(x_coords))
            y_unique = len(set(y_coords))
            
            if x_unique < len(coordinates) * 0.3 or y_unique < len(coordinates) * 0.3:
                bot_indicators.append("Grid-like pattern")
            
            # Determine classification
            is_likely_bot = len(bot_indicators) >= 2
            
            results.append({
                'session_id': session_id,
                'total_movements': len(coordinates),
                'total_distance': total_distance,
                'straight_line_distance': straight_line_distance,
                'efficiency': efficiency,
                'distance_cv': distance_cv,
                'bot_indicators': bot_indicators,
                'is_likely_bot': is_likely_bot
            })
            
            print(f"   📏 Total distance: {total_distance:.1f}")
            print(f"   🎯 Straight line distance: {straight_line_distance:.1f}")
            print(f"   ⚡ Efficiency: {efficiency:.3f}")
            print(f"   📊 Distance CV: {distance_cv:.3f}")
            print(f"   🚨 Bot indicators: {', '.join(bot_indicators) if bot_indicators else 'None'}")
            print(f"   🤖 Classification: {'LIKELY BOT' if is_likely_bot else 'LIKELY HUMAN'}")
        
        return results
    
    def analyze_login_attempts(self, filename: str = 'login_attempts.json'):
        """Analyze login attempts."""
        print(f"\n{'='*60}")
        print(f"ANALYZING LOGIN ATTEMPTS: {filename}")
        print(f"{'='*60}")
        
        data = self.load_log_file(filename)
        if not data:
            return
        
        results = []
        
        for i, attempt in enumerate(data):
            print(f"\n📊 Login Attempt {i+1}")
            
            timestamp = attempt.get('timestamp', 'unknown')
            success = attempt.get('details', {}).get('success', False)
            mouse_movements = attempt.get('details', {}).get('mouseMovementData', [])
            
            print(f"   Timestamp: {timestamp}")
            print(f"   Success: {success}")
            print(f"   Mouse movements: {len(mouse_movements)}")
            
            # Analyze timing patterns
            if i > 0:
                try:
                    prev_timestamp = data[i-1].get('timestamp', '')
                    t1 = pd.to_datetime(prev_timestamp)
                    t2 = pd.to_datetime(timestamp)
                    time_diff = (t2 - t1).total_seconds()
                    print(f"   ⏱️  Time since last attempt: {time_diff:.1f}s")
                    
                    if time_diff < 5:  # Very rapid attempts
                        print("   🚨 WARNING: Very rapid login attempts (possible bot)")
                except:
                    pass
            
            # Analyze mouse movement patterns
            if len(mouse_movements) > 10:
                coordinates = []
                for movement in mouse_movements:
                    if movement.startswith('m(') and movement.endswith(')'):
                        coords = movement[2:-1].split(',')
                        if len(coords) == 2:
                            try:
                                x, y = int(coords[0]), int(coords[1])
                                coordinates.append([x, y])
                            except ValueError:
                                continue
                
                if len(coordinates) > 5:
                    coords_array = np.array(coordinates)
                    
                    # Calculate movement characteristics
                    total_distance = 0
                    for j in range(1, len(coords_array)):
                        dx = coords_array[j][0] - coords_array[j-1][0]
                        dy = coords_array[j][1] - coords_array[j-1][1]
                        total_distance += np.sqrt(dx**2 + dy**2)
                    
                    avg_movement = total_distance / (len(coords_array) - 1) if len(coords_array) > 1 else 0
                    print(f"   🖱️  Average movement: {avg_movement:.1f} pixels")
                    
                    if avg_movement < 2:  # Very small movements
                        print("   🚨 WARNING: Very small mouse movements (possible bot)")
            
            results.append({
                'attempt_id': i+1,
                'timestamp': timestamp,
                'success': success,
                'mouse_movements': len(mouse_movements)
            })
        
        return results
    
    def analyze_events(self, filename: str = 'events.json'):
        """Analyze the events file (combined data)."""
        print(f"\n{'='*60}")
        print(f"ANALYZING EVENTS: {filename}")
        print(f"{'='*60}")
        
        data = self.load_log_file(filename)
        if not data:
            return []
        
        results = []
        
        # Check if it's the combined format
        if isinstance(data, dict):
            mouse_movements = data.get('mouseMovements', [])
            web_logs = data.get('webLogs', [])
            
            print(f"   Mouse movements: {len(mouse_movements)}")
            print(f"   Web logs: {len(web_logs)}")
            
            if len(mouse_movements) == 0:
                print("   ⚠️  No mouse movements in events file")
            else:
                print("   📊 Analyzing mouse movements...")
                # Analyze mouse movements
                coordinates = []
                for movement in mouse_movements:
                    if movement.startswith('m(') and movement.endswith(')'):
                        coords = movement[2:-1].split(',')
                        if len(coords) == 2:
                            try:
                                x, y = int(coords[0]), int(coords[1])
                                coordinates.append([x, y])
                            except ValueError:
                                continue
                
                if len(coordinates) > 5:
                    coords_array = np.array(coordinates)
                    
                    # Calculate movement characteristics
                    total_distance = 0
                    for i in range(1, len(coords_array)):
                        dx = coords_array[i][0] - coords_array[i-1][0]
                        dy = coords_array[i][1] - coords_array[i-1][1]
                        total_distance += np.sqrt(dx**2 + dy**2)
                    
                    avg_movement = total_distance / (len(coords_array) - 1) if len(coords_array) > 1 else 0
                    print(f"   🖱️  Average movement: {avg_movement:.1f} pixels")
                    print(f"   📏 Total distance: {total_distance:.1f}")
            
            if len(web_logs) > 0:
                print("   📊 Analyzing web logs...")
                # Analyze web logs
                methods = [log.get('method', '') for log in web_logs]
                status_codes = [log.get('status', 0) for log in web_logs]
                bytes_sent = [log.get('bytes', 0) for log in web_logs]
                
                total_requests = len(web_logs)
                unique_methods = len(set(methods))
                avg_bytes = np.mean(bytes_sent) if bytes_sent else 0
                error_rate = sum(1 for s in status_codes if s >= 400) / total_requests if total_requests > 0 else 0
                
                print(f"   📈 Total requests: {total_requests}")
                print(f"   🔧 Unique methods: {unique_methods}")
                print(f"   📦 Avg bytes: {avg_bytes:.1f}")
                print(f"   ❌ Error rate: {error_rate:.2%}")
            
            results.append({
                'mouse_movements': len(mouse_movements),
                'web_logs': len(web_logs),
                'has_data': len(mouse_movements) > 0 or len(web_logs) > 0
            })
        
        return results
    
    def run_analysis(self):
        """Run analysis on all log files."""
        print("🚀 STARTING LOG FILE ANALYSIS")
        print("="*80)
        
        # Analyze each file
        web_results = self.analyze_web_logs()
        mouse_results = self.analyze_mouse_movements()
        login_results = self.analyze_login_attempts()
        events_results = self.analyze_events()
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        if web_results:
            bot_count = sum(1 for r in web_results if r['is_likely_bot'])
            print(f"Web Logs: {len(web_results)} sessions, {bot_count} likely bots")
        
        if mouse_results:
            bot_count = sum(1 for r in mouse_results if r['is_likely_bot'])
            print(f"Mouse Movements: {len(mouse_results)} sessions, {bot_count} likely bots")
        
        if login_results:
            success_count = sum(1 for r in login_results if r['success'])
            print(f"Login Attempts: {len(login_results)} attempts, {success_count} successful")
        
        if events_results:
            print(f"Events: {len(events_results)} events analyzed")
        
        print(f"\n✅ Analysis completed!")

def main():
    """Main function to run the log file analysis."""
    # Initialize tester
    tester = SimpleLogTester()
    
    # Run analysis
    tester.run_analysis()

if __name__ == "__main__":
    main()
