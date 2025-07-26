#!/usr/bin/env python3
"""
Test script for the Web Bot Detection system.
This script demonstrates how to use the three main components:
1. Web Log Detection Bot
2. Mouse Movement Detection Bot  
3. Fusion Module

Make sure you have the required dependencies installed:
pip install pandas numpy scikit-learn tensorflow keras
"""

import os
import sys
from web_log_detection_bot import WebLogDetectionBot
from mouse_movements_detection_bot import MouseMovementDetectionBot
from fusion import BotDetectionFusion

def test_individual_components():
    """Test individual components with sample data"""
    print("=== Testing Individual Components ===\n")
    
    # Dataset paths
    dataset_base = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase1'
    
    # Check if dataset exists
    if not os.path.exists(dataset_base):
        print(f"Dataset not found at: {dataset_base}")
        print("Please ensure the dataset is available at the specified path.")
        return False
    
    # Test Web Log Detection Bot
    print("1. Testing Web Log Detection Bot...")
    try:
        web_detector_d1 = WebLogDetectionBot(dataset_type='D1')
        web_detector_d2 = WebLogDetectionBot(dataset_type='D2')
        print("   ✓ Web log detectors initialized successfully")
    except Exception as e:
        print(f"   ✗ Error initializing web log detector: {e}")
        return False
    
    # Test Mouse Movement Detection Bot
    print("2. Testing Mouse Movement Detection Bot...")
    try:
        mouse_detector = MouseMovementDetectionBot()
        print("   ✓ Mouse movement detector initialized successfully")
    except Exception as e:
        print(f"   ✗ Error initializing mouse movement detector: {e}")
        return False
    
    # Test Fusion Module
    print("3. Testing Fusion Module...")
    try:
        fusion_d1 = BotDetectionFusion(dataset_type='D1')
        fusion_d2 = BotDetectionFusion(dataset_type='D2')
        print("   ✓ Fusion modules initialized successfully")
    except Exception as e:
        print(f"   ✗ Error initializing fusion module: {e}")
        return False
    
    return True

def test_with_sample_session():
    """Test with a sample session if available"""
    print("\n=== Testing with Sample Session ===\n")
    
    dataset_base = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase1'
    
    # Check for available sessions in D1
    d1_mouse_dir = os.path.join(dataset_base, 'data/mouse_movements/humans_and_moderate_bots')
    if os.path.exists(d1_mouse_dir):
        sessions = [d for d in os.listdir(d1_mouse_dir) if os.path.isdir(os.path.join(d1_mouse_dir, d))]
        if sessions:
            sample_session = sessions[0]
            print(f"Found sample session: {sample_session}")
            
            # Test mouse movement detection
            try:
                mouse_detector = MouseMovementDetectionBot()
                mouse_data_dir = os.path.join(dataset_base, 'data/mouse_movements')
                score = mouse_detector.process_session_data(sample_session, mouse_data_dir, 'phase1', 'D1')
                print(f"Mouse movement score for {sample_session}: {score:.3f}")
            except Exception as e:
                print(f"Error processing mouse movement data: {e}")
            
            # Test web log detection (if web logs are available)
            web_logs_dir = os.path.join(dataset_base, 'data/web_logs')
            if os.path.exists(web_logs_dir):
                web_log_files = [f for f in os.listdir(web_logs_dir) if f.endswith('.txt')]
                if web_log_files:
                    try:
                        web_detector = WebLogDetectionBot(dataset_type='D1')
                        web_log_path = os.path.join(web_logs_dir, web_log_files[0])
                        web_scores = web_detector.get_web_log_score(web_log_path)
                        if sample_session in web_scores:
                            print(f"Web log score for {sample_session}: {web_scores[sample_session]:.3f}")
                        else:
                            print(f"Session {sample_session} not found in web logs")
                    except Exception as e:
                        print(f"Error processing web log data: {e}")
            
            # Test fusion
            try:
                fusion = BotDetectionFusion(dataset_type='D1')
                if web_log_files:
                    web_log_path = os.path.join(web_logs_dir, web_log_files[0])
                    result = fusion.process_session(sample_session, web_log_path, mouse_data_dir, 'phase1')
                    print(f"Fusion result for {sample_session}:")
                    print(f"  - Mouse score: {result['score_mv']:.3f}")
                    print(f"  - Web log score: {result['score_wl']:.3f}")
                    print(f"  - Fused score: {result['score_tot']:.3f}")
                    print(f"  - Classification: {result['classification']}")
            except Exception as e:
                print(f"Error in fusion processing: {e}")
        else:
            print("No sessions found in D1 mouse movement data")
    else:
        print("D1 mouse movement directory not found")

def show_dataset_info():
    """Show information about the dataset structure"""
    print("\n=== Dataset Information ===\n")
    
    dataset_base = '/Users/khatuaryan/Desktop/Aryan/Studies/Projects/CAPSTONE/dataset/phase1'
    
    if not os.path.exists(dataset_base):
        print(f"Dataset not found at: {dataset_base}")
        return
    
    # Check annotations
    annotations_dir = os.path.join(dataset_base, 'annotations')
    if os.path.exists(annotations_dir):
        print("Available annotation sets:")
        for subdir in os.listdir(annotations_dir):
            subdir_path = os.path.join(annotations_dir, subdir)
            if os.path.isdir(subdir_path):
                print(f"  - {subdir}")
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            print(f"    - {file}: {len(lines)} sessions")
    
    # Check mouse movement data
    mouse_dir = os.path.join(dataset_base, 'data/mouse_movements')
    if os.path.exists(mouse_dir):
        print("\nMouse movement data:")
        for subdir in os.listdir(mouse_dir):
            subdir_path = os.path.join(mouse_dir, subdir)
            if os.path.isdir(subdir_path):
                sessions = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
                print(f"  - {subdir}: {len(sessions)} sessions")
    
    # Check web logs
    web_logs_dir = os.path.join(dataset_base, 'data/web_logs')
    if os.path.exists(web_logs_dir):
        log_files = [f for f in os.listdir(web_logs_dir) if f.endswith('.txt')]
        print(f"\nWeb log files: {len(log_files)} files")
        for log_file in log_files[:5]:  # Show first 5
            print(f"  - {log_file}")
        if len(log_files) > 5:
            print(f"  ... and {len(log_files) - 5} more")

def main():
    """Main test function"""
    print("Web Bot Detection System Test")
    print("=" * 40)
    
    # Show dataset information
    show_dataset_info()
    
    # Test individual components
    if test_individual_components():
        print("\n✓ All components initialized successfully!")
        
        # Test with sample data
        test_with_sample_session()
        
        print("\n=== Test Complete ===")
        print("\nTo use the system:")
        print("1. Train the models with your labeled data")
        print("2. Use the fusion module to process sessions")
        print("3. Evaluate performance using the built-in metrics")
        
    else:
        print("\n✗ Component initialization failed!")
        print("Please check the error messages above and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()
