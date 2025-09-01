#!/usr/bin/env python3
"""
Bot Detection System Demo

This script demonstrates the bot detection system using pre-trained models
without requiring training data.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List

# Add src directory to path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# Import our detection modules
from core.web_log_detection_bot import WebLogDetectionBot
from core.mouse_movements_detection_bot import MouseMovementDetectionBot
from core.fusion import BotDetectionFusion

def create_sample_web_log_features():
    """Create sample web log features for demonstration."""
    # Create a sample feature vector with 19 features
    features = {
        'total_requests': 25,
        'unique_pages': 8,
        'avg_session_duration': 180.5,
        'requests_per_minute': 8.3,
        'unique_user_agents': 1,
        'unique_ips': 1,
        'status_200_ratio': 0.92,
        'status_404_ratio': 0.04,
        'status_500_ratio': 0.0,
        'get_requests_ratio': 0.88,
        'post_requests_ratio': 0.12,
        'head_requests_ratio': 0.0,
        'image_requests_ratio': 0.32,
        'css_requests_ratio': 0.16,
        'js_requests_ratio': 0.12,
        'html_requests_ratio': 0.40,
        'referrer_present_ratio': 0.76,
        'direct_access_ratio': 0.24,
        'bounce_rate': 0.08
    }
    return pd.DataFrame([features])

def create_sample_mouse_movements():
    """Create sample mouse movement matrices for demonstration."""
    # Create a 50x50 matrix representing mouse movements
    # This simulates mouse movement patterns
    matrix = np.random.rand(50, 50)
    # Normalize to 0-1 range
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    return [matrix]  # Return as list of matrices

def main():
    """Main function to demonstrate the bot detection system."""
    print("=== BOT DETECTION SYSTEM DEMO ===")
    print("Using pre-trained models for demonstration")
    print("="*50)
    
    # Check if models exist
    web_log_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'web_log_detector_comprehensive.pkl')
    mouse_movement_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'mouse_movement_detector_comprehensive.h5')
    
    if not os.path.exists(web_log_model_path):
        print(f"❌ Web log model not found at {web_log_model_path}")
        return
    
    if not os.path.exists(mouse_movement_model_path):
        print(f"❌ Mouse movement model not found at {mouse_movement_model_path}")
        return
    
    print("✅ Pre-trained models found")
    
    # Initialize fusion with existing models
    try:
        fusion = BotDetectionFusion(
            web_log_model_path=web_log_model_path,
            mouse_movement_model_path=mouse_movement_model_path
        )
        print("✅ Fusion module initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing fusion: {e}")
        return
    
    # Demonstrate fusion logic with example scenarios
    print("\n" + "="*50)
    print("FUSION LOGIC DEMONSTRATION")
    print("="*50)
    
    scenarios = [
        {
            'name': 'High Confidence Mouse Movement (Bot)',
            'mouse_score': 0.85,
            'web_log_score': 0.45,
            'expected_fusion': 'mouse_only'
        },
        {
            'name': 'Low Confidence Mouse Movement (Human)',
            'mouse_score': 0.15,
            'web_log_score': 0.65,
            'expected_fusion': 'mouse_only'
        },
        {
            'name': 'Moderate Mouse Movement (Weighted Average)',
            'mouse_score': 0.55,
            'web_log_score': 0.60,
            'expected_fusion': 'weighted_average'
        },
        {
            'name': 'Conflicting Signals (Weighted Average)',
            'mouse_score': 0.40,
            'web_log_score': 0.70,
            'expected_fusion': 'weighted_average'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        result = fusion.process_session(
            mouse_score=scenario['mouse_score'],
            web_log_score=scenario['web_log_score']
        )
        print(f"Mouse Score: {scenario['mouse_score']:.3f}")
        print(f"Web Log Score: {scenario['web_log_score']:.3f}")
        print(f"Fusion Method: {result['fusion_method']}")
        print(f"Final Score: {result['final_score']:.3f}")
        print(f"Classification: {'BOT' if result['is_bot'] else 'HUMAN'}")
        print(f"Expected Fusion: {scenario['expected_fusion']}")
    
    # Test with sample data
    print("\n" + "="*50)
    print("SAMPLE DATA TESTING")
    print("="*50)
    
    try:
        # Create sample data
        sample_web_log = create_sample_web_log_features()
        sample_mouse = create_sample_mouse_movements()
        
        print("✅ Created sample data")
        
        # Test with sample data
        result = fusion.process_session(
            web_log_features=sample_web_log,
            mouse_movement_matrices=sample_mouse
        )
        
        print(f"\nSample Session Result:")
        print(f"Web Log Score: {result['web_log_score']:.3f}")
        print(f"Mouse Score: {result['mouse_score']:.3f}")
        print(f"Fusion Method: {result['fusion_method']}")
        print(f"Final Score: {result['final_score']:.3f}")
        print(f"Classification: {'BOT' if result['is_bot'] else 'HUMAN'}")
        
    except Exception as e:
        print(f"❌ Error testing with sample data: {e}")
    
    print("\n" + "="*50)
    print("SYSTEM READY FOR USE")
    print("="*50)
    print("The bot detection system is now ready to:")
    print("1. Process web server logs for session analysis")
    print("2. Analyze mouse movement patterns")
    print("3. Combine both signals using intelligent fusion")
    print("4. Provide robust bot detection that's harder to evade")
    print("\nTo use with real data:")
    print("- Prepare web log features as pandas DataFrame")
    print("- Prepare mouse movement data as list of numpy arrays")
    print("- Call fusion.process_session() with your data")
    print("="*50)

if __name__ == "__main__":
    main() 