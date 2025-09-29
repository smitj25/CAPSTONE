#!/usr/bin/env python3
"""
Comprehensive Model Analysis and Results Generation

This script provides a complete analysis of all bot detection models including:
- Individual model performance (Mouse Movement, Web Log Detection)
- Fusion algorithm effectiveness
- Training and testing metrics
- Real-world performance analysis
- Honeypot integration effectiveness
- Cross-validation results
- Performance benchmarks
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add the src/core directory to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(project_root, 'src', 'core'))

try:
    from fusion import BotDetectionFusion
    from optimized_bot_detection import OptimizedBotDetector
    from mouse_movements import MouseMovementDetectionBot
    from web_logs import WebLogDetectionBot
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

class ComprehensiveModelAnalyzer:
    """
    Comprehensive analyzer for all bot detection models and fusion algorithms.
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.models_dir = os.path.join(project_root, 'models')
        self.logs_dir = os.path.join(project_root, 'login_page', 'src', 'logs')
        self.results = {
            'analysis_date': datetime.now().isoformat(),
            'project_info': {
                'project_root': project_root,
                'models_directory': self.models_dir,
                'logs_directory': self.logs_dir
            },
            'individual_models': {},
            'fusion_analysis': {},
            'real_world_performance': {},
            'training_metrics': {},
            'testing_metrics': {},
            'honeypot_effectiveness': {},
            'performance_benchmarks': {},
            'recommendations': []
        }
    
    def analyze_model_files(self) -> Dict[str, Any]:
        """Analyze model file characteristics and availability."""
        print("📁 Analyzing Model Files...")
        
        model_info = {}
        model_files = {
            'mouse_movement_model': 'mouse_model.h5',
            'web_log_model': 'logs_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                model_info[model_name] = {
                    'filename': filename,
                    'filepath': filepath,
                    'file_size_bytes': file_size,
                    'file_size_mb': file_size / (1024 * 1024),
                    'exists': True,
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                }
                print(f"  ✅ {model_name}: {filename} ({file_size / (1024*1024):.1f} MB)")
            else:
                model_info[model_name] = {
                    'filename': filename,
                    'filepath': filepath,
                    'exists': False
                }
                print(f"  ❌ {model_name}: {filename} - NOT FOUND")
        
        return model_info
    
    def test_fusion_algorithm(self) -> Dict[str, Any]:
        """Test the fusion algorithm with various scenarios."""
        print("\n🧪 Testing Fusion Algorithm...")
        
        try:
            # Initialize fusion with model paths
            mouse_model_path = os.path.join(self.models_dir, 'mouse_model.h5')
            web_log_model_path = os.path.join(self.models_dir, 'logs_model.pkl')
            
            fusion = BotDetectionFusion(
                web_log_model_path=web_log_model_path,
                mouse_movement_model_path=mouse_model_path
            )
            
            # Define test scenarios
            test_scenarios = [
                {
                    'name': 'Human - Normal Behavior',
                    'mouse_score': 0.15,
                    'web_log_score': 0.2,
                    'honeypot_data': None,
                    'expected_bot': False,
                    'description': 'Normal human behavior with low bot scores'
                },
                {
                    'name': 'Bot - High Mouse Score',
                    'mouse_score': 0.8,
                    'web_log_score': 0.3,
                    'honeypot_data': None,
                    'expected_bot': True,
                    'description': 'Bot with suspicious mouse movements'
                },
                {
                    'name': 'Bot - High Web Log Score',
                    'mouse_score': 0.4,
                    'web_log_score': 0.7,
                    'honeypot_data': None,
                    'expected_bot': True,
                    'description': 'Bot with suspicious web log patterns'
                },
                {
                    'name': 'Bot - Honeypot Triggered (Common Value)',
                    'mouse_score': 0.2,
                    'web_log_score': 0.3,
                    'honeypot_data': {'isTriggered': True, 'honeypotValue': 'bot'},
                    'expected_bot': True,
                    'description': 'Bot caught by honeypot with common bot value'
                },
                {
                    'name': 'Bot - Honeypot Triggered (Any Value)',
                    'mouse_score': 0.3,
                    'web_log_score': 0.4,
                    'honeypot_data': {'isTriggered': True, 'honeypotValue': 'spam'},
                    'expected_bot': True,
                    'description': 'Bot caught by honeypot with any filled value'
                },
                {
                    'name': 'Human - Low Scores, No Honeypot',
                    'mouse_score': 0.25,
                    'web_log_score': 0.35,
                    'honeypot_data': None,
                    'expected_bot': False,
                    'description': 'Human with low but acceptable scores'
                },
                {
                    'name': 'Edge Case - Borderline Mouse Score',
                    'mouse_score': 0.65,
                    'web_log_score': 0.4,
                    'honeypot_data': None,
                    'expected_bot': True,
                    'description': 'Borderline case near high threshold'
                },
                {
                    'name': 'Edge Case - Borderline Low Mouse Score',
                    'mouse_score': 0.35,
                    'web_log_score': 0.4,
                    'honeypot_data': None,
                    'expected_bot': False,
                    'description': 'Borderline case near low threshold'
                }
            ]
            
            fusion_results = {
                'total_tests': len(test_scenarios),
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'accuracy': 0.0,
                'test_details': [],
                'fusion_method_usage': {
                    'honeypot_weighted': 0,
                    'mouse_only': 0,
                    'weighted_average': 0
                },
                'performance_metrics': {
                    'avg_processing_time': 0.0,
                    'total_processing_time': 0.0
                }
            }
            
            total_processing_time = 0
            
            for i, test in enumerate(test_scenarios):
                print(f"  Test {i+1}: {test['name']}")
                start_time = time.time()
                
                try:
                    # Calculate honeypot score first
                    honeypot_score = 0.0
                    if test['honeypot_data']:
                        honeypot_score = fusion.calculate_honeypot_score(test['honeypot_data'])
                    
                    # Use the correct method signature
                    result = fusion.process_session(
                        web_log_features=None,
                        mouse_movement_matrices=None,
                        honeypot_data=test['honeypot_data'],
                        mouse_score=test['mouse_score'],
                        web_log_score=test['web_log_score'],
                        honeypot_score=honeypot_score
                    )
                    
                    processing_time = time.time() - start_time
                    total_processing_time += processing_time
                    
                    is_bot = result['is_bot']
                    final_score = result['final_score']
                    fusion_method = result['fusion_method']
                    
                    # Track fusion method usage
                    fusion_results['fusion_method_usage'][fusion_method] += 1
                    
                    # Check if prediction is correct
                    if is_bot == test['expected_bot']:
                        fusion_results['correct_predictions'] += 1
                        status = "✅ PASSED"
                    else:
                        fusion_results['incorrect_predictions'] += 1
                        status = "❌ FAILED"
                    
                    test_detail = {
                        'test_number': i + 1,
                        'scenario_name': test['name'],
                        'description': test['description'],
                        'input': {
                            'mouse_score': test['mouse_score'],
                            'web_log_score': test['web_log_score'],
                            'honeypot_data': test['honeypot_data']
                        },
                        'output': {
                            'mouse_score': result['mouse_score'],
                            'web_log_score': result['web_log_score'],
                            'honeypot_score': result['honeypot_score'],
                            'final_score': final_score,
                            'is_bot': is_bot,
                            'fusion_method': fusion_method
                        },
                        'expected': {
                            'is_bot': test['expected_bot']
                        },
                        'processing_time': processing_time,
                        'status': status
                    }
                    
                    fusion_results['test_details'].append(test_detail)
                    
                    print(f"    {status} - Score: {final_score:.3f}, Method: {fusion_method}")
                    
                except Exception as e:
                    print(f"    ❌ ERROR: {e}")
                    fusion_results['incorrect_predictions'] += 1
                    fusion_results['test_details'].append({
                        'test_number': i + 1,
                        'scenario_name': test['name'],
                        'error': str(e),
                        'status': "❌ ERROR"
                    })
            
            # Calculate metrics
            fusion_results['accuracy'] = (fusion_results['correct_predictions'] / fusion_results['total_tests']) * 100
            fusion_results['performance_metrics']['total_processing_time'] = total_processing_time
            fusion_results['performance_metrics']['avg_processing_time'] = total_processing_time / fusion_results['total_tests']
            
            # Calculate fusion method percentages
            for method, count in fusion_results['fusion_method_usage'].items():
                fusion_results['fusion_method_usage'][method] = (count / fusion_results['total_tests']) * 100
            
            print(f"\n  📊 Fusion Algorithm Results:")
            print(f"    Accuracy: {fusion_results['accuracy']:.1f}% ({fusion_results['correct_predictions']}/{fusion_results['total_tests']})")
            print(f"    Average Processing Time: {fusion_results['performance_metrics']['avg_processing_time']:.4f}s")
            print(f"    Honeypot Weighted: {fusion_results['fusion_method_usage']['honeypot_weighted']:.1f}%")
            print(f"    Mouse Only: {fusion_results['fusion_method_usage']['mouse_only']:.1f}%")
            print(f"    Weighted Average: {fusion_results['fusion_method_usage']['weighted_average']:.1f}%")
            
            return fusion_results
            
        except Exception as e:
            print(f"  ❌ Fusion algorithm test failed: {e}")
            return {'error': str(e)}
    
    def test_optimized_detector(self) -> Dict[str, Any]:
        """Test the optimized bot detector with real data."""
        print("\n🚀 Testing Optimized Bot Detector...")
        
        try:
            detector = OptimizedBotDetector()
            
            # Test with different session IDs
            test_sessions = ['test_session_1', 'test_session_2', 'test_session_3']
            detector_results = {
                'total_tests': len(test_sessions),
                'successful_tests': 0,
                'failed_tests': 0,
                'test_details': [],
                'performance_metrics': {
                    'avg_processing_time': 0.0,
                    'total_processing_time': 0.0
                }
            }
            
            total_processing_time = 0
            
            for session_id in test_sessions:
                print(f"  Testing session: {session_id}")
                start_time = time.time()
                
                try:
                    # Change to project root for proper path resolution
                    original_cwd = os.getcwd()
                    os.chdir(self.project_root)
                    
                    results = detector.detect_bot_fast(session_id)
                    
                    os.chdir(original_cwd)
                    
                    processing_time = time.time() - start_time
                    total_processing_time += processing_time
                    
                    if 'error' not in results:
                        detector_results['successful_tests'] += 1
                        status = "✅ SUCCESS"
                        
                        test_detail = {
                            'session_id': session_id,
                            'processing_time': processing_time,
                            'results_summary': {
                                'mouse_movements': len(results.get('mouseMovements', [])),
                                'web_logs': 1 if results.get('webLogs') else 0,
                                'login_attempts': len(results.get('loginAttempts', [])),
                                'has_error': 'error' in results
                            },
                            'status': status
                        }
                        
                        print(f"    {status} - {processing_time:.3f}s")
                    else:
                        detector_results['failed_tests'] += 1
                        status = "❌ FAILED"
                        
                        test_detail = {
                            'session_id': session_id,
                            'processing_time': processing_time,
                            'error': results.get('error', 'Unknown error'),
                            'status': status
                        }
                        
                        print(f"    {status} - {results.get('error', 'Unknown error')}")
                    
                    detector_results['test_details'].append(test_detail)
                    
                except Exception as e:
                    detector_results['failed_tests'] += 1
                    detector_results['test_details'].append({
                        'session_id': session_id,
                        'error': str(e),
                        'status': "❌ ERROR"
                    })
                    print(f"    ❌ ERROR: {e}")
            
            # Calculate metrics
            detector_results['performance_metrics']['total_processing_time'] = total_processing_time
            detector_results['performance_metrics']['avg_processing_time'] = total_processing_time / detector_results['total_tests']
            detector_results['success_rate'] = (detector_results['successful_tests'] / detector_results['total_tests']) * 100
            
            print(f"\n  📊 Optimized Detector Results:")
            print(f"    Success Rate: {detector_results['success_rate']:.1f}% ({detector_results['successful_tests']}/{detector_results['total_tests']})")
            print(f"    Average Processing Time: {detector_results['performance_metrics']['avg_processing_time']:.3f}s")
            
            return detector_results
            
        except Exception as e:
            print(f"  ❌ Optimized detector test failed: {e}")
            return {'error': str(e)}
    
    def analyze_real_world_data(self) -> Dict[str, Any]:
        """Analyze real-world data from logs directory."""
        print("\n🌍 Analyzing Real-World Data...")
        
        real_world_analysis = {
            'data_files_analyzed': [],
            'total_sessions': 0,
            'data_quality': {},
            'patterns_detected': {},
            'file_statistics': {}
        }
        
        log_files = ['mouse_movements.json', 'web_logs.json', 'events.json', 'honeypot.json', 'behavior.json']
        
        for log_file in log_files:
            filepath = os.path.join(self.logs_dir, log_file)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    file_size = os.path.getsize(filepath)
                    
                    file_stats = {
                        'filename': log_file,
                        'file_size_bytes': file_size,
                        'file_size_kb': file_size / 1024,
                        'data_structure': type(data).__name__,
                        'entries_count': 0
                    }
                    
                    if isinstance(data, list):
                        file_stats['entries_count'] = len(data)
                    elif isinstance(data, dict):
                        file_stats['entries_count'] = sum(len(v) if isinstance(v, list) else 1 for v in data.values())
                        file_stats['keys'] = list(data.keys())
                    
                    real_world_analysis['data_files_analyzed'].append(file_stats)
                    real_world_analysis['file_statistics'][log_file] = file_stats
                    
                    print(f"  ✅ {log_file}: {file_stats['entries_count']} entries ({file_size / 1024:.1f} KB)")
                    
                    # Extract unique session IDs
                    if log_file == 'events.json' and 'webLogs' in data:
                        sessions = set(entry.get('sessionId') for entry in data['webLogs'] if 'sessionId' in entry)
                        real_world_analysis['total_sessions'] = max(real_world_analysis['total_sessions'], len(sessions))
                    
                except Exception as e:
                    print(f"  ❌ {log_file}: Error reading file - {e}")
            else:
                print(f"  ❌ {log_file}: File not found")
        
        print(f"\n  📊 Real-World Data Summary:")
        print(f"    Files Analyzed: {len(real_world_analysis['data_files_analyzed'])}")
        print(f"    Total Sessions: {real_world_analysis['total_sessions']}")
        
        return real_world_analysis
    
    def generate_performance_benchmarks(self) -> Dict[str, Any]:
        """Generate performance benchmarks for different scenarios."""
        print("\n⚡ Generating Performance Benchmarks...")
        
        benchmarks = {
            'fusion_algorithm_benchmarks': {
                'single_prediction_times': [],
                'batch_prediction_times': [],
                'memory_usage_estimates': {}
            },
            'model_loading_times': {},
            'end_to_end_processing_times': []
        }
        
        # Test fusion algorithm performance
        try:
            mouse_model_path = os.path.join(self.models_dir, 'mouse_model.h5')
            web_log_model_path = os.path.join(self.models_dir, 'logs_model.pkl')
            
            fusion = BotDetectionFusion(
                web_log_model_path=web_log_model_path,
                mouse_movement_model_path=mouse_model_path
            )
            
            # Single prediction benchmark
            test_cases = [
                (0.2, 0.3, None),
                (0.7, 0.4, None),
                (0.3, 0.2, {'isTriggered': True, 'honeypotValue': 'bot'})
            ]
            
            for mouse_score, web_score, honeypot_data in test_cases:
                start_time = time.time()
                
                # Calculate honeypot score
                honeypot_score = 0.0
                if honeypot_data:
                    honeypot_score = fusion.calculate_honeypot_score(honeypot_data)
                
                result = fusion.process_session(
                    web_log_features=None,
                    mouse_movement_matrices=None,
                    honeypot_data=honeypot_data,
                    mouse_score=mouse_score,
                    web_log_score=web_score,
                    honeypot_score=honeypot_score
                )
                processing_time = time.time() - start_time
                
                benchmarks['fusion_algorithm_benchmarks']['single_prediction_times'].append({
                    'mouse_score': mouse_score,
                    'web_score': web_score,
                    'honeypot_triggered': honeypot_data is not None,
                    'processing_time': processing_time,
                    'fusion_method': result['fusion_method']
                })
            
            # Calculate average processing time
            avg_time = np.mean([t['processing_time'] for t in benchmarks['fusion_algorithm_benchmarks']['single_prediction_times']])
            benchmarks['fusion_algorithm_benchmarks']['average_processing_time'] = avg_time
            
            print(f"  ✅ Fusion Algorithm Benchmarks:")
            print(f"    Average Processing Time: {avg_time:.4f}s")
            print(f"    Test Cases: {len(test_cases)}")
            
        except Exception as e:
            print(f"  ❌ Fusion benchmark failed: {e}")
            benchmarks['fusion_algorithm_benchmarks']['error'] = str(e)
        
        return benchmarks
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Check model availability
        model_info = self.results.get('individual_models', {})
        missing_models = [name for name, info in model_info.items() if not info.get('exists', False)]
        
        if missing_models:
            recommendations.append(f"🚨 Critical: Missing model files: {', '.join(missing_models)}")
        
        # Check fusion algorithm performance
        fusion_results = self.results.get('fusion_analysis', {})
        if fusion_results.get('accuracy', 0) < 90:
            recommendations.append("⚠️ Fusion algorithm accuracy below 90% - consider retraining models")
        
        # Check processing performance
        fusion_benchmarks = self.results.get('performance_benchmarks', {}).get('fusion_algorithm_benchmarks', {})
        avg_time = fusion_benchmarks.get('average_processing_time', 0)
        if avg_time > 0.1:
            recommendations.append("⚡ Consider optimizing fusion algorithm - processing time > 100ms")
        
        # Check real-world data
        real_world = self.results.get('real_world_performance', {})
        if real_world.get('total_sessions', 0) == 0:
            recommendations.append("📊 No real-world session data found - consider generating test data")
        
        # Honeypot effectiveness
        fusion_usage = self.results.get('fusion_analysis', {}).get('fusion_method_usage', {})
        honeypot_usage = fusion_usage.get('honeypot_weighted', 0)
        if honeypot_usage > 0:
            recommendations.append(f"🕷️ Honeypot is actively catching bots ({honeypot_usage:.1f}% of cases)")
        
        # General recommendations
        recommendations.extend([
            "📈 Consider implementing model versioning for better tracking",
            "🔄 Set up automated model retraining pipeline",
            "📊 Implement comprehensive logging for production monitoring",
            "🛡️ Add rate limiting and DDoS protection for production use",
            "🧪 Implement A/B testing for fusion algorithm improvements"
        ])
        
        return recommendations
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run the complete comprehensive analysis."""
        print("🔍 Starting Comprehensive Model Analysis...")
        print("=" * 60)
        
        # 1. Analyze model files
        self.results['individual_models'] = self.analyze_model_files()
        
        # 2. Test fusion algorithm
        self.results['fusion_analysis'] = self.test_fusion_algorithm()
        
        # 3. Test optimized detector
        self.results['real_world_performance']['optimized_detector'] = self.test_optimized_detector()
        
        # 4. Analyze real-world data
        self.results['real_world_performance']['data_analysis'] = self.analyze_real_world_data()
        
        # 5. Generate performance benchmarks
        self.results['performance_benchmarks'] = self.generate_performance_benchmarks()
        
        # 6. Generate recommendations
        self.results['recommendations'] = self.generate_recommendations()
        
        # 7. Generate summary statistics
        self.results['summary'] = self.generate_summary()
        
        print("\n" + "=" * 60)
        print("✅ Comprehensive Analysis Complete!")
        print("=" * 60)
        
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all analysis results."""
        summary = {
            'overall_status': 'PASS',
            'critical_issues': 0,
            'warnings': 0,
            'key_metrics': {},
            'system_health': 'HEALTHY'
        }
        
        # Check for critical issues
        model_info = self.results.get('individual_models', {})
        missing_models = [name for name, info in model_info.items() if not info.get('exists', False)]
        if missing_models:
            summary['critical_issues'] += len(missing_models)
            summary['overall_status'] = 'FAIL'
            summary['system_health'] = 'CRITICAL'
        
        # Check fusion algorithm performance
        fusion_results = self.results.get('fusion_analysis', {})
        if fusion_results.get('accuracy', 0) < 80:
            summary['warnings'] += 1
            if summary['system_health'] == 'HEALTHY':
                summary['system_health'] = 'WARNING'
        
        # Key metrics
        summary['key_metrics'] = {
            'fusion_accuracy': fusion_results.get('accuracy', 0),
            'avg_processing_time': fusion_results.get('performance_metrics', {}).get('avg_processing_time', 0),
            'models_available': len([name for name, info in model_info.items() if info.get('exists', False)]),
            'total_recommendations': len(self.results.get('recommendations', []))
        }
        
        return summary
    
    def save_results(self, output_file: str = None) -> str:
        """Save comprehensive results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            output_file = os.path.join(self.project_root, 'Documentation', f'analysis_report_{timestamp}.json')
        
        # Ensure Documentation directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")
        return output_file

def main():
    """Main function to run comprehensive analysis."""
    print("🤖 Comprehensive Bot Detection Model Analysis")
    print("=" * 60)
    
    # Get project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    project_root = os.path.abspath(project_root)
    
    # Initialize analyzer
    analyzer = ComprehensiveModelAnalyzer(project_root)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Save results
    output_file = analyzer.save_results()
    
    # Print final summary
    summary = results.get('summary', {})
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    print(f"   System Health: {summary.get('system_health', 'UNKNOWN')}")
    print(f"   Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"   Warnings: {summary.get('warnings', 0)}")
    print(f"   Fusion Accuracy: {summary.get('key_metrics', {}).get('fusion_accuracy', 0):.1f}%")
    print(f"   Models Available: {summary.get('key_metrics', {}).get('models_available', 0)}/2")
    
    print(f"\n📊 Analysis Complete! Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
