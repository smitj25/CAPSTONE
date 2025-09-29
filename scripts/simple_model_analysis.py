#!/usr/bin/env python3
"""
Simple Model Analysis and Results Generation

This script provides a focused analysis of the bot detection system using the optimized detector.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any

# Add the src/core directory to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(project_root, 'src', 'core'))

try:
    from core.optimized_bot_detection import OptimizedBotDetector
    print("✅ Successfully imported OptimizedBotDetector")
except ImportError as e:
    print(f"❌ Failed to import OptimizedBotDetector: {e}")
    sys.exit(1)

class SimpleModelAnalyzer:
    """Simple analyzer for bot detection system."""
    
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
            'model_availability': {},
            'detector_performance': {},
            'real_world_data': {},
            'performance_benchmarks': {},
            'summary': {}
        }
    
    def check_model_availability(self) -> Dict[str, Any]:
        """Check if model files are available."""
        print("📁 Checking Model Availability...")
        
        model_files = {
            'mouse_movement_model': 'mouse_model.h5',
            'web_log_model': 'logs_model.pkl'
        }
        
        model_info = {}
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                model_info[model_name] = {
                    'filename': filename,
                    'filepath': filepath,
                    'file_size_mb': file_size / (1024 * 1024),
                    'exists': True,
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                }
                print(f"  ✅ {model_name}: {filename} ({file_size / (1024*1024):.1f} MB)")
            else:
                model_info[model_name] = {
                    'filename': filename,
                    'exists': False
                }
                print(f"  ❌ {model_name}: {filename} - NOT FOUND")
        
        return model_info
    
    def test_detector_performance(self) -> Dict[str, Any]:
        """Test the optimized detector performance."""
        print("\n🚀 Testing Optimized Detector Performance...")
        
        detector_results = {
            'test_sessions': [],
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'avg_processing_time': 0.0,
            'performance_summary': {}
        }
        
        # Test with different session IDs
        test_sessions = [
            'test_human_behavior',
            'test_bot_behavior', 
            'test_edge_case',
            'test_honeypot_triggered',
            'test_normal_user'
        ]
        
        total_processing_time = 0
        
        for session_id in test_sessions:
            print(f"  Testing session: {session_id}")
            start_time = time.time()
            
            try:
                # Initialize detector
                detector = OptimizedBotDetector()
                
                # Change to project root for proper path resolution
                original_cwd = os.getcwd()
                os.chdir(self.project_root)
                
                # Run detection
                results = detector.detect_bot_fast(session_id)
                
                # Restore original directory
                os.chdir(original_cwd)
                
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                detector_results['total_tests'] += 1
                
                if 'error' not in results:
                    detector_results['successful_tests'] += 1
                    status = "✅ SUCCESS"
                    
                    # Extract key metrics
                    login_attempts = results.get('loginAttempts', [])
                    mouse_movements = results.get('mouseMovements', [])
                    web_logs = results.get('webLogs', {})
                    
                    test_detail = {
                        'session_id': session_id,
                        'processing_time': processing_time,
                        'status': status,
                        'results_summary': {
                            'mouse_movements_count': len(mouse_movements),
                            'web_logs_available': web_logs is not None,
                            'login_attempts_count': len(login_attempts),
                            'has_error': 'error' in results
                        },
                        'bot_detection_results': []
                    }
                    
                    # Extract bot detection results from login attempts
                    for attempt in login_attempts:
                        bot_result = {
                            'attempt_number': attempt.get('attemptNumber', 0),
                            'mouse_score': attempt.get('mouseScore', 0.0),
                            'web_score': attempt.get('webScore', 0.0),
                            'honeypot_score': attempt.get('honeypotScore', 0.0),
                            'fusion_score': attempt.get('fusionScore', 0.0),
                            'classification': attempt.get('classification', 'UNKNOWN')
                        }
                        test_detail['bot_detection_results'].append(bot_result)
                    
                    print(f"    {status} - {processing_time:.3f}s - {len(login_attempts)} login attempts")
                    
                else:
                    detector_results['failed_tests'] += 1
                    status = "❌ FAILED"
                    
                    test_detail = {
                        'session_id': session_id,
                        'processing_time': processing_time,
                        'status': status,
                        'error': results.get('error', 'Unknown error')
                    }
                    
                    print(f"    {status} - {results.get('error', 'Unknown error')}")
                
                detector_results['test_sessions'].append(test_detail)
                
            except Exception as e:
                detector_results['failed_tests'] += 1
                detector_results['total_tests'] += 1
                
                test_detail = {
                    'session_id': session_id,
                    'status': "❌ ERROR",
                    'error': str(e)
                }
                
                detector_results['test_sessions'].append(test_detail)
                print(f"    ❌ ERROR: {e}")
        
        # Calculate performance metrics
        if detector_results['total_tests'] > 0:
            detector_results['avg_processing_time'] = total_processing_time / detector_results['total_tests']
            detector_results['success_rate'] = (detector_results['successful_tests'] / detector_results['total_tests']) * 100
            
            detector_results['performance_summary'] = {
                'total_tests': detector_results['total_tests'],
                'successful_tests': detector_results['successful_tests'],
                'failed_tests': detector_results['failed_tests'],
                'success_rate_percent': detector_results['success_rate'],
                'avg_processing_time_seconds': detector_results['avg_processing_time'],
                'total_processing_time_seconds': total_processing_time
            }
        
        print(f"\n  📊 Detector Performance Summary:")
        print(f"    Success Rate: {detector_results.get('success_rate', 0):.1f}% ({detector_results['successful_tests']}/{detector_results['total_tests']})")
        print(f"    Average Processing Time: {detector_results['avg_processing_time']:.3f}s")
        
        return detector_results
    
    def analyze_real_world_data(self) -> Dict[str, Any]:
        """Analyze real-world data from logs directory."""
        print("\n🌍 Analyzing Real-World Data...")
        
        data_analysis = {
            'files_analyzed': [],
            'total_files': 0,
            'total_entries': 0,
            'data_quality': 'GOOD',
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
                    entries_count = 0
                    
                    if isinstance(data, list):
                        entries_count = len(data)
                    elif isinstance(data, dict):
                        entries_count = sum(len(v) if isinstance(v, list) else 1 for v in data.values())
                    
                    file_stats = {
                        'filename': log_file,
                        'file_size_kb': file_size / 1024,
                        'entries_count': entries_count,
                        'exists': True
                    }
                    
                    data_analysis['files_analyzed'].append(file_stats)
                    data_analysis['file_statistics'][log_file] = file_stats
                    data_analysis['total_entries'] += entries_count
                    
                    print(f"  ✅ {log_file}: {entries_count} entries ({file_size / 1024:.1f} KB)")
                    
                except Exception as e:
                    print(f"  ❌ {log_file}: Error reading file - {e}")
                    data_analysis['data_quality'] = 'POOR'
            else:
                print(f"  ❌ {log_file}: File not found")
                data_analysis['data_quality'] = 'POOR'
        
        data_analysis['total_files'] = len(data_analysis['files_analyzed'])
        
        print(f"\n  📊 Real-World Data Summary:")
        print(f"    Files Analyzed: {data_analysis['total_files']}")
        print(f"    Total Entries: {data_analysis['total_entries']}")
        print(f"    Data Quality: {data_analysis['data_quality']}")
        
        return data_analysis
    
    def generate_performance_benchmarks(self) -> Dict[str, Any]:
        """Generate performance benchmarks."""
        print("\n⚡ Generating Performance Benchmarks...")
        
        benchmarks = {
            'detector_loading_time': 0.0,
            'single_detection_time': 0.0,
            'batch_processing_capability': 'UNKNOWN',
            'memory_usage_estimate': 'LOW',
            'scalability_rating': 'GOOD'
        }
        
        try:
            # Test detector loading time
            print("  Testing detector loading time...")
            start_time = time.time()
            
            detector = OptimizedBotDetector()
            detector.load_models_once()
            
            loading_time = time.time() - start_time
            benchmarks['detector_loading_time'] = loading_time
            
            print(f"    Detector loading time: {loading_time:.3f}s")
            
            # Test single detection time
            print("  Testing single detection time...")
            start_time = time.time()
            
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            results = detector.detect_bot_fast('benchmark_test')
            
            os.chdir(original_cwd)
            
            detection_time = time.time() - start_time
            benchmarks['single_detection_time'] = detection_time
            
            print(f"    Single detection time: {detection_time:.3f}s")
            
            # Determine scalability rating based on performance
            if detection_time < 0.1:
                benchmarks['scalability_rating'] = 'EXCELLENT'
            elif detection_time < 0.5:
                benchmarks['scalability_rating'] = 'GOOD'
            elif detection_time < 1.0:
                benchmarks['scalability_rating'] = 'FAIR'
            else:
                benchmarks['scalability_rating'] = 'POOR'
            
            # Estimate batch processing capability
            if detection_time < 0.05:
                benchmarks['batch_processing_capability'] = 'HIGH (20+ requests/sec)'
            elif detection_time < 0.1:
                benchmarks['batch_processing_capability'] = 'MEDIUM (10+ requests/sec)'
            else:
                benchmarks['batch_processing_capability'] = 'LOW (<10 requests/sec)'
            
        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")
            benchmarks['error'] = str(e)
        
        return benchmarks
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all results."""
        summary = {
            'overall_status': 'PASS',
            'system_health': 'HEALTHY',
            'critical_issues': 0,
            'warnings': 0,
            'key_metrics': {},
            'recommendations': []
        }
        
        # Check model availability
        model_info = self.results.get('model_availability', {})
        missing_models = [name for name, info in model_info.items() if not info.get('exists', False)]
        
        if missing_models:
            summary['critical_issues'] += len(missing_models)
            summary['overall_status'] = 'FAIL'
            summary['system_health'] = 'CRITICAL'
            summary['recommendations'].append(f"🚨 Critical: Missing model files: {', '.join(missing_models)}")
        
        # Check detector performance
        detector_perf = self.results.get('detector_performance', {})
        success_rate = detector_perf.get('success_rate', 0)
        
        if success_rate < 80:
            summary['warnings'] += 1
            if summary['system_health'] == 'HEALTHY':
                summary['system_health'] = 'WARNING'
            summary['recommendations'].append(f"⚠️ Detector success rate below 80%: {success_rate:.1f}%")
        
        # Check processing time
        avg_time = detector_perf.get('avg_processing_time', 0)
        if avg_time > 1.0:
            summary['warnings'] += 1
            if summary['system_health'] == 'HEALTHY':
                summary['system_health'] = 'WARNING'
            summary['recommendations'].append(f"⚡ Processing time is high: {avg_time:.3f}s")
        
        # Key metrics
        summary['key_metrics'] = {
            'models_available': len([name for name, info in model_info.items() if info.get('exists', False)]),
            'total_models': len(model_info),
            'detector_success_rate': success_rate,
            'avg_processing_time': avg_time,
            'real_world_entries': self.results.get('real_world_data', {}).get('total_entries', 0)
        }
        
        # Add general recommendations
        summary['recommendations'].extend([
            "📊 Consider implementing comprehensive logging for production monitoring",
            "🔄 Set up automated model retraining pipeline",
            "🛡️ Add rate limiting and DDoS protection for production use",
            "📈 Implement model versioning for better tracking"
        ])
        
        return summary
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis."""
        print("🔍 Starting Simple Model Analysis...")
        print("=" * 50)
        
        # 1. Check model availability
        self.results['model_availability'] = self.check_model_availability()
        
        # 2. Test detector performance
        self.results['detector_performance'] = self.test_detector_performance()
        
        # 3. Analyze real-world data
        self.results['real_world_data'] = self.analyze_real_world_data()
        
        # 4. Generate performance benchmarks
        self.results['performance_benchmarks'] = self.generate_performance_benchmarks()
        
        # 5. Generate summary
        self.results['summary'] = self.generate_summary()
        
        print("\n" + "=" * 50)
        print("✅ Simple Model Analysis Complete!")
        print("=" * 50)
        
        return self.results
    
    def save_results(self, output_file: str = None) -> str:
        """Save results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.project_root, 'Documentation', f'simple_model_analysis_{timestamp}.json')
        
        # Ensure Documentation directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")
        return output_file

def main():
    """Main function."""
    print("🤖 Simple Bot Detection Model Analysis")
    print("=" * 50)
    
    # Get project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    project_root = os.path.abspath(project_root)
    
    # Initialize analyzer
    analyzer = SimpleModelAnalyzer(project_root)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Save results
    output_file = analyzer.save_results()
    
    # Print final summary
    summary = results.get('summary', {})
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    print(f"   System Health: {summary.get('system_health', 'UNKNOWN')}")
    print(f"   Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"   Warnings: {summary.get('warnings', 0)}")
    
    key_metrics = summary.get('key_metrics', {})
    print(f"   Models Available: {key_metrics.get('models_available', 0)}/{key_metrics.get('total_models', 0)}")
    print(f"   Success Rate: {key_metrics.get('detector_success_rate', 0):.1f}%")
    print(f"   Avg Processing Time: {key_metrics.get('avg_processing_time', 0):.3f}s")
    
    print(f"\n📊 Analysis Complete! Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
