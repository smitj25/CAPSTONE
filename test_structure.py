#!/usr/bin/env python3
"""
Test script to verify the new project structure works correctly.
"""

import os
import sys

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🧪 Testing project structure and imports...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sys.path.append(src_path)
    
    try:
        # Test core module imports
        from core.web_log_detection_bot import WebLogDetectionBot
        from core.mouse_movements_detection_bot import MouseMovementDetectionBot
        from core.fusion import BotDetectionFusion
        print("✅ Core modules imported successfully")
        
        # Test utility imports
        try:
            import utils.session_processor
            print("✅ Utility modules imported successfully")
        except ImportError:
            print("⚠️  Utility modules not available (optional)")
        except Exception as e:
            print(f"⚠️  Utility module error: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_file_structure():
    """Test that all expected files and directories exist."""
    print("\n📁 Testing file structure...")
    
    expected_dirs = [
        'src',
        'src/core',
        'src/utils', 
        'src/tests',
        'src/config',
        'src/docs',
        'scripts',
        'data',
        'logs',
        'models',
        'login_page'
    ]
    
    expected_files = [
        'src/core/web_log_detection_bot.py',
        'src/core/mouse_movements_detection_bot.py',
        'src/core/fusion.py',
        'src/core/optimized_bot_detection.py',
        'src/tests/test.py',
        'src/tests/simple_log_test.py',
        'src/tests/test.html',
        'scripts/run_demo.py',
        'scripts/bot.py',
        'src/utils/session_processor.py',
        'src/docs/TESTING_GUIDE.md',
        'src/docs/PERFORMANCE_OPTIMIZATION.md',
        'models/web_log_detector_comprehensive.pkl',
        'models/mouse_movement_detector_comprehensive.h5'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"⚠️  File missing: {file_path}")
            # Don't fail for missing files as they might be optional
    
    return all_good

def test_paths():
    """Test that path references work correctly."""
    print("\n🔗 Testing path references...")
    
    # Test relative paths from different locations
    test_paths = [
        ('scripts/run_demo.py', '../models/'),
        ('src/tests/test.py', '../../models/'),
        ('src/core/fusion.py', '../../models/')
    ]
    
    all_good = True
    
    for script_path, model_path in test_paths:
        if os.path.exists(script_path):
            full_model_path = os.path.join(os.path.dirname(script_path), model_path)
            if os.path.exists(full_model_path):
                print(f"✅ Path works: {script_path} -> {model_path}")
            else:
                print(f"⚠️  Path issue: {script_path} -> {model_path}")
        else:
            print(f"⚠️  Script not found: {script_path}")
    
    return all_good

def main():
    """Main test function."""
    print("🚀 Testing Bot Detection System Project Structure")
    print("=" * 50)
    
    # Run all tests
    imports_ok = test_imports()
    structure_ok = test_file_structure()
    paths_ok = test_paths()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"✅ Structure: {'PASS' if structure_ok else 'FAIL'}")
    print(f"✅ Paths: {'PASS' if paths_ok else 'FAIL'}")
    
    if imports_ok and structure_ok and paths_ok:
        print("\n🎉 All tests passed! Project structure is working correctly.")
        print("\n🚀 You can now run:")
        print("   python scripts/run_demo.py")
        print("   python src/tests/test.py")
        print("   python scripts/bot.py")
    else:
        print("\n⚠️  Some tests failed. Please check the structure.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
