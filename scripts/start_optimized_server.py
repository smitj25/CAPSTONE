#!/usr/bin/env python3
"""
Optimized Server Startup Script

This script preloads ML models and starts the server with optimized settings
for faster bot detection performance.
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path

def preload_models():
    """Preload ML models in a separate process."""
    print("[LOADING] Preloading ML models...")
    
    try:
        # Run optimized script to load models
        result = subprocess.run([
            sys.executable, 'optimized_bot_detection.py', '--fast'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("[SUCCESS] Models preloaded successfully!")
            return True
        else:
            print(f"[WARNING] Model preloading failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Model preloading timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Error preloading models: {e}")
        return False

def start_login_server():
    """Start the login page server."""
    print("[LAUNCH] Starting login page server...")
    
    login_dir = Path("login_page")
    if not login_dir.exists():
        print("[ERROR] Login page directory not found!")
        return None
    
    try:
        # Change to login page directory
        os.chdir(login_dir)
        
        # Start the development server
        server_process = subprocess.Popen([
            'npm', 'run', 'dev'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("[SUCCESS] Login server started!")
        return server_process
        
    except Exception as e:
        print(f"[ERROR] Error starting login server: {e}")
        return None

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n[STOP] Shutting down servers...")
    sys.exit(0)

def main():
    """Main function to start optimized server."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("[LAUNCH] Starting Optimized Bot Detection Server")
    print("=" * 50)
    
    # Preload models
    models_loaded = preload_models()
    
    # Start login server
    login_server_process = start_login_server()
    
    if not login_server_process:
        print("[ERROR] Failed to start login server")
        sys.exit(1)
    
    print("\n[SUCCESS] Server startup complete!")
    print("[STATS] Performance optimizations enabled:")
    print("  • Model caching and preloading")
    print("  • Optimized preprocessing")
    print("  • Result caching (30s TTL)")
    print("  • Rate limiting (3 requests per 5s)")
    print("  • Async processing with timeouts")
    print("  • Request deduplication")
    print("\n[WEB] Login page: http://localhost:5173")
    print("[BOT] Bot detection: Optimized for speed")
    print("\nPress Ctrl+C to stop all servers")
    
    try:
        # Monitor processes
        while True:
            # Check if login server is still running
            if login_server_process.poll() is not None:
                print("[ERROR] Login server stopped unexpectedly")
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")
    
    finally:
        # Clean up processes
        if login_server_process:
            login_server_process.terminate()
            print("[SUCCESS] Login server stopped")

if __name__ == "__main__":
    main()
