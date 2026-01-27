"""
InspectAI - Quick Start Script
Starts the backend and opens the frontend in a browser
"""

import subprocess
import time
import webbrowser
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import torch
        print("✓ Dependencies installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False

def check_model():
    """Check if at least one model is trained"""
    models_dir = "models/patchcore"
    if not os.path.exists(models_dir):
        print("✗ No models found")
        print("\nPlease train at least one model:")
        print("  python train.py --category bottle")
        return False
    
    categories = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not categories:
        print("✗ No trained models found")
        print("\nPlease train at least one model:")
        print("  python train.py --category bottle")
        return False
    
    print(f"✓ Found {len(categories)} trained model(s): {', '.join(categories)}")
    return True

def start_backend():
    """Start the FastAPI backend"""
    print("\n🚀 Starting backend server...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return backend_process

def start_frontend():
    """Start a simple HTTP server for the frontend"""
    print("🌐 Starting frontend server...")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "3000", "--directory", "frontend"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return frontend_process

def main():
    print("="*60)
    print("  InspectAI - Industrial Visual Inspection System")
    print("="*60)
    print()
    
    # Check dependencies
    print("Checking system...")
    if not check_dependencies():
        return
    
    # Check models
    if not check_model():
        return
    
    print("\n" + "="*60)
    
    # Start backend
    backend = start_backend()
    time.sleep(3)  # Wait for backend to start
    
    # Start frontend
    frontend = start_frontend()
    time.sleep(2)  # Wait for frontend to start
    
    print("\n✅ InspectAI is running!")
    print("\n📍 Access points:")
    print("   Frontend: http://localhost:3000")
    print("   Backend API: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    print("\n⏹️  Press Ctrl+C to stop both servers")
    print("="*60)
    
    # Open browser
    time.sleep(1)
    webbrowser.open("http://localhost:3000")
    
    # Keep running
    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        backend.terminate()
        frontend.terminate()
        print("✅ Servers stopped")

if __name__ == "__main__":
    main()
