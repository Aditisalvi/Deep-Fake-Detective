#!/usr/bin/env python3
"""
DeepFake Detective - Launch Script
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'torch', 'torchvision', 'Pillow', 
        'numpy', 'plotly', 'cv2', 'matplotlib', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Packages installed successfully!")

def check_model_file():
    """Check if the model file exists"""
    model_path = Path('deepfake_model.pth')
    if not model_path.exists():
        print("❌ Model file 'deepfake_model.pth' not found!")
        print("Please ensure the model file is in the current directory.")
        return False
    print("✅ Model file found!")
    return True

def main():
    print("🕵️ DeepFake Detective - Starting Application...")
    print("=" * 50)
    
    # Check requirements
    try:
        check_requirements()
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return
    
    # Check model file
    if not check_model_file():
        return
    
    print("🚀 Launching Streamlit application...")
    print("📱 The app will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("=" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.maxUploadSize', '200'  # Allow larger image uploads
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()
