#!/usr/bin/env python3
"""
Simple startup script for Telemetry Analyzer
Handles setup and launches the web application
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Ensure we have Python 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def setup_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        result = subprocess.run([sys.executable, "-m", "venv", "venv"])
        if result.returncode != 0:
            print("❌ Failed to create virtual environment")
            sys.exit(1)
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment found")
    
    return venv_path

def get_venv_python(venv_path):
    """Get path to Python executable in virtual environment"""
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "python.exe"
    else:  # Unix-like
        return venv_path / "bin" / "python"

def install_dependencies(venv_python):
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    result = subprocess.run([
        str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Failed to install dependencies")
        print(result.stderr)
        sys.exit(1)
    
    print("✅ Dependencies installed")

def check_analysis_modules():
    """Check if analysis modules are available"""
    analysis_dir = Path("analysis")
    required_files = [
        "data_loader.py",
        "lap_analyzer.py", 
        "config.py",
        "__init__.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (analysis_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing analysis files: {missing_files}")
        print("Please ensure your analysis modules are in the 'analysis/' directory")
        return False
    
    print("✅ Analysis modules found")
    return True

def create_templates_directory():
    """Create templates directory if it doesn't exist"""
    templates_dir = Path("templates")
    if not templates_dir.exists():
        print("📁 Creating templates directory...")
        templates_dir.mkdir()
        
        # Create a basic dashboard template if none exists
        dashboard_template = templates_dir / "dashboard.html"
        if not dashboard_template.exists():
            print("ℹ️  You'll need to create HTML templates in the 'templates/' directory")
    
    return templates_dir.exists()

def start_application(venv_python):
    """Start the FastAPI application"""
    print("🚀 Starting Telemetry Analyzer...")
    print("📍 Open your browser to: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Start the application
        subprocess.run([
            str(venv_python), "app.py"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped")

def main():
    """Main startup sequence"""
    print("🏁 Telemetry Analyzer Startup")
    print("=" * 40)
    
    # Check requirements
    check_python_version()
    
    # Setup virtual environment
    venv_path = setup_virtual_environment()
    venv_python = get_venv_python(venv_path)
    
    # Install dependencies
    install_dependencies(venv_python)
    
    # Check analysis modules
    if not check_analysis_modules():
        sys.exit(1)
    
    # Create templates directory
    create_templates_directory()
    
    # Start application
    start_application(venv_python)

if __name__ == "__main__":
    main()