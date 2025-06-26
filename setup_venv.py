#!/usr/bin/env python3
"""
Virtual Environment Setup Script for Leave Policy QA API
This script automates the setup of a virtual environment and project dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_venv(venv_name="venv"):
    """Create virtual environment"""
    if os.path.exists(venv_name):
        print(f"⚠️  Virtual environment '{venv_name}' already exists")
        return True
    
    return run_command(f"python -m venv {venv_name}", f"Creating virtual environment '{venv_name}'")

def get_activate_command(venv_name="venv"):
    """Get the appropriate activation command based on OS"""
    system = platform.system().lower()
    
    if system == "windows":
        return f"{venv_name}\\Scripts\\activate"
    else:  # Linux, macOS
        return f"source {venv_name}/bin/activate"

def install_dependencies(venv_name="venv"):
    """Install project dependencies"""
    system = platform.system().lower()
    
    if system == "windows":
        pip_path = f"{venv_name}\\Scripts\\pip"
    else:
        pip_path = f"{venv_name}/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_path} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    return run_command(f"{pip_path} install -r requirements.txt", "Installing project dependencies")

def create_env_file():
    """Create .env file from example if it doesn't exist"""
    if os.path.exists(".env"):
        print("⚠️  .env file already exists")
        return True
    
    if os.path.exists("env_example.txt"):
        return run_command("copy env_example.txt .env" if platform.system().lower() == "windows" else "cp env_example.txt .env", "Creating .env file from example")
    else:
        print("❌ env_example.txt not found")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Leave Policy QA API Virtual Environment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    venv_name = "venv"
    if not create_venv(venv_name):
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(venv_name):
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print(f"1. Activate the virtual environment:")
    print(f"   {get_activate_command(venv_name)}")
    print("\n2. Edit the .env file with your OpenAI API key:")
    print("   OPENAI_API_KEY=your_actual_openai_api_key_here")
    print("\n3. Ingest your PDF:")
    print("   python ingest_pdf.py path/to/your/leave_policy.pdf")
    print("\n4. Start the API:")
    print("   python app.py")
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 