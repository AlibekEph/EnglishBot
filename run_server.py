#!/usr/bin/env python3
"""
FastAPI-Pydantic Compatibility Wrapper Script
This script ensures compatible versions of FastAPI and Pydantic are installed
before running the local_llm_server.py application.
"""
import os
import sys
import subprocess
import importlib.metadata

def check_version(package, required_version):
    """Check if installed package version matches required version"""
    try:
        installed_version = importlib.metadata.version(package)
        print(f"Found {package} version {installed_version}")
        return installed_version == required_version
    except importlib.metadata.PackageNotFoundError:
        print(f"{package} not found")
        return False

def install_package(package, version):
    """Install a specific version of a package"""
    print(f"Installing {package}=={version}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])

def main():
    """Main function to check and install dependencies, then run the server"""
    # Required versions for compatibility
    required_versions = {
        "fastapi": "0.109.2",
        "pydantic": "1.10.12"
    }
    
    # Check and install required versions
    # for package, version in required_versions.items():
    #     if not check_version(package, version):
    #         install_package(package, version)
    
    # print("Running local_llm_server.py with compatible dependencies...")
    # # Execute the server script
    # server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_llm_server.py")
    # os.execv(sys.executable, [sys.executable, server_script])

if __name__ == "__main__":
    main()

