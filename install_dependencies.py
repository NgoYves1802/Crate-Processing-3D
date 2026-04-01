"""
Install all dependencies for Crate Vision project.

This script automatically detects the Python path and installs all required packages.

Usage:
    python install_dependencies.py
"""

import sys
import subprocess
import os

import PyQt5


def get_python_path():
    """Get the Python interpreter path."""
    return sys.executable


def install_package(python_path, package, verbose=True):
    """Install a single package using pip."""
    try:
        if verbose:
            print(f"Installing {package}...")
        subprocess.check_call(
            [python_path, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.STDOUT
        )
        if verbose:
            print(f"  ✓ {package} installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to install {package}")
        return False


def main():
    """Main installation function."""
    python_path = get_python_path()
    
    print("=" * 50)
    print("Crate Vision - Dependency Installer")
    print("=" * 50)
    print(f"Python path: {python_path}")
    print()
    
    # List of all required packages
    packages = [
        # Core dependencies
        "numpy",
        "scipy",
        "Pillow",
        
        # Image processing & visualization
        "opencv-python",
        "matplotlib",
        
        # Machine learning (PyTorch)
        "torch",
        "torchvision",
        
        # ONNX Runtime for AI inference
        "onnxruntime",
        
        # scikit-learn
        "scikit-learn",
        
        # Progress bar
        "tqdm",

        #GUi
        "PyQt5",
    ]
    
    # Optional packages
    optional_packages = [
        "onnx",  # For ONNX model export/verification
    ]
    
    success_count = 0
    fail_count = 0
    
    print("Installing core dependencies...")
    print("-" * 30)
    
    for pkg in packages:
        if install_package(python_path, pkg):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("Installing optional dependencies...")
    print("-" * 30)
    
    for pkg in optional_packages:
        if install_package(python_path, pkg):
            success_count += 1
        else:
            # Optional packages failing is not critical
            print(f"  ! {pkg} optional - skipping")
    
    print()
    print("=" * 50)
    print(f"Installation complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print("=" * 50)
    
    if fail_count > 0:
        print("\nNote: Some packages may have failed due to:")
        print("  - Network issues")
        print("  - Already installed")
        print("  - Platform compatibility")
        
    print("\nYou can now run the Crate Vision pipeline!")
    print("Example: python main.py")


if __name__ == "__main__":
    main()