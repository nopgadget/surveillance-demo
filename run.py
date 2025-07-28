#!/usr/bin/env python3
"""
Surveillance Demo Launcher
A simple launcher for the modular surveillance demo application.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for the surveillance demo."""
    try:
        # Import and run the modular application
        from src.main import main as run_app
        run_app()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 