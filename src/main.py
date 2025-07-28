import os
from pathlib import Path
from . import AppConfig, SurveillanceDemo

# Set environment variable for OpenCV
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    """Main entry point for the surveillance demo."""
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("img").mkdir(exist_ok=True)
    
    try:
        # Load configuration
        config = AppConfig.from_toml()
        
        # Create and run the surveillance demo
        app = SurveillanceDemo(config)
        app.run()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 