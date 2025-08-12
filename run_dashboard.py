#!/usr/bin/env python3
"""
Launch script for Job Market Analyzer Pro
Starts the unified dashboard with a single command
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the unified dashboard"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    main_script = project_root / "src" / "main.py"
    
    if not main_script.exists():
        print("Error: main.py not found in src directory")
        sys.exit(1)
    
    try:
        # Launch the dashboard
        print("🚀 Starting Job Market Analyzer Pro...")
        print("🌐 Launching unified dashboard...")
        
        # Run the main script with dashboard command
        subprocess.run([
            sys.executable, 
            str(main_script), 
            "dashboard"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
