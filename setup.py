"""
Setup script for Job Market Analyzer.
Installs dependencies and sets up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'data/raw', 'data/processed', 'reports', 'reports/charts', 'reports/pdfs']
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created successfully")

def setup_environment():
    """Set up environment file."""
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_example.exists() and not env_file.exists():
        print("🔧 Setting up environment file...")
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Environment file created (.env)")
        print("   You can edit .env to customize settings")
    else:
        print("ℹ️ Environment file already exists or .env.example not found")

def test_installation():
    """Test if the installation was successful."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        sys.path.append('src')
        from scrapers.base_scraper import BaseScraper
        from database.db_manager import DatabaseManager
        from analyzers.salary_analyzer import SalaryAnalyzer
        
        print("✅ All modules import successfully")
        
        # Test database creation
        db_manager = DatabaseManager('data/test.db')
        print("✅ Database initialization works")
        
        # Clean up test database
        test_db = Path('data/test.db')
        if test_db.exists():
            test_db.unlink()
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Job Market Analyzer Setup")
    print("=" * 40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Test installation
    if success and test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the demo: python demo.py")
        print("2. Try scraping: python src/main.py scrape --site indeed --query 'python developer' --location 'remote'")
        print("3. Analyze data: python src/main.py analyze --type salary")
    else:
        print("\n⚠️ Setup completed with some issues")
        print("Please check the error messages above")

if __name__ == '__main__':
    main()
