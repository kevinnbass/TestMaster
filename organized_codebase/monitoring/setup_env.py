#!/usr/bin/env python3
"""
Setup Environment Variables Script
=================================

This script helps you set up the environment variables for the TestMaster Dashboard.
It creates a .env file with your API keys and other configuration.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with user input for API keys"""

    print("🚀 TestMaster Dashboard Environment Setup")
    print("=" * 50)

    env_content = """# TestMaster Dashboard Environment Variables

# API Keys for LLM Services (Required)
GEMINI_API_KEY=
GOOGLE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=change_this_to_a_random_secret_key

# Database Configuration
DATABASE_URL=sqlite:///dashboard.db
REDIS_URL=redis://localhost:6379/0

# Server Configuration
HOST=0.0.0.0
PORT=5000
WORKERS=4

# Security Settings
ENABLE_SECURITY=false
CORS_ORIGINS=*

# Monitoring & Analytics
ENABLE_ANALYTICS=true
ANALYTICS_DATABASE=./data/analytics.db

# Cache Settings
CACHE_TYPE=filesystem
CACHE_DEFAULT_TIMEOUT=300

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/dashboard.log

# Feature Flags
ENABLE_LLM_FEATURES=true
ENABLE_REAL_TIME_MONITORING=true
ENABLE_ADVANCED_ANALYTICS=true
"""

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    env_file_path = script_dir / '.env'

    # Check if .env already exists
    if env_file_path.exists():
        overwrite = input("⚠️  .env file already exists. Overwrite? (y/N): ")
        if overwrite.lower() != 'y':
            print("ℹ️  Setup cancelled. Existing .env file preserved.")
            return

    # Write the .env file
    with open(env_file_path, 'w') as f:
        f.write(env_content)

    print("✅ .env file created successfully!")
    print(f"📁 Location: {env_file_path}")
    print()
    print("🔧 Next Steps:")
    print("1. Edit the .env file and add your API keys")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the server: python server.py")
    print()
    print("📚 API Keys Required:")
    print("   - GEMINI_API_KEY: For Google's Gemini AI")
    print("   - GOOGLE_API_KEY: For Google services")
    print("   - OPENAI_API_KEY: For OpenAI GPT models")
    print("   - ANTHROPIC_API_KEY: For Claude AI")
    print()
    print("💡 Tip: You can get API keys from:")
    print("   - Google AI Studio: https://makersuite.google.com/app/apikey")
    print("   - OpenAI: https://platform.openai.com/api-keys")
    print("   - Anthropic: https://console.anthropic.com/")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking Dependencies...")

    required_packages = [
        'flask',
        'python-dotenv',
        'requests'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print("
⚠️  Missing packages. Install with:"        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies are available!")
    return True

def main():
    """Main setup function"""
    try:
        create_env_file()
        check_dependencies()

        print("
🎉 Setup Complete!"        print("📝 Remember to add your API keys to the .env file")
        print("🚀 You can now run: python server.py")

    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")

if __name__ == '__main__':
    main()

