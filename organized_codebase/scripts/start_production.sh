#!/bin/bash
# TestMaster Dashboard Production Startup Script
# Usage: ./start_production.sh

set -e

echo "🚀 Starting TestMaster Dashboard in Production Mode"
echo "================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing production dependencies..."
pip install -r requirements.txt

# Set production environment variables
export ENVIRONMENT=production
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create log directory if it doesn't exist
mkdir -p /var/log/testmaster 2>/dev/null || mkdir -p logs

# Start Gunicorn with configuration
echo "🌐 Starting Gunicorn server..."
echo "   - Workers: $(python -c 'import multiprocessing; print(multiprocessing.cpu_count() * 2 + 1)')"
echo "   - Binding: 0.0.0.0:5000"
echo "   - Configuration: gunicorn_config.py"
echo ""

gunicorn --config gunicorn_config.py wsgi:application