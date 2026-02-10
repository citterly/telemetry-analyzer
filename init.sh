#!/bin/bash
# Bootstrap script for aim-telemetry development environment
# Run this at the start of each coding session

set -e

echo "=== aim-telemetry Bootstrap ==="

# Confirm directory
echo "Working directory: $(pwd)"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Venv activated: $(which python)"
elif [ -f ~/miniconda3/bin/activate ]; then
    source ~/miniconda3/bin/activate
    echo "Conda activated"
else
    echo "Warning: No venv or conda found, using system Python"
fi

# Install/verify dependencies
echo "Checking dependencies..."
pip install -q pandas pyarrow fastapi uvicorn jinja2 python-multipart numpy scipy matplotlib 2>/dev/null || true

# Create required directories
mkdir -p data/uploads data/exports/processed data/metadata static

# Check for existing server
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "Server already running at http://127.0.0.1:8000"
else
    echo "Starting server..."
    nohup python -m uvicorn src.main.app:app --host 127.0.0.1 --port 8000 > /tmp/telemetry-server.log 2>&1 &
    sleep 2
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "Server started at http://127.0.0.1:8000"
    else
        echo "Warning: Server failed to start. Check /tmp/telemetry-server.log"
    fi
fi

# Run quick smoke test
echo "Running smoke tests..."
python -c "
import sys
sys.path.insert(0, '.')

# Test imports
try:
    from src.config.config import PROJECT_ROOT
    print('  Config: OK')
except Exception as e:
    print(f'  Config: FAIL - {e}')

try:
    from src.config.vehicle_config import TRANSMISSION_SCENARIOS
    print(f'  Vehicle config: OK ({len(TRANSMISSION_SCENARIOS)} scenarios)')
except Exception as e:
    print(f'  Vehicle config: FAIL - {e}')

try:
    from src.analysis.lap_analyzer import LapAnalyzer
    print('  Lap analyzer: OK')
except Exception as e:
    print(f'  Lap analyzer: FAIL - {e}')

try:
    from src.analysis.gear_calculator import GearCalculator
    print('  Gear calculator: OK')
except Exception as e:
    print(f'  Gear calculator: FAIL - {e}')

# Check for parquet files
from pathlib import Path
parquets = list(Path('data/exports').rglob('*.parquet'))
print(f'  Parquet files: {len(parquets)} available')
"

echo ""
echo "=== Bootstrap complete ==="
echo "Run 'cat features.json | python -m json.tool' to see feature status"
echo "Run 'tail -50 claude-progress.txt' to see recent progress"
