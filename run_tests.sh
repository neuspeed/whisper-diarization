#!/bin/bash
# Quick test runner script for common operations

echo "Whisper Diarization - Test Runner"
echo "=================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing test dependencies..."
    pip install pytest pytest-cov pytest-mock numpy soundfile nltk
fi

# Parse arguments
if [ $# -eq 0 ]; then
    # No arguments - show menu
    echo -e "${BLUE}Available commands:${NC}"
    echo ""
    echo "  ./run_tests.sh all          - Run all tests"
    echo "  ./run_tests.sh fast         - Run fast tests (skip slow/integration)"
    echo "  ./run_tests.sh coverage     - Run tests with coverage report"
    echo "  ./run_tests.sh helpers      - Run helpers tests only"
    echo "  ./run_tests.sh diarize      - Run diarization tests only"
    echo "  ./run_tests.sh models       - Run model tests only"
    echo "  ./run_tests.sh edge         - Run edge case tests only"
    echo "  ./run_tests.sh verbose      - Run all tests with verbose output"
    echo "  ./run_tests.sh watch        - Watch mode (re-run on file change)"
    echo ""
    exit 0
fi

case "$1" in
    all)
        echo "Running all tests..."
        pytest tests/ -v
        ;;
    fast)
        echo "Running fast tests (excluding slow/integration)..."
        pytest tests/ -v -m "not slow and not integration"
        ;;
    coverage)
        echo "Running tests with coverage report..."
        pytest tests/ --cov=src/diarizer --cov-report=html --cov-report=term-missing
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
        if command -v xdg-open &> /dev/null; then
            xdg-open htmlcov/index.html
        elif command -v open &> /dev/null; then
            open htmlcov/index.html
        fi
        ;;
    helpers)
        echo "Running helpers tests..."
        pytest tests/test_helpers.py -v
        ;;
    diarize)
        echo "Running diarization pipeline tests..."
        pytest tests/test_diarize.py -v
        ;;
    models)
        echo "Running diarization model tests..."
        pytest tests/test_diarization_models.py -v
        ;;
    edge)
        echo "Running edge case tests..."
        pytest tests/test_edge_cases.py -v
        ;;
    verbose)
        echo "Running all tests with verbose output..."
        pytest tests/ -vv --tb=short
        ;;
    watch)
        echo "Starting watch mode (requires pytest-watch)..."
        if ! command -v ptw &> /dev/null; then
            echo "Installing pytest-watch..."
            pip install pytest-watch
        fi
        ptw tests/
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use './run_tests.sh' to see available commands"
        exit 1
        ;;
esac
