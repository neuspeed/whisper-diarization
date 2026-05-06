# Test Setup and Execution Guide

## Quick Start

### 1. Install Test Dependencies

```bash
# Navigate to project root
cd whisper-diarization

# Install the package in development mode with test dependencies
pip install -e ".[test]"
```

If you don't have a setup.cfg or pyproject.toml with test extras defined, install manually:

```bash
pip install pytest pytest-cov pytest-mock
pip install numpy torch soundfile nltk
```

### 2. Run All Tests

```bash
# Basic test run
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/diarizer --cov-report=html --cov-report=term-missing
```

## Installation Details

### Required Test Dependencies

| Package | Purpose | Installation |
|---------|---------|--------------|
| pytest | Test framework | `pip install pytest` |
| pytest-cov | Coverage reporting | `pip install pytest-cov` |
| pytest-mock | Mocking support | `pip install pytest-mock` |
| numpy | Numerical operations | `pip install numpy` |
| soundfile | Audio file I/O | `pip install soundfile` |
| nltk | NLP utilities | `pip install nltk` |

### Optional Dependencies

For running integration tests and testing with actual models:

```bash
pip install faster-whisper
pip install deepmultilingualpunctuation
pip install nemo-toolkit
```

## Test Execution

### Run Specific Test Categories

```bash
# Only unit tests
pytest tests/ -m "not integration and not requires_models"

# Only tests in a specific file
pytest tests/test_helpers.py

# Only tests in a specific class
pytest tests/test_helpers.py::TestGetWordTsAnchor

# Only a specific test
pytest tests/test_helpers.py::TestGetWordTsAnchor::test_anchor_start
```

### Run with Different Output Formats

```bash
# Minimal output
pytest tests/ -q

# Show print statements
pytest tests/ -s

# Show local variables on failure
pytest tests/ -l

# Stop on first failure
pytest tests/ -x

# Stop after N failures
pytest tests/ --maxfail=3
```

### Coverage Analysis

```bash
# Generate HTML coverage report
pytest tests/ --cov=src/diarizer --cov-report=html
# Open htmlcov/index.html in browser

# Show missing lines in terminal
pytest tests/ --cov=src/diarizer --cov-report=term-missing

# Detailed coverage for specific file
pytest tests/ --cov=src/diarizer/helpers --cov-report=term-missing
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Install PyTorch
```bash
pip install torch
# Or for CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "ModuleNotFoundError: No module named 'diarizer'"

**Solution**: Install package in development mode
```bash
pip install -e .
```

### Issue: Tests skip with "soundfile not installed"

**Solution**: Install soundfile
```bash
pip install soundfile
# Or install with system audio support
conda install -c conda-forge soundfile  # if using conda
```

### Issue: "Can't find NLTK data"

**Solution**: Download NLTK data
```python
import nltk
nltk.download('punkt')
```

Or run from Python:
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Advanced Usage

### Running Tests in Parallel

Install pytest-xdist:
```bash
pip install pytest-xdist
```

Run tests in parallel:
```bash
pytest tests/ -n auto  # Uses available CPU cores
pytest tests/ -n 4     # Uses 4 workers
```

### Generate JUnit XML Report

```bash
pytest tests/ --junit-xml=test_results.xml
```

### Generate JSON Report

Install pytest-json-report:
```bash
pip install pytest-json-report
```

Run with JSON output:
```bash
pytest tests/ --json-report --json-report-file=report.json
```

### Profile Tests (Find Slow Tests)

```bash
pytest tests/ --durations=10  # Show 10 slowest tests
pytest tests/ --durations=0   # Show all durations
```

### Watch Mode (Re-run tests on file change)

Install pytest-watch:
```bash
pip install pytest-watch
```

Run in watch mode:
```bash
ptw tests/  # Re-runs tests when files change
```

## CI/CD Integration

### GitHub Actions

Tests are automatically run on push/PR to main/develop branches.
See `.github/workflows/tests.yml` for configuration.

### Local CI-like Testing

Run the same tests as CI locally:
```bash
# Run tests with coverage
pytest tests/ -v --cov=src/diarizer --cov-report=term-missing

# Run linting checks
flake8 src/diarizer tests/
black --check src/diarizer tests/
isort --check-only src/diarizer tests/
```

## Test Development Tips

### Creating New Tests

1. Place test files in `tests/` directory
2. Name files `test_*.py` or `*_test.py`
3. Use clear test names: `test_function_behavior()`
4. Use fixtures from `conftest.py`

Example test:
```python
def test_function_with_valid_input(sample_word_timestamps):
    """Test that function handles valid input correctly."""
    from diarizer.helpers import some_function
    
    result = some_function(sample_word_timestamps)
    
    assert result is not None
    assert isinstance(result, list)
```

### Using Fixtures

```python
def test_with_fixtures(temp_directory, sample_word_timestamps):
    """Test using predefined fixtures."""
    # temp_directory is a temporary directory path
    # sample_word_timestamps is sample timing data
    
    assert os.path.exists(temp_directory)
    assert len(sample_word_timestamps) > 0
```

### Mocking External Calls

```python
from unittest.mock import patch, Mock

@patch('diarizer.helpers.external_function')
def test_with_mock(mock_external):
    """Test with mocked external function."""
    mock_external.return_value = "mocked_result"
    
    result = function_under_test()
    
    assert result == "mocked_result"
    mock_external.assert_called_once()
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    ("en", "english"),
    ("fr", "french"),
    ("de", "german"),
])
def test_language_codes(input, expected):
    """Test multiple input/output pairs."""
    result = get_language_name(input)
    assert result == expected
```

## Performance Tips

1. **Use markers to skip slow tests during development**
   ```bash
   pytest tests/ -m "not slow"
   ```

2. **Run only changed tests**
   ```bash
   pytest tests/ --lf  # Last failed
   pytest tests/ --ff  # Failed first, then others
   ```

3. **Use fixtures with appropriate scope**
   - `function` - default, reset between tests
   - `class` - shared within test class
   - `module` - shared within test module
   - `session` - shared across entire session

## Debugging Tests

### Print Debug Information

```python
def test_with_debugging(sample_data):
    """Debug test output."""
    import pdb
    pdb.set_trace()  # Debugger breakpoint
    
    result = function_under_test(sample_data)
    print(f"Result: {result}")  # Use -s flag to see output
```

### Pytest Debugging

```bash
# Show print statements
pytest tests/ -s

# Drop into debugger on failure
pytest tests/ --pdb

# Drop into debugger on first failure
pytest tests/ --pdb --first
```

## Best Practices

1. ✅ Keep tests fast - use mocks for external dependencies
2. ✅ Use descriptive test names - explain what's being tested
3. ✅ One assertion per test - test one behavior
4. ✅ Use fixtures - reduce duplication
5. ✅ Test edge cases - empty inputs, boundary conditions
6. ✅ Mock external services - don't rely on network/models
7. ✅ Clean up resources - use fixtures with proper cleanup
8. ✅ Keep tests independent - don't depend on test order

## Getting Help

- **Pytest documentation**: https://docs.pytest.org/
- **Python testing**: https://docs.python.org/3/library/unittest.html
- **Test-driven development**: https://en.wikipedia.org/wiki/Test-driven_development
