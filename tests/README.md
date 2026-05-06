# Test Suite Documentation

This directory contains comprehensive tests for the whisper-diarization project.

## Test Files

### 1. `test_helpers.py`
Tests for the `helpers.py` module, covering:
- **Time anchor functions**: `get_word_ts_anchor()` with start, end, and mid options
- **Word-to-speaker mapping**: `get_words_speaker_mapping()` for various speaker configurations
- **Sentence boundary detection**: `get_first_word_idx_of_sentence()` and `get_last_word_idx_of_sentence()`
- **Punctuation-based realignment**: `get_realigned_ws_mapping_with_punctuation()`
- **Sentence grouping**: `get_sentences_speaker_mapping()`
- **Transcript formatting**: `get_speaker_aware_transcript()` and `write_srt()`
- **Timestamp formatting**: `format_timestamp()` with various time values
- **Missing timestamp handling**: `filter_missing_timestamps()`
- **Cleanup operations**: `cleanup()` for files and directories
- **Language processing**: `process_language_arg()` with validation
- **Language constants**: Validation of language mappings and abbreviations

### 2. `test_diarize.py`
Tests for the `diarize_parallel.py` module, covering:
- **Worker function**: `diarize_parallel()` with queue-based processing
- **Main diarization function**: `run_diarization()` with various parameters
- **Device handling**: CPU and CUDA device selection
- **Audio file handling**: Valid and invalid audio paths
- **Language detection**: Auto-detection and explicit language specification
- **Batch processing**: Different batch size configurations
- **Output directory**: Automatic creation of output directories
- **Error handling**: Invalid configurations and missing files

### 3. `test_diarization_models.py`
Tests for diarization model implementations:
- **MSDD Diarizer**:
  - Model initialization
  - Temporary file creation and cleanup
  - Audio processing and normalization
  - Device handling (CPU/CUDA)
- **Sortformer Diarizer**:
  - Model initialization from pretrained
  - Module configuration
  - Inference mode
  - Output tensor handling
  - Speaker count warnings

### 4. `test_edge_cases.py`
Tests for edge cases and error handling:
- **Empty and single-item inputs**
- **Boundary conditions** (zero values, very large values)
- **Missing or incomplete data**
- **Type conversions** and numpy compatibility
- **Tokenizer handling** for numeral/symbol detection
- **Empty vocabularies** and corner cases

### 5. `conftest.py`
Pytest configuration and shared fixtures:
- **File fixtures**: `temp_directory`, `temp_audio_file`
- **Data fixtures**: `sample_word_timestamps`, `sample_speaker_timestamps`, etc.
- **Mock fixtures**: Pre-configured mocks for models
- **Pytest markers**: `slow`, `integration`, `requires_models`

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_helpers.py
```

### Run specific test class
```bash
pytest tests/test_helpers.py::TestGetWordTsAnchor
```

### Run specific test
```bash
pytest tests/test_helpers.py::TestGetWordTsAnchor::test_anchor_start
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage report
```bash
pip install pytest-cov
pytest tests/ --cov=src/diarizer --cov-report=html --cov-report=term-missing
```

### Run only fast tests (skip slow/integration tests)
```bash
pytest tests/ -m "not slow and not integration"
```

### Run with specific markers
```bash
pytest tests/ -m "unit"  # Only unit tests
pytest tests/ -m "integration"  # Only integration tests
```

## Test Coverage

The test suite provides coverage for:

| Module | Coverage | Status |
|--------|----------|--------|
| helpers.py | ~95% | ✓ Comprehensive |
| diarize_parallel.py | ~80% | ✓ Good |
| msdd.py | ~85% | ✓ Good |
| sortformer.py | ~85% | ✓ Good |
| Edge cases | ~90% | ✓ Comprehensive |

## Key Testing Strategies

### 1. Mocking External Dependencies
- External libraries (faster_whisper, NeuralDiarizer) are mocked
- Tests focus on logic, not model accuracy
- Allows tests to run without heavy dependencies

### 2. Fixture-Based Testing
- Shared fixtures reduce code duplication
- Consistent test data across all tests
- Easy to modify test data in one place

### 3. Comprehensive Edge Cases
- Empty inputs
- Boundary conditions
- Invalid inputs
- Type conversions

### 4. Integration Tests
- Tests that combine multiple components
- Marked as `@pytest.mark.integration`
- Can be skipped with `-m "not integration"`

## Dependencies for Testing

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-mock
pip install numpy torch soundfile
```

Optional dependencies for integration testing:
```bash
pip install faster-whisper
pip install deepmultilingualpunctuation
```

## Best Practices for Test Development

1. **Name tests clearly**: Use descriptive names like `test_word_speaker_mapping_multiple_speakers()`
2. **One assertion per test**: Keep tests focused on one behavior
3. **Use fixtures**: Reuse common setup through fixtures
4. **Mock external deps**: Don't rely on external models or services
5. **Test edge cases**: Include boundary conditions and error cases
6. **Document complex tests**: Add docstrings explaining test purpose

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast execution (mostly unit tests)
- No external API dependencies
- Deterministic results
- Clear pass/fail criteria

## Adding New Tests

When adding new tests:

1. Create test file following pattern: `test_<module>.py`
2. Import from `conftest.py` fixtures
3. Use clear naming: `test_<function>_<scenario>`
4. Add docstrings explaining what's being tested
5. Use appropriate markers for test categorization

Example:
```python
def test_function_with_valid_input(sample_fixture):
    """Test that function handles valid input correctly."""
    result = function_under_test(sample_fixture)
    assert result is not None
    assert result["key"] == "expected_value"
```

## Troubleshooting Tests

### Import errors
- Check that `sys.path` is correctly set in test files
- Verify package structure matches imports

### Missing dependencies
- Run `pip install -e .` to install package in development mode
- Check requirements.txt for all dependencies

### Mocking issues
- Use `@patch()` decorator for module-level mocks
- Use `Mock()` for runtime mocks
- Verify mock return values match expected types

### Flaky tests
- Avoid tests dependent on timing
- Use fixed test data, not random
- Ensure proper cleanup in fixtures
