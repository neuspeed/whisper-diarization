"""
Pytest configuration and shared fixtures for all tests.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import optional dependencies
try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None


@pytest.fixture
def temp_directory():
    """Create a temporary directory that's cleaned up after the test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_audio_file(temp_directory):
    """Create a temporary audio file for testing."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")
    
    if np is None:
        pytest.skip("numpy not installed")
    
    audio_path = os.path.join(temp_directory, "test_audio.wav")
    # Create a simple audio file (1 second, 16kHz, mono)
    audio_data = np.random.randn(16000).astype(np.float32)
    sf.write(audio_path, audio_data, 16000)
    return audio_path


@pytest.fixture
def sample_audio_tensor():
    """Create a sample audio tensor for testing."""
    if torch is None:
        pytest.skip("torch not installed")
    # 16kHz, 1 second of audio
    return torch.randn(16000)


@pytest.fixture
def sample_word_timestamps():
    """Create sample word timestamp data."""
    return [
        {"start": 0.0, "end": 0.5, "text": "hello"},
        {"start": 0.5, "end": 1.0, "text": "world"},
        {"start": 1.0, "end": 1.5, "text": "test"},
    ]


@pytest.fixture
def sample_speaker_timestamps():
    """Create sample speaker timestamp data."""
    return [
        (0, 1500, "Speaker 1"),
        (1500, 2500, "Speaker 2"),
    ]


@pytest.fixture
def sample_word_speaker_mapping():
    """Create sample word-to-speaker mapping."""
    return [
        {"word": "hello", "speaker": "Speaker 1", "start_time": 0, "end_time": 500},
        {"word": "world", "speaker": "Speaker 1", "start_time": 500, "end_time": 1000},
        {"word": "test", "speaker": "Speaker 2", "start_time": 1000, "end_time": 1500},
    ]


@pytest.fixture
def sample_sentences():
    """Create sample sentences for transcript testing."""
    return [
        {
            "speaker": "Speaker 1",
            "text": "Hello world ",
            "start_time": 0,
            "end_time": 1000,
        },
        {
            "speaker": "Speaker 2",
            "text": "How are you ",
            "start_time": 1000,
            "end_time": 2000,
        },
    ]


@pytest.fixture
def sample_transcript():
    """Create sample transcript for SRT writing."""
    return [
        {
            "speaker": "Speaker 1",
            "text": "Hello world",
            "start_time": 0,
            "end_time": 1500,
        },
        {
            "speaker": "Speaker 2",
            "text": "Hi there",
            "start_time": 1500,
            "end_time": 3000,
        },
    ]


@pytest.fixture
def mock_whisper_model():
    """Create a mock Whisper model."""
    from unittest.mock import Mock
    
    mock = Mock()
    mock.transcribe.return_value = [
        Mock(start=0, end=1.0, text="hello world"),
    ]
    return mock


@pytest.fixture
def mock_msdd_diarizer():
    """Create a mock MSDD diarizer."""
    from unittest.mock import Mock
    
    mock = Mock()
    mock.diarize.return_value = [
        (0, 1.0, "Speaker 1"),
    ]
    return mock


@pytest.fixture
def mock_sortformer_diarizer():
    """Create a mock Sortformer diarizer."""
    from unittest.mock import Mock
    
    if torch is None:
        pytest.skip("torch not installed")
    
    mock = Mock()
    mock.diarize.return_value = torch.randn(1, 16000, 4)  # Mock speaker predictions
    return mock


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_models: marks tests that require actual models"
    )


@pytest.fixture(scope="session")
def resource_warning_filters():
    """Filter resource warnings during tests."""
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
