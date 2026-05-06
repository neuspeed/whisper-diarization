"""
Additional tests for edge cases and error handling.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diarizer.helpers import (
    get_word_ts_anchor,
    get_realigned_ws_mapping_with_punctuation,
    find_numeral_symbol_tokens,
)


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_empty_word_speaker_mapping(self):
        """Test realignment with empty mapping."""
        word_speaker_mapping = []
        # Should handle empty list gracefully
        result = get_realigned_ws_mapping_with_punctuation(word_speaker_mapping)
        assert result == []

    def test_single_word_speaker_mapping(self):
        """Test realignment with single word."""
        word_speaker_mapping = [
            {"word": "hello.", "speaker": "Speaker 1"},
        ]
        result = get_realigned_ws_mapping_with_punctuation(word_speaker_mapping)
        assert len(result) == 1
        assert result[0]["speaker"] == "Speaker 1"

    def test_word_ts_anchor_equal_start_end(self):
        """Test anchor when start equals end."""
        result = get_word_ts_anchor(100, 100, option="start")
        assert result == 100
        
        result = get_word_ts_anchor(100, 100, option="mid")
        assert result == 100

    def test_word_ts_anchor_zero_values(self):
        """Test anchor with zero values."""
        result = get_word_ts_anchor(0, 0, option="start")
        assert result == 0
        
        result = get_word_ts_anchor(0, 1000, option="mid")
        assert result == 500

    def test_word_ts_anchor_negative_values(self):
        """Test anchor with negative values (edge case)."""
        result = get_word_ts_anchor(-100, -50, option="start")
        assert result == -100

    def test_word_ts_anchor_very_large_values(self):
        """Test anchor with very large values."""
        large_val = 1e10
        result = get_word_ts_anchor(large_val, large_val * 2, option="mid")
        assert result == large_val * 1.5

    def test_realignment_all_same_speaker(self):
        """Test realignment when all words are same speaker."""
        word_speaker_mapping = [
            {"word": "hello", "speaker": "Speaker 1"},
            {"word": "world", "speaker": "Speaker 1"},
            {"word": "test.", "speaker": "Speaker 1"},
        ]
        
        result = get_realigned_ws_mapping_with_punctuation(word_speaker_mapping)
        
        assert len(result) == 3
        assert all(item["speaker"] == "Speaker 1" for item in result)

    def test_realignment_alternating_speakers(self):
        """Test realignment with alternating speakers."""
        word_speaker_mapping = [
            {"word": "hello", "speaker": "Speaker 1"},
            {"word": "world", "speaker": "Speaker 2"},
            {"word": "test", "speaker": "Speaker 1"},
            {"word": "case.", "speaker": "Speaker 2"},
        ]
        
        result = get_realigned_ws_mapping_with_punctuation(word_speaker_mapping)
        
        assert len(result) == 4
        assert all("speaker" in item for item in result)

    def test_realignment_with_max_words_limit(self):
        """Test realignment with different max_words_in_sentence values."""
        word_speaker_mapping = [
            {"word": word, "speaker": "Speaker 1"}
            for word in ["one", "two", "three", "four", "five.", "six"]
        ]
        
        # Test with different max_words
        for max_words in [2, 5, 10]:
            result = get_realigned_ws_mapping_with_punctuation(
                word_speaker_mapping, max_words_in_sentence=max_words
            )
            assert len(result) == len(word_speaker_mapping)


class TestErrorHandlingAndValidation:
    """Tests for error handling and input validation."""

    def test_process_language_arg_empty_string(self):
        """Test process_language_arg with empty string."""
        from diarizer.helpers import process_language_arg
        
        # Empty string should raise error or be handled
        with pytest.raises((ValueError, KeyError)):
            process_language_arg("", "base")

    def test_find_numeral_symbol_tokens_with_mock_tokenizer(self):
        """Test find_numeral_symbol_tokens function."""
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab.return_value = {
            "hello": 0,
            "1": 1,
            "2": 2,
            "$": 3,
            "%": 4,
            "world": 5,
        }
        
        result = find_numeral_symbol_tokens(mock_tokenizer)
        
        # Should contain -1 (added by default) and tokens with numerals/symbols
        assert -1 in result
        assert 1 in result  # "1"
        assert 2 in result  # "2"
        assert 3 in result  # "$"
        assert 4 in result  # "%"
        assert 0 not in result  # "hello"
        assert 5 not in result  # "world"

    def test_find_numeral_symbol_tokens_empty_vocab(self):
        """Test find_numeral_symbol_tokens with empty vocabulary."""
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab.return_value = {}
        
        result = find_numeral_symbol_tokens(mock_tokenizer)
        
        # Should contain at least -1
        assert -1 in result

    def test_find_numeral_symbol_tokens_all_numerals(self):
        """Test find_numeral_symbol_tokens with all numeral tokens."""
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab.return_value = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
        }
        
        result = find_numeral_symbol_tokens(mock_tokenizer)
        
        # Should contain all tokens plus -1
        assert len(result) == 7  # 6 tokens + -1


class TestBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_format_timestamp_fractional_milliseconds(self):
        """Test format_timestamp with fractional milliseconds."""
        from diarizer.helpers import format_timestamp
        
        # Test with decimal milliseconds
        result = format_timestamp(1234.5)
        assert isinstance(result, str)
        assert ":" in result

    def test_format_timestamp_max_milliseconds(self):
        """Test format_timestamp with maximum millisecond value."""
        from diarizer.helpers import format_timestamp
        
        # Test with 999 milliseconds
        result = format_timestamp(59999)
        assert "59,999" in result or "59.999" in result

    def test_word_speaker_mapping_single_word(self):
        """Test word speaker mapping with single word."""
        from diarizer.helpers import get_words_speaker_mapping
        
        wrd_ts = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        spk_ts = [(0, 1000, "Speaker 1")]
        
        result = get_words_speaker_mapping(wrd_ts, spk_ts)
        
        assert len(result) == 1
        assert result[0]["word"] == "hello"

    def test_word_speaker_mapping_word_at_speaker_boundary(self):
        """Test word timing exactly at speaker turn boundary."""
        from diarizer.helpers import get_words_speaker_mapping
        
        # Word timing exactly at the boundary
        wrd_ts = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        spk_ts = [(0, 1000, "Speaker 1")]
        
        result = get_words_speaker_mapping(wrd_ts, spk_ts)
        
        assert len(result) == 1
        assert result[0]["speaker"] == "Speaker 1"

    def test_get_sentences_with_empty_text(self):
        """Test get_sentences_speaker_mapping with empty text."""
        from diarizer.helpers import get_sentences_speaker_mapping
        
        word_speaker_mapping = [
            {"word": "", "speaker": "Speaker 1", "start_time": 0, "end_time": 1000},
        ]
        spk_ts = [(0, 1000, "Speaker 1")]
        
        result = get_sentences_speaker_mapping(word_speaker_mapping, spk_ts)
        
        assert len(result) > 0

    def test_filter_missing_timestamps_all_missing(self):
        """Test filter_missing_timestamps when all timestamps are missing."""
        from diarizer.helpers import filter_missing_timestamps
        
        word_timestamps = [
            {"word": "hello", "start": None, "end": None},
            {"word": "world", "start": None, "end": None},
        ]
        
        result = filter_missing_timestamps(word_timestamps, initial_timestamp=0, final_timestamp=2.0)
        
        # Should fill in missing timestamps
        assert len(result) >= 1
        assert result[0]["start"] is not None or result[0]["word"] is None

    def test_filter_missing_timestamps_with_none_final(self):
        """Test filter_missing_timestamps with None as final_timestamp."""
        from diarizer.helpers import filter_missing_timestamps
        
        word_timestamps = [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": None, "end": None},
        ]
        
        result = filter_missing_timestamps(word_timestamps, initial_timestamp=0, final_timestamp=None)
        
        # Should handle None final timestamp gracefully
        assert len(result) >= 1


class TestTypeHandling:
    """Tests for type handling and conversions."""

    def test_word_ts_anchor_float_results(self):
        """Test that anchor functions return float results."""
        from diarizer.helpers import get_word_ts_anchor
        
        result = get_word_ts_anchor(1, 2, "mid")
        assert isinstance(result, (int, float))

    def test_word_ts_anchor_with_numpy_values(self):
        """Test anchor with numpy values."""
        result = get_word_ts_anchor(np.float32(100), np.float32(200), "mid")
        # Should work with numpy types
        assert isinstance(result, (int, float, np.floating))

    def test_words_speaker_mapping_output_types(self):
        """Test that output types are correct."""
        from diarizer.helpers import get_words_speaker_mapping
        
        wrd_ts = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        spk_ts = [(0, 1000, "Speaker 1")]
        
        result = get_words_speaker_mapping(wrd_ts, spk_ts)
        
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert isinstance(result[0]["start_time"], (int, float))
        assert isinstance(result[0]["end_time"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
