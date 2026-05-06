"""
Tests for the helpers module.
"""

import os
import tempfile
from io import StringIO
import pytest
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diarizer.helpers import (
    get_word_ts_anchor,
    get_words_speaker_mapping,
    get_first_word_idx_of_sentence,
    get_last_word_idx_of_sentence,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    format_timestamp,
    write_srt,
    find_numeral_symbol_tokens,
    filter_missing_timestamps,
    cleanup,
    process_language_arg,
    LANGUAGES,
    TO_LANGUAGE_CODE,
    punct_model_langs,
    whisper_langs,
    langs_to_iso,
)


class TestGetWordTsAnchor:
    """Tests for get_word_ts_anchor function."""

    def test_anchor_start(self):
        """Test start anchor (default)."""
        result = get_word_ts_anchor(100, 200, option="start")
        assert result == 100

    def test_anchor_end(self):
        """Test end anchor."""
        result = get_word_ts_anchor(100, 200, option="end")
        assert result == 200

    def test_anchor_mid(self):
        """Test mid anchor."""
        result = get_word_ts_anchor(100, 200, option="mid")
        assert result == 150

    def test_anchor_default(self):
        """Test default behavior (start)."""
        result = get_word_ts_anchor(100, 200)
        assert result == 100

    def test_anchor_with_float_values(self):
        """Test with float values."""
        result = get_word_ts_anchor(100.5, 200.5, option="mid")
        assert result == 150.5


class TestGetWordsSpeakerMapping:
    """Tests for get_words_speaker_mapping function."""

    def test_simple_mapping(self):
        """Test basic word to speaker mapping."""
        wrd_ts = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        spk_ts = [(0, 2000, "Speaker 1")]
        
        result = get_words_speaker_mapping(wrd_ts, spk_ts)
        
        assert len(result) == 2
        assert result[0]["word"] == "hello"
        assert result[0]["speaker"] == "Speaker 1"
        assert result[0]["start_time"] == 0
        assert result[0]["end_time"] == 1000

    def test_multiple_speakers(self):
        """Test mapping with multiple speaker turns."""
        wrd_ts = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.5, "end": 2.5, "text": "world"},
        ]
        spk_ts = [(0, 1000, "Speaker 1"), (1000, 3000, "Speaker 2")]
        
        result = get_words_speaker_mapping(wrd_ts, spk_ts)
        
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"

    def test_word_anchor_options(self):
        """Test different word anchor options."""
        wrd_ts = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        spk_ts = [(0, 1500, "Speaker 1"), (1500, 2500, "Speaker 2")]
        
        # With mid anchor, the first word should still be Speaker 1
        result = get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="mid")
        assert len(result) == 2


class TestGetFirstWordIdxOfSentence:
    """Tests for get_first_word_idx_of_sentence function."""

    def test_first_word_of_text(self):
        """Test when word is at the beginning."""
        word_list = ["hello.", "world"]
        speaker_list = ["Speaker 1", "Speaker 1"]
        
        result = get_first_word_idx_of_sentence(1, word_list, speaker_list, max_words=10)
        assert result == 0

    def test_mid_sentence(self):
        """Test when word is in the middle of sentence."""
        word_list = ["hello", "world"]
        speaker_list = ["Speaker 1", "Speaker 1"]
        
        result = get_first_word_idx_of_sentence(1, word_list, speaker_list, max_words=10)
        assert result == 0

    def test_different_speaker(self):
        """Test when previous word is from different speaker."""
        word_list = ["hello.", "world"]
        speaker_list = ["Speaker 1", "Speaker 2"]
        
        result = get_first_word_idx_of_sentence(1, word_list, speaker_list, max_words=10)
        assert result == 1

    def test_sentence_boundary(self):
        """Test at sentence boundary."""
        word_list = ["hello.", "world.", "test"]
        speaker_list = ["Speaker 1", "Speaker 1", "Speaker 1"]
        
        result = get_first_word_idx_of_sentence(2, word_list, speaker_list, max_words=10)
        assert result == 2


class TestGetLastWordIdxOfSentence:
    """Tests for get_last_word_idx_of_sentence function."""

    def test_last_word_of_text(self):
        """Test when word is at the end."""
        word_list = ["hello", "world."]
        
        result = get_last_word_idx_of_sentence(0, word_list, max_words=10)
        assert result == 1

    def test_sentence_ending(self):
        """Test with sentence-ending punctuation."""
        word_list = ["hello.", "world"]
        
        result = get_last_word_idx_of_sentence(0, word_list, max_words=10)
        assert result == 0

    def test_multiple_words_with_limit(self):
        """Test with max_words limit."""
        word_list = ["one", "two", "three", "four", "five."]
        
        result = get_last_word_idx_of_sentence(0, word_list, max_words=2)
        assert result == 1


class TestGetRealignedWsMappingWithPunctuation:
    """Tests for get_realigned_ws_mapping_with_punctuation function."""

    def test_no_realignment_needed(self):
        """Test when no realignment is needed."""
        word_speaker_mapping = [
            {"word": "hello.", "speaker": "Speaker 1"},
            {"word": "world.", "speaker": "Speaker 2"},
        ]
        
        result = get_realigned_ws_mapping_with_punctuation(word_speaker_mapping)
        
        assert len(result) == 2
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"

    def test_realignment_with_speaker_change(self):
        """Test realignment when speakers change mid-sentence."""
        word_speaker_mapping = [
            {"word": "hello", "speaker": "Speaker 1"},
            {"word": "world", "speaker": "Speaker 2"},
            {"word": "test.", "speaker": "Speaker 1"},
        ]
        
        result = get_realigned_ws_mapping_with_punctuation(word_speaker_mapping)
        
        assert len(result) == 3
        # The middle word might be realigned based on majority
        assert all("speaker" in item for item in result)


class TestGetSentencesSpeakerMapping:
    """Tests for get_sentences_speaker_mapping function."""

    def test_single_speaker_simple(self):
        """Test with single speaker and simple mapping."""
        word_speaker_mapping = [
            {"word": "hello", "speaker": "Speaker 1", "start_time": 0, "end_time": 1000},
            {"word": "world.", "speaker": "Speaker 1", "start_time": 1000, "end_time": 2000},
        ]
        spk_ts = [(0, 2000, "Speaker 1")]
        
        result = get_sentences_speaker_mapping(word_speaker_mapping, spk_ts)
        
        assert len(result) > 0
        assert all("speaker" in item and "text" in item for item in result)

    def test_multiple_speakers(self):
        """Test with multiple speakers."""
        word_speaker_mapping = [
            {"word": "hello.", "speaker": "Speaker 1", "start_time": 0, "end_time": 1000},
            {"word": "hi.", "speaker": "Speaker 2", "start_time": 1000, "end_time": 2000},
        ]
        spk_ts = [(0, 1000, "Speaker 1"), (1000, 2000, "Speaker 2")]
        
        result = get_sentences_speaker_mapping(word_speaker_mapping, spk_ts)
        
        assert len(result) >= 1
        assert all("speaker" in item for item in result)


class TestGetSpeakerAwareTranscript:
    """Tests for get_speaker_aware_transcript function."""

    def test_single_speaker_transcript(self):
        """Test transcript with single speaker."""
        sentences = [
            {"speaker": "Speaker 1", "text": "hello world"},
            {"speaker": "Speaker 1", "text": "how are you"},
        ]
        
        output = StringIO()
        get_speaker_aware_transcript(sentences, output)
        result = output.getvalue()
        
        assert "Speaker 1:" in result
        assert "hello world" in result
        assert "how are you" in result

    def test_multiple_speakers_transcript(self):
        """Test transcript with multiple speakers."""
        sentences = [
            {"speaker": "Speaker 1", "text": "hello"},
            {"speaker": "Speaker 2", "text": "hi"},
        ]
        
        output = StringIO()
        get_speaker_aware_transcript(sentences, output)
        result = output.getvalue()
        
        assert "Speaker 1:" in result
        assert "Speaker 2:" in result
        assert "\n\nSpeaker 2:" in result  # Check for paragraph break


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_seconds_only(self):
        """Test formatting seconds only."""
        result = format_timestamp(1500)
        assert "00:01,500" in result

    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        result = format_timestamp(65000)
        assert "01:05,000" in result

    def test_hours_and_minutes(self):
        """Test formatting hours and minutes."""
        result = format_timestamp(3661000)
        assert "01:01:01,000" in result

    def test_zero_timestamp(self):
        """Test zero timestamp."""
        result = format_timestamp(0)
        assert "00:00,000" in result

    def test_large_timestamp(self):
        """Test large timestamp."""
        result = format_timestamp(359999000)
        assert "99:59:59,999" in result

    def test_decimal_marker(self):
        """Test custom decimal marker."""
        result = format_timestamp(1500, decimal_marker=".")
        assert "00:01.500" in result

    def test_negative_timestamp_raises(self):
        """Test that negative timestamp raises error."""
        with pytest.raises(AssertionError):
            format_timestamp(-100)


class TestWriteSrt:
    """Tests for write_srt function."""

    def test_write_srt_single_segment(self):
        """Test writing SRT with single segment."""
        transcript = [
            {
                "speaker": "Speaker 1",
                "text": "Hello world",
                "start_time": 0,
                "end_time": 2000,
            }
        ]
        
        output = StringIO()
        write_srt(transcript, output)
        result = output.getvalue()
        
        assert "1" in result
        assert "00:00,000 --> 00:02,000" in result
        assert "Speaker 1: Hello world" in result

    def test_write_srt_multiple_segments(self):
        """Test writing SRT with multiple segments."""
        transcript = [
            {"speaker": "Speaker 1", "text": "Hello", "start_time": 0, "end_time": 1000},
            {"speaker": "Speaker 2", "text": "Hi", "start_time": 1000, "end_time": 2000},
        ]
        
        output = StringIO()
        write_srt(transcript, output)
        result = output.getvalue()
        
        assert "1" in result
        assert "2" in result
        assert "Speaker 1: Hello" in result
        assert "Speaker 2: Hi" in result

    def test_write_srt_arrow_replacement(self):
        """Test that --> is replaced with -> in text."""
        transcript = [
            {
                "speaker": "Speaker 1",
                "text": "Hello --> world",
                "start_time": 0,
                "end_time": 2000,
            }
        ]
        
        output = StringIO()
        write_srt(transcript, output)
        result = output.getvalue()
        
        # The text should have --> replaced with ->
        assert "Hello -> world" in result


class TestFilterMissingTimestamps:
    """Tests for filter_missing_timestamps function."""

    def test_all_timestamps_present(self):
        """Test when all timestamps are present."""
        word_timestamps = [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": 1.0, "end": 2.0},
        ]
        
        result = filter_missing_timestamps(word_timestamps, initial_timestamp=0, final_timestamp=2.0)
        
        assert len(result) == 2
        assert result[0]["word"] == "hello"
        assert result[1]["word"] == "world"

    def test_missing_first_timestamp(self):
        """Test when first word has no start timestamp."""
        word_timestamps = [
            {"word": "hello", "start": None, "end": None},
            {"word": "world", "start": 1.0, "end": 2.0},
        ]
        
        result = filter_missing_timestamps(word_timestamps, initial_timestamp=0, final_timestamp=2.0)
        
        assert result[0]["start"] == 0

    def test_missing_middle_timestamp(self):
        """Test when middle word has missing timestamps."""
        word_timestamps = [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "dear", "start": None, "end": None},
            {"word": "world", "start": 1.5, "end": 2.5},
        ]
        
        result = filter_missing_timestamps(word_timestamps, initial_timestamp=0, final_timestamp=2.5)
        
        assert len(result) >= 2
        assert result[0]["word"] == "hello"


class TestCleanup:
    """Tests for cleanup function."""

    def test_cleanup_file(self):
        """Test cleanup of a file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b"test content")
        
        assert os.path.exists(temp_file)
        cleanup(temp_file)
        assert not os.path.exists(temp_file)

    def test_cleanup_directory(self):
        """Test cleanup of a directory."""
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        assert os.path.exists(temp_dir)
        cleanup(temp_dir)
        assert not os.path.exists(temp_dir)

    def test_cleanup_nonexistent_raises(self):
        """Test that cleanup raises error for nonexistent path."""
        with pytest.raises(ValueError):
            cleanup("/nonexistent/path/that/does/not/exist")


class TestProcessLanguageArg:
    """Tests for process_language_arg function."""

    def test_valid_language_code(self):
        """Test with valid language code."""
        result = process_language_arg("en", "base")
        assert result == "en"

    def test_valid_language_name(self):
        """Test with valid language name."""
        result = process_language_arg("english", "base")
        assert result == "en"

    def test_language_name_uppercase(self):
        """Test that uppercase language names are handled."""
        result = process_language_arg("ENGLISH", "base")
        assert result == "en"

    def test_language_alias(self):
        """Test language aliases."""
        result = process_language_arg("burmese", "base")
        assert result == "my"

    def test_english_only_model_with_english(self):
        """Test English-only model with English language."""
        result = process_language_arg("en", "base.en")
        assert result == "en"

    def test_english_only_model_with_other_language(self):
        """Test English-only model with non-English language raises error."""
        with pytest.raises(ValueError):
            process_language_arg("fr", "base.en")

    def test_none_language(self):
        """Test None language (auto-detect)."""
        result = process_language_arg(None, "base")
        assert result is None

    def test_invalid_language_raises(self):
        """Test that invalid language raises error."""
        with pytest.raises(ValueError):
            process_language_arg("xyz123invalid", "base")


class TestLanguageConstants:
    """Tests for language-related constants."""

    def test_languages_dict_not_empty(self):
        """Test that LANGUAGES dict is populated."""
        assert len(LANGUAGES) > 0
        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "english"

    def test_to_language_code_contains_base_languages(self):
        """Test that TO_LANGUAGE_CODE contains all base languages."""
        for code, lang_name in LANGUAGES.items():
            assert lang_name in TO_LANGUAGE_CODE
            assert TO_LANGUAGE_CODE[lang_name] == code

    def test_punct_model_langs_valid(self):
        """Test that punct_model_langs contains valid language codes."""
        assert len(punct_model_langs) > 0
        assert "en" in punct_model_langs

    def test_whisper_langs_contains_codes_and_names(self):
        """Test that whisper_langs contains both codes and names."""
        assert len(whisper_langs) > 0
        assert "en" in whisper_langs  # Code
        assert "English" in whisper_langs  # Name

    def test_langs_to_iso_valid_mappings(self):
        """Test that langs_to_iso has valid mappings."""
        assert len(langs_to_iso) > 0
        assert langs_to_iso["en"] == "eng"
        assert langs_to_iso["fr"] == "fre"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
