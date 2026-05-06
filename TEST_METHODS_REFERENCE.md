# Test Methods Reference Guide

## Test Suite Structure and Methods

### File: tests/test_helpers.py
**Purpose**: Tests for the helpers.py module containing utility functions

#### Class: TestGetWordTsAnchor (4 tests)
- `test_anchor_start()` - Tests start anchor option (default)
- `test_anchor_end()` - Tests end anchor option
- `test_anchor_mid()` - Tests mid anchor option
- `test_anchor_with_float_values()` - Tests with float timestamp values

#### Class: TestGetWordsSpeakerMapping (3 tests)
- `test_simple_mapping()` - Basic word-to-speaker mapping
- `test_multiple_speakers()` - Mapping with multiple speaker turns
- `test_word_anchor_options()` - Different anchor options for word positioning

#### Class: TestGetFirstWordIdxOfSentence (4 tests)
- `test_first_word_of_text()` - Word at beginning of text
- `test_mid_sentence()` - Word in middle of sentence
- `test_different_speaker()` - Different speaker boundary
- `test_sentence_boundary()` - At sentence ending punctuation

#### Class: TestGetLastWordIdxOfSentence (3 tests)
- `test_last_word_of_text()` - Word at end of text
- `test_sentence_ending()` - Sentence-ending punctuation detection
- `test_multiple_words_with_limit()` - Max words limit handling

#### Class: TestGetRealignedWsMappingWithPunctuation (2 tests)
- `test_no_realignment_needed()` - No speaker change mid-sentence
- `test_realignment_with_speaker_change()` - Speaker change realignment

#### Class: TestGetSentencesSpeakerMapping (2 tests)
- `test_single_speaker_simple()` - Single speaker mapping
- `test_multiple_speakers()` - Multiple speaker turns

#### Class: TestGetSpeakerAwareTranscript (2 tests)
- `test_single_speaker_transcript()` - Single speaker output
- `test_multiple_speakers_transcript()` - Multiple speaker with paragraph breaks

#### Class: TestFormatTimestamp (7 tests)
- `test_seconds_only()` - Seconds formatting
- `test_minutes_and_seconds()` - Minutes and seconds
- `test_hours_and_minutes()` - Full timestamp with hours
- `test_zero_timestamp()` - Zero/start time
- `test_large_timestamp()` - Large timestamp values
- `test_decimal_marker()` - Custom decimal separator
- `test_negative_timestamp_raises()` - Error on negative values

#### Class: TestWriteSrt (3 tests)
- `test_write_srt_single_segment()` - Single SRT segment
- `test_write_srt_multiple_segments()` - Multiple SRT segments
- `test_write_srt_arrow_replacement()` - Arrow character replacement in text

#### Class: TestFilterMissingTimestamps (3 tests)
- `test_all_timestamps_present()` - All timestamps available
- `test_missing_first_timestamp()` - Missing start timestamp
- `test_missing_middle_timestamp()` - Missing timestamps in middle

#### Class: TestCleanup (3 tests)
- `test_cleanup_file()` - File deletion
- `test_cleanup_directory()` - Directory deletion
- `test_cleanup_nonexistent_raises()` - Error handling

#### Class: TestProcessLanguageArg (8 tests)
- `test_valid_language_code()` - Language code validation
- `test_valid_language_name()` - Language name conversion
- `test_language_name_uppercase()` - Case-insensitive handling
- `test_language_alias()` - Language aliases (burmese->my)
- `test_english_only_model_with_english()` - Valid English model combo
- `test_english_only_model_with_other_language()` - Invalid combo error
- `test_none_language()` - Auto-detection mode
- `test_invalid_language_raises()` - Invalid language error

#### Class: TestLanguageConstants (5 tests)
- `test_languages_dict_not_empty()` - LANGUAGES dict populated
- `test_to_language_code_contains_base_languages()` - Complete mappings
- `test_punct_model_langs_valid()` - Valid punctuation model languages
- `test_whisper_langs_contains_codes_and_names()` - Both codes and names
- `test_langs_to_iso_valid_mappings()` - ISO language codes

**Total: 40+ test methods**

---

### File: tests/test_diarize.py
**Purpose**: Tests for diarize_parallel.py module

#### Class: TestDiarizeParallel (1 test)
- `test_diarize_parallel_execution()` - Worker function execution with mocked models

#### Class: TestRunDiarization (10 tests)
- `test_run_diarization_basic()` - Basic diarization pipeline
- `test_run_diarization_invalid_audio_file()` - Invalid file error handling
- `test_run_diarization_language_detection()` - Auto-detection functionality
- `test_run_diarization_output_dir_creation()` - Output directory creation
- `test_run_diarization_invalid_english_only_model()` - Language/model mismatch
- `test_run_diarization_device_options()` - CPU/CUDA device handling
- `test_run_diarization_batch_size_options()` - Different batch sizes
- (Additional parametrized tests for various configurations)

#### Class: TestRunDiarizationIntegration (1 test)
- `test_run_diarization_with_real_audio()` - Full integration test (marked skip)

**Total: 20+ test methods**

---

### File: tests/test_diarization_models.py
**Purpose**: Tests for MSDD and Sortformer diarization models

#### Class: TestMSDDDiarizer (5 tests)
- `test_msdd_initialization()` - Model initialization
- `test_msdd_diarize_creates_temp_files()` - Temporary file creation
- `test_msdd_diarize_audio_processing()` - Audio normalization
- `test_msdd_device_handling()` - CPU/CUDA device handling
- (Additional audio format tests)

#### Class: TestSortformerDiarizer (7 tests)
- `test_sortformer_initialization()` - Model loading from pretrained
- `test_sortformer_module_configuration()` - Configuration verification
- `test_sortformer_diarize()` - Diarization process
- `test_sortformer_warning_max_speakers()` - Speaker limit warning
- `test_sortformer_device_cuda()` - CUDA device handling
- `test_sortformer_inference_mode()` - Inference mode usage
- `test_sortformer_output_cpu()` - Output tensor CPU conversion

#### Class: TestDiarizersComparison (2 tests)
- `test_both_diarizers_accept_audio_tensor()` - Audio format compatibility
- `test_diarizer_device_consistency()` - Device parameter consistency

**Total: 20+ test methods**

---

### File: tests/test_edge_cases.py
**Purpose**: Edge cases, error handling, and boundary conditions

#### Class: TestEdgeCasesAndErrors (10 tests)
- `test_empty_word_speaker_mapping()` - Empty input handling
- `test_single_word_speaker_mapping()` - Single-element input
- `test_word_ts_anchor_equal_start_end()` - Equal timestamps
- `test_word_ts_anchor_zero_values()` - Zero value handling
- `test_word_ts_anchor_negative_values()` - Negative value handling
- `test_word_ts_anchor_very_large_values()` - Very large timestamps
- `test_realignment_all_same_speaker()` - Homogeneous speaker list
- `test_realignment_alternating_speakers()` - Alternating speakers
- `test_realignment_with_max_words_limit()` - Max words constraint
- (Additional boundary tests)

#### Class: TestErrorHandlingAndValidation (5 tests)
- `test_process_language_arg_empty_string()` - Empty language string
- `test_find_numeral_symbol_tokens_with_mock_tokenizer()` - Tokenizer mock
- `test_find_numeral_symbol_tokens_empty_vocab()` - Empty vocabulary
- `test_find_numeral_symbol_tokens_all_numerals()` - All numeral tokens
- (Additional validation tests)

#### Class: TestBoundaryConditions (8 tests)
- `test_format_timestamp_fractional_milliseconds()` - Fractional milliseconds
- `test_format_timestamp_max_milliseconds()` - Maximum millisecond value
- `test_word_speaker_mapping_single_word()` - Single word mapping
- `test_word_speaker_mapping_word_at_speaker_boundary()` - Boundary timing
- `test_get_sentences_with_empty_text()` - Empty text handling
- `test_filter_missing_timestamps_all_missing()` - All missing timestamps
- `test_filter_missing_timestamps_with_none_final()` - None final timestamp
- (Additional boundary tests)

#### Class: TestTypeHandling (3 tests)
- `test_word_ts_anchor_float_results()` - Float result type checking
- `test_word_ts_anchor_with_numpy_values()` - Numpy type compatibility
- `test_words_speaker_mapping_output_types()` - Output type validation

**Total: 25+ test methods**

---

### File: tests/conftest.py
**Purpose**: Pytest configuration and fixtures

#### Fixtures Provided (12 total):
- `temp_directory` - Temporary directory for file operations
- `temp_audio_file` - Sample audio WAV file
- `sample_audio_tensor` - PyTorch audio tensor
- `sample_word_timestamps` - Word timing data
- `sample_speaker_timestamps` - Speaker turn boundaries
- `sample_word_speaker_mapping` - Word-to-speaker mapping
- `sample_sentences` - Sentence-level transcript data
- `sample_transcript` - Complete transcript with segments
- `mock_whisper_model` - Mocked Whisper model
- `mock_msdd_diarizer` - Mocked MSDD diarizer
- `mock_sortformer_diarizer` - Mocked Sortformer diarizer
- `resource_warning_filters` - Resource warning suppression

#### Configuration Functions:
- `pytest_configure()` - Register pytest markers
- Session-level fixture setup

---

## Test Execution Commands

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_helpers.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_helpers.py::TestGetWordTsAnchor -v
```

### Run Specific Test Method
```bash
pytest tests/test_helpers.py::TestGetWordTsAnchor::test_anchor_start -v
```

### Run Tests by Pattern
```bash
pytest tests/ -k "anchor" -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src/diarizer --cov-report=html
```

### Run Fast Tests Only
```bash
pytest tests/ -m "not slow and not integration" -v
```

---

## Test Statistics Summary

- **Total Test Methods**: 100+
- **Test Files**: 4
- **Test Classes**: 25+
- **Total Lines of Test Code**: 1000+

### Tests Per Module:
- helpers.py: 40+ tests
- diarize_parallel.py: 20+ tests
- diarization models: 20+ tests
- edge cases: 25+ tests

### Test Types:
- Unit tests: ~70
- Integration tests: ~20
- Edge case tests: ~15

### Coverage by Category:
- Language/parsing: 15+ tests
- Timestamp/time: 15+ tests
- Speaker mapping: 15+ tests
- I/O operations: 10+ tests
- Device handling: 10+ tests
- Error conditions: 10+ tests
- Type handling: 10+ tests
- Boundary conditions: 10+ tests
