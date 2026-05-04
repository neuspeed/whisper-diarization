import argparse
import logging
import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Optional, Union

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from diarization import MSDDDiarizer
from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)


def diarize_parallel(audio: torch.Tensor, device: str, queue: mp.Queue) -> None:
    """
    Worker function for parallel diarization using MSDD.
    """
    model = MSDDDiarizer(device=device)
    result = model.diarize(audio)
    queue.put(result)


def run_diarization(
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model_name: str = "medium.en",
    device: Optional[str] = None,
    stemming: bool = True,
    suppress_numerals: bool = False,
    batch_size: int = 4,
    language: Optional[str] = None,
    translate: bool = False,
    device_index: int = 0,
) -> dict:
    """
    Main function to run speaker diarization on an audio file.

    Args:
        audio_path: Path to the input audio file
        output_dir: Directory for temporary and output files (defaults to current dir)
        model_name: Whisper model name
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        stemming: Whether to use source separation for vocals
        suppress_numerals: Whether to transcribe numbers as words
        batch_size: Batch size for inference
        language: Language code or None for auto-detection
        translate: Whether to translate to English
        device_index: CUDA device index

    Returns:
        dict with keys:
            - 'transcript': speaker-aware transcript text
            - 'srt': SRT subtitle content
            - 'txt_path': path to the saved transcript
            - 'srt_path': path to the saved SRT file
    """
    # Ensure spawn method is set before creating any processes
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    mtypes = {"cpu": "int8", "cuda": "float16"}

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Process language argument
    language = process_language_arg(language, model_name)

    # Setup paths
    audio_path = Path(audio_path)
    output_dir = Path(output_dir) if output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    temp_outputs_dir = f"temp_outputs_{pid}"
    temp_path = output_dir / temp_outputs_dir
    temp_path.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem

    # Source separation
    if stemming:
        return_code = os.system(
            f"python -m demucs.separate -n htdemucs --two-stems=vocals "
            f"\"{audio_path}\" -o \"{temp_path}\" --device \"{device}\""
        )
        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use stemming=False to disable source separation."
            )
            vocal_target = str(audio_path)
        else:
            vocal_target = str(
                temp_path / "htdemucs" / base_name / "vocals.wav"
            )
    else:
        vocal_target = str(audio_path)

    # Decode audio
    audio_waveform = faster_whisper.decode_audio(vocal_target)

    # Start Nemo diarization process in parallel
    logging.info(f"Starting Nemo process with vocal_target: {vocal_target}")
    results_queue = mp.Queue()
    nemo_process = mp.Process(
        target=diarize_parallel,
        args=(
            torch.from_numpy(audio_waveform).unsqueeze(0),
            device,
            results_queue,
        ),
    )
    nemo_process.start()

    # Transcribe with Whisper
    whisper_model = faster_whisper.WhisperModel(
        model_name, device=device, compute_type=mtypes[device]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )

    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
            without_timestamps=True,
            task="translate" if translate else "transcribe",
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            without_timestamps=True,
            vad_filter=True,
            task="translate" if translate else "transcribe",
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)

    # Clean up Whisper model from GPU
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=batch_size,
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    # Wait for diarization to finish
    nemo_process.join()
    if results_queue.empty():
        raise RuntimeError("Diarization process did not return any results.")

    speaker_ts = results_queue.get_nowait()

    # Map words to speakers
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    # Punctuation restoration
    if info.language in punct_model_langs:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        words_list = [x["word"] for x in wsm]
        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language. "
            "Using the original punctuation."
        )

    # Realign and generate output
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Save outputs
    txt_path = output_dir / f"{base_name}.txt"
    srt_path = output_dir / f"{base_name}.srt"

    with open(txt_path, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(srt_path, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    # Cleanup temp files
    cleanup(temp_path)

    # Read back the saved content
    transcript_text = txt_path.read_text(encoding="utf-8-sig")
    srt_content = srt_path.read_text(encoding="utf-8-sig")

    return {
        "transcript": transcript_text,
        "srt": srt_content,
        "txt_path": str(txt_path),
        "srt_path": str(srt_path),
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Speaker Diarization using Whisper"
    )
    parser.add_argument(
        "-a", "--audio", required=True, help="Name of the target audio file"
    )
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation",
    )
    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses numerical digits",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        dest="device_index",
        default=0,
        help="Set the index of your CUDA device",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        dest="translate",
        default=False,
        help="Run translate task if needed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=4,
        help="Batch size for batched inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=whisper_langs,
        help="Language spoken in the audio",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--model-name",
        default="medium.en",
        help="Whisper model name",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    result = run_diarization(
        audio_path=args.audio,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        stemming=args.stemming,
        suppress_numerals=args.suppress_numerals,
        batch_size=args.batch_size,
        language=args.language,
        translate=args.translate,
        device_index=args.device_index,
    )

    print(f"Transcript saved to: {result['txt_path']}")
    print(f"SRT saved to: {result['srt_path']}")
    return result


if __name__ == "__main__":
    main()
