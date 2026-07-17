"""
Local diarization tuning.

Takes a prepped WAV and a cached word-list checkpoint from a previous run.
Re-runs pyannote diarization with different settings and re-renders the
transcript/SRT/JSON/coverage outputs. Originals are never overwritten —
new files are suffixed with a configurable tag.

**CRITICAL** (fixed 2026-07-10 after the Elastrin regression):

  This script MUST call `merge_overlapping_words()` on the loaded
  checkpoint. The checkpoint is the *pre-merge* word list from the
  transcription batch loop and carries duplicates from the sliding-window
  overlap zones. Skipping merge (previous behavior) produced tuned
  transcripts with 56% more text than the untuned ones, most of it
  duplicate sentences.

  Also loads audio duration and runs the same tail-gap / coverage guards
  the main pipeline runs.

Usage:
    python tune_diarization.py <prepped.wav> <words_checkpoint.json> \\
        --diarization-model pyannote/speaker-diarization-3.1 \\
        --num-speakers 3 \\
        --tag tuned_3spk \\
        [--smoothing-sec 2.0]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf

# Reuse the renderers, assignment, audit, segment-building functions.
sys.path.insert(0, str(Path(__file__).parent))
import canary_transcribe as ct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to prepped 16k mono WAV")
    parser.add_argument("words_checkpoint", help="Path to *.checkpoint.json")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Force this many speakers; omit for auto-detect")
    parser.add_argument("--smoothing-sec", type=float, default=2.0,
                        help="Speaker-assignment rolling window")
    parser.add_argument("--tag", required=True,
                        help="Suffix tag for output files (e.g. tuned_3spk)")
    parser.add_argument("--model-name", default=None,
                        help="canary|parakeet — auto-inferred from checkpoint name")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    words_path = Path(args.words_checkpoint).resolve()
    if not audio_path.is_file():
        print(f"ERROR: audio not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    if not words_path.is_file():
        print(f"ERROR: words checkpoint not found: {words_path}", file=sys.stderr)
        sys.exit(1)

    # Infer model name from checkpoint filename if not given.
    if not args.model_name:
        cn = words_path.stem.lower()
        if "canary" in cn:
            args.model_name = "canary"
        elif "parakeet" in cn:
            args.model_name = "parakeet"
        else:
            args.model_name = "model"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    # base = audio path with .wav stripped (so e.g. "PeteDepo.prepped")
    base = str(audio_path.with_suffix(""))
    out_base = f"{base}.{args.model_name}.{args.tag}"

    # Refuse to overwrite an existing tagged output. (User said don't overwrite
    # for now; we GC later.)
    transcript_path = Path(out_base + ".transcript.txt")
    if transcript_path.exists():
        print(f"ERROR: {transcript_path} already exists. Use a different --tag.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] Loading words: {words_path.name}")
    with open(words_path) as f:
        words = json.load(f)
    print(f"  {len(words)} raw words loaded from checkpoint")

    # CRITICAL FIX (2026-07-10): the checkpoint is pre-merge. Skipping the
    # merge here was the root cause of tuned transcripts having 56% more
    # text (mostly duplicates) than the untuned ones on Elastrin.
    print(f"[{time.strftime('%H:%M:%S')}] Merging overlap zones")
    words = ct.merge_overlapping_words(words)
    print(f"  {len(words)} words after overlap partition")

    # Also load the audio duration for the tail-gap guard.
    with sf.SoundFile(str(audio_path)) as f:
        audio_duration_sec = f.frames / f.samplerate

    print(f"[{time.strftime('%H:%M:%S')}] Running diarization: "
          f"{args.diarization_model}, num_speakers={args.num_speakers}")
    t0 = time.time()
    turns = ct.run_diarization(str(audio_path), hf_token,
                               args.diarization_model, args.num_speakers)
    dia_sec = time.time() - t0
    print(f"  diarization took {dia_sec/60:.1f} min, {len(turns)} turns")

    spk_dur = {}
    for t in turns:
        spk_dur[t["speaker"]] = spk_dur.get(t["speaker"], 0) + (t["end"] - t["start"])
    total_speech = sum(spk_dur.values())
    print(f"  speakers detected: {len(spk_dur)}, total speech "
          f"{total_speech:.1f} s ({total_speech/60:.1f} min)")
    for spk, dur in sorted(spk_dur.items(), key=lambda x: -x[1]):
        pct = dur / total_speech * 100 if total_speech else 0
        print(f"    {spk}: {dur:.1f}s ({pct:.0f}%)")

    print(f"[{time.strftime('%H:%M:%S')}] Smoothing speakers "
          f"(window={args.smoothing_sec}s)")
    words = ct.assign_speakers_smoothed(words, turns, args.smoothing_sec)
    segments = ct.build_segments(words)
    print(f"  {len(segments)} segments")

    print(f"[{time.strftime('%H:%M:%S')}] Writing outputs: {out_base}.*")
    ct.write_transcript_txt(transcript_path, segments)
    ct.write_srt(Path(out_base + ".srt"), segments)
    ct.write_segments_json(Path(out_base + ".segments.json"), segments)
    ct.write_transcript_txt(Path(out_base + ".final.txt"), segments)
    ct.write_jsonl(Path(out_base + ".final.jsonl"), segments)
    ct.write_speakers_json(Path(out_base + ".speakers.json"), segments, turns)
    with open(out_base + ".diarization.json", "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2, ensure_ascii=False)

    audit = ct.audit_pyannote_coverage(turns, segments, 0.5, 2.0)
    audit_meta = dict(audit)
    audit_meta["diarization_model"] = args.diarization_model
    audit_meta["num_speakers_setting"] = args.num_speakers
    audit_meta["smoothing_sec"] = args.smoothing_sec
    audit_meta["num_speakers_detected"] = len(spk_dur)
    audit_meta["speakers_time"] = spk_dur
    audit_meta["audio_duration_sec"] = audio_duration_sec
    with open(out_base + ".coverage.json", "w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2, ensure_ascii=False)

    print(f"  coverage: {audit['coverage_pct']:.2f}%, {len(audit['gaps'])} gaps >= 2s")

    # Tail-gap guard (same as main pipeline).
    warnings = []
    ok, tail_gap_sec = ct.check_tail_gap(segments, audio_duration_sec, 30.0)
    if not ok:
        w = {
            "model": args.model_name,
            "severity": "critical",
            "kind": "trailing_gap",
            "audio_duration_sec": audio_duration_sec,
            "last_segment_end_sec": audio_duration_sec - tail_gap_sec,
            "trailing_gap_sec": tail_gap_sec,
            "message": (
                f"tuned transcript ends {tail_gap_sec:.1f}s before end of audio. "
                f"Was Elastrin's silent data-loss failure mode."
            ),
        }
        warnings.append(w)
        print(f"  CRITICAL: tail gap {tail_gap_sec:.1f}s")
    if audit["coverage_pct"] < 95.0:
        warnings.append({
            "model": args.model_name,
            "severity": "high",
            "kind": "coverage_below_threshold",
            "coverage_pct": audit["coverage_pct"],
            "threshold_pct": 95.0,
            "message": f"coverage {audit['coverage_pct']:.2f}% below 95% floor",
        })
        print(f"  WARNING: coverage {audit['coverage_pct']:.2f}% below 95% floor")
    with open(out_base + ".warnings.json", "w", encoding="utf-8") as f:
        json.dump(warnings, f, indent=2, ensure_ascii=False)

    # Per-segment speaker distribution for quick comparison. NOTE: word counts
    # are honest now because merge_overlapping_words removed the sliding-window
    # duplicates.
    seg_word_count = {}
    seg_time = {}
    for s in segments:
        wc = len(s["text"].split())
        seg_word_count[s["speaker"]] = seg_word_count.get(s["speaker"], 0) + wc
        seg_time[s["speaker"]] = seg_time.get(s["speaker"], 0) + (s["end"] - s["start"])
    print("  final speaker distribution (post-assign+smooth):")
    for spk in sorted(seg_word_count.keys(), key=lambda x: -seg_word_count[x]):
        label = ct.speaker_label(spk)
        print(f"    {label} ({spk}): {seg_word_count[spk]} words, "
              f"{seg_time[spk]:.1f}s")

    print(f"[{time.strftime('%H:%M:%S')}] Done. Tag: {args.tag}")


if __name__ == "__main__":
    main()
