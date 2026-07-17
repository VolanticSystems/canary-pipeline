"""
Remote-side (Vast.ai) transcription + diarization driver.

Robustness invariants (introduced after PeteDepo run #6 fiasco 2026-06-25):

  - **Batched transcription.** NeMo's transcribe() is called ONCE per batch
    of N chunks (default 32), not once per chunk. Per-call overhead amortizes.

  - **Per-batch progress log.** Every batch completion appends a line to
    `progress.log` (next to the input WAV). Tail-able from outside. Survives
    SSH disconnects because the python is detached.

  - **Hypothesis shape validation on warmup.** Before the real loop, do one
    warmup transcribe and verify the returned object has the expected
    word-timestamp structure. Abort loudly if not.

  - **GPU verification on model load.** Check `next(model.parameters()).device`
    is CUDA after `.to(cuda)`. Abort if the model is silently on CPU.

  - **Periodic checkpointing.** Every 5 batches, dump accumulated words to a
    `.{model}.checkpoint.json` sidecar. Future restart can resume.

  - **Done flag.** When the script completes successfully, write a `done.flag`
    file. The orchestrator polls for this file instead of waiting on a long-
    running SSH session.

Usage on the remote box (typically via nohup from the provisioner):
    python canary_transcribe.py <prepped.wav> [--config config.yaml]
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def speaker_label(spk):
    if not spk or spk == "SPEAKER_UNKNOWN":
        return "Speaker ?"
    try:
        n = int(spk.split("_")[1]) + 1
        return f"Speaker {n}"
    except (IndexError, ValueError):
        return spk


def fmt_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Silence detection (energy-based)
# ---------------------------------------------------------------------------

def find_silences(audio, sr, threshold_db, min_duration_ms):
    """Find silent regions via RMS energy. Returns list of (start_sample, end_sample)."""
    frame_ms = 50
    frame_samples = max(1, int(sr * frame_ms / 1000))
    threshold_lin = 10 ** (threshold_db / 20)
    min_samples = int(sr * min_duration_ms / 1000)

    silences = []
    in_silence = False
    silence_start = 0
    for i in range(0, len(audio) - frame_samples + 1, frame_samples):
        rms = float(np.sqrt(np.mean(audio[i:i + frame_samples] ** 2)))
        is_silent = rms < threshold_lin
        if is_silent and not in_silence:
            in_silence = True
            silence_start = i
        elif not is_silent and in_silence:
            in_silence = False
            if (i - silence_start) >= min_samples:
                silences.append((silence_start, i))
    if in_silence and (len(audio) - silence_start) >= min_samples:
        silences.append((silence_start, len(audio)))
    return silences


def compute_windows(num_samples, sr, window_sec, target_overlap_sec, min_overlap_sec,
                    silences, anchor_tolerance_sec):
    """Compute silence-anchored window boundaries.

    Returns list of (start_sample, end_sample). Window edges snap to silences
    within +/- anchor_tolerance_sec. Falls back to target time if no silence
    found in that tolerance. Enforces a minimum overlap floor between windows."""
    window_samples = int(window_sec * sr)
    target_step = window_samples - int(target_overlap_sec * sr)
    min_overlap_samples = int(min_overlap_sec * sr)
    tolerance_samples = int(anchor_tolerance_sec * sr)

    silence_midpoints = [(s + e) // 2 for s, e in silences]

    def snap(target_sample):
        if not silence_midpoints:
            return target_sample
        best = target_sample
        best_dist = tolerance_samples + 1
        for mid in silence_midpoints:
            d = abs(mid - target_sample)
            if d <= tolerance_samples and d < best_dist:
                best = mid
                best_dist = d
            if mid - target_sample > tolerance_samples:
                break
        return best

    windows = []
    pos = 0
    safety_count = 0
    while pos < num_samples and safety_count < 10_000:
        safety_count += 1
        target_end = pos + window_samples
        if target_end >= num_samples:
            end = num_samples
        else:
            end = snap(target_end)
        windows.append((pos, end))
        if end >= num_samples:
            break
        next_target_start = end - int(target_overlap_sec * sr)
        next_start = snap(next_target_start)
        max_next_start = end - min_overlap_samples
        if next_start > max_next_start:
            next_start = max_next_start
        if next_start <= pos:
            next_start = pos + max(1, target_step // 2)
        pos = next_start

    # Drop a tiny tail window if any
    while windows and (windows[-1][1] - windows[-1][0]) < int(sr * 0.3):
        windows.pop()
    return windows


# ---------------------------------------------------------------------------
# Model loading + verification
# ---------------------------------------------------------------------------

def load_asr_model(model_cfg):
    loader = model_cfg.get("loader", "ASRModel")
    model_id = model_cfg["id"]
    log(f"Loading ASR: {model_id} ({loader})")
    if loader == "EncDecMultiTaskModel":
        from nemo.collections.asr.models import EncDecMultiTaskModel
        model = EncDecMultiTaskModel.from_pretrained(model_id)
        try:
            decode_cfg = model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            model.change_decoding_strategy(decode_cfg)
            log("  set beam_size=1 (greedy decoding)")
        except Exception as e:
            log(f"  WARN: could not set beam_size=1: {e}")
    else:
        from nemo.collections.asr.models import ASRModel
        model = ASRModel.from_pretrained(model_id)
    model = model.to(torch.device("cuda"))
    model.eval()

    # Verify model is actually on GPU.
    p = next(model.parameters())
    if p.device.type != "cuda":
        raise RuntimeError(f"Model is on {p.device}, not CUDA — refusing to proceed (would be CPU-slow)")
    log(f"  model on {p.device} ({torch.cuda.get_device_name(p.device)})")
    log(f"  GPU mem after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model


def validate_hypothesis_shape(hyp, model_name):
    """Abort loudly if the warmup hypothesis doesn't have the expected
    word-timestamp structure. Better to fail fast than silently emit zero
    timestamps for the entire run."""
    if hyp is None:
        raise RuntimeError(f"{model_name}: transcribe returned None on warmup")
    ts = getattr(hyp, "timestamp", None)
    if ts is None and isinstance(hyp, dict):
        ts = hyp.get("timestamp")
    if not ts or not isinstance(ts, dict):
        raise RuntimeError(
            f"{model_name}: hypothesis.timestamp is not a dict (got {type(ts).__name__})")
    if "word" not in ts:
        raise RuntimeError(
            f"{model_name}: hypothesis.timestamp has no 'word' key (got {list(ts.keys())})")
    words = ts["word"] or []
    if not words:
        log(f"  WARN: {model_name}: warmup produced zero words (silent audio?). Proceeding.")
        return
    first = words[0]
    if not isinstance(first, dict):
        raise RuntimeError(
            f"{model_name}: word entry is not a dict (got {type(first).__name__})")
    needed = ["word", "start", "end"]
    missing = [k for k in needed if k not in first]
    if missing:
        raise RuntimeError(
            f"{model_name}: word dict missing keys {missing}; got {list(first.keys())}")
    log(f"  first word sample: {first}")


def extract_word_timestamps(hyp):
    if hyp is None:
        return []
    ts = getattr(hyp, "timestamp", None)
    if ts is None and isinstance(hyp, dict):
        ts = hyp.get("timestamp")
    if not ts or not isinstance(ts, dict):
        return []
    return list(ts.get("word", []) or [])


# ---------------------------------------------------------------------------
# Batched transcription with progress logging
# ---------------------------------------------------------------------------

def batched_transcribe(model, model_name, audio, sr, windows, transcribe_kwargs,
                       batch_size, progress_path, checkpoint_path):
    """Run all windows through the model in batches of `batch_size`. Returns
    a flat list of word dicts with absolute timestamps."""
    # Slice all chunks up front.
    chunks = []
    for w_start, w_end in windows:
        chunks.append(audio[w_start:w_end])

    n_chunks = len(chunks)
    n_batches = (n_chunks + batch_size - 1) // batch_size

    # Warmup with one tiny chunk first.
    log(f"  warming up {model_name}...")
    t0 = time.time()
    warmup_chunk = audio[:min(len(audio), int(sr * 5))]
    warmup_out = model.transcribe([warmup_chunk], batch_size=1, **transcribe_kwargs)
    warmup_sec = time.time() - t0
    log(f"  warmup took {warmup_sec:.2f} sec")
    validate_hypothesis_shape(warmup_out[0] if warmup_out else None, model_name)

    all_words = []
    run_start = time.time()

    with open(progress_path, "w") as pf:
        pf.write(f"# {model_name} progress log; one line per batch\n")
        pf.write(f"# columns: batch_idx, chunks_done, total_chunks, batch_wall_sec, "
                 f"cum_sec, eta_sec, words_so_far\n")
        pf.flush()

        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, n_chunks)
            batch_chunks = chunks[b_start:b_end]
            t0 = time.time()
            try:
                out = model.transcribe(
                    batch_chunks, batch_size=len(batch_chunks), **transcribe_kwargs)
            except Exception as e:
                log(f"  batch {b_idx} ({b_start}..{b_end}) FAILED: {type(e).__name__}: {e}")
                batch_wall = time.time() - t0
                pf.write(f"{b_idx}, {b_end}, {n_chunks}, {batch_wall:.2f}, FAIL\n")
                pf.flush()
                continue
            batch_wall = time.time() - t0

            for j, hyp in enumerate(out):
                window_idx = b_start + j
                w_start_samp, w_end_samp = windows[window_idx]
                win_start_sec = w_start_samp / sr
                win_end_sec = w_end_samp / sr
                win_center_sec = (win_start_sec + win_end_sec) / 2
                for w in extract_word_timestamps(hyp):
                    ws = float(w.get("start", 0.0))
                    we = float(w.get("end", ws))
                    wt = str(w.get("word", "")).strip()
                    if not wt:
                        continue
                    all_words.append({
                        "abs_start": ws + win_start_sec,
                        "abs_end": we + win_start_sec,
                        "text": wt,
                        "window_idx": window_idx,
                        "window_center": win_center_sec,
                    })

            cum_sec = time.time() - run_start
            eta_sec = (cum_sec / b_end * (n_chunks - b_end)) if b_end else 0.0
            line = (f"{b_idx}, {b_end}, {n_chunks}, "
                    f"{batch_wall:.2f}, {cum_sec:.1f}, {eta_sec:.1f}, {len(all_words)}")
            pf.write(line + "\n")
            pf.flush()
            log(f"  batch {b_idx+1}/{n_batches} ({b_end}/{n_chunks}): "
                f"{batch_wall:.2f}s, ETA {eta_sec/60:.1f}min, words={len(all_words)}")

            # Checkpoint after every batch. Previously wrote every 5 batches
            # with NO final write, so the tail could be lost silently if the
            # last batch wasn't the 5th. Elastrin First Meeting hit this and
            # lost 5:42 of audio from the tuned outputs. Fix: write every time.
            tmp = checkpoint_path + ".tmp"
            with open(tmp, "w") as cf:
                json.dump(all_words, cf)
            os.replace(tmp, checkpoint_path)

    # Belt AND suspenders: explicit final write after the loop, even if the
    # per-batch write above got skipped for any reason.
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "w") as cf:
        json.dump(all_words, cf)
    os.replace(tmp, checkpoint_path)

    log(f"  {model_name} total: {(time.time()-run_start)/60:.1f} min, {len(all_words)} words")
    return all_words


# ---------------------------------------------------------------------------
# Overlap merge
# ---------------------------------------------------------------------------

_NORM_RE = re.compile(r"[^a-z0-9]")


def normalize_text(s):
    return _NORM_RE.sub("", s.lower())


def merge_overlapping_words(all_words):
    """Partition-based dedup of overlapping windows.

    Each window "owns" the time range where its center is closer than any
    other window's center. Words are kept only if their timestamp lies in
    their emitting window's owned range.

    Correctness: with 30 s windows stepped every 10 s (67% overlap), a
    single audio moment is inside up to 3 windows. Center-distance splits
    give each interior window a clean 10-second ownership band; word
    duplicates naturally fall out.

    This replaces an earlier text-match approach that missed duplicates
    when the ASR paraphrased slightly across windows ("did not" vs
    "didn't") and silently doubled words like `didn't didn't`. See
    TRANSCRIPTION-ISSUES.md (2026-07-10) for the incident.
    """
    if not all_words:
        return []

    # Collect the unique (window_idx, window_center) pairs.
    centers_by_idx = {}
    for w in all_words:
        centers_by_idx[w["window_idx"]] = w["window_center"]

    # Sort windows by center time.
    sorted_windows = sorted(centers_by_idx.items(), key=lambda kv: kv[1])
    win_idxs = [wi for wi, _ in sorted_windows]
    win_centers = [c for _, c in sorted_windows]

    # For each window, compute [lo, hi) ownership interval using midpoints
    # between adjacent window centers.
    owner_range = {}
    for i, wi in enumerate(win_idxs):
        c = win_centers[i]
        lo = -float("inf") if i == 0 else (win_centers[i - 1] + c) / 2.0
        hi = float("inf") if i == len(win_idxs) - 1 else (c + win_centers[i + 1]) / 2.0
        owner_range[wi] = (lo, hi)

    kept = []
    dropped = 0
    for w in all_words:
        lo, hi = owner_range[w["window_idx"]]
        if lo <= w["abs_start"] < hi:
            kept.append(w)
        else:
            dropped += 1

    kept.sort(key=lambda w: w["abs_start"])
    log(f"  merged: {len(kept)} words kept, {dropped} duplicates dropped by ownership partition")
    return kept


# ---------------------------------------------------------------------------
# Diarization (pyannote 3.x)
# ---------------------------------------------------------------------------

def run_diarization(audio_path, hf_token, model_id, num_speakers):
    from pyannote.audio import Pipeline
    log(f"Loading pyannote: {model_id}")
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
    if pipeline is None:
        raise RuntimeError(
            "Pipeline.from_pretrained returned None — bad HF token or terms not accepted")
    pipeline = pipeline.to(torch.device("cuda"))
    log("Running diarization on full audio")
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    diarization = pipeline(audio_path, **kwargs)
    turns = []
    for turn, _, spk in diarization.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)})
    log(f"  turns: {len(turns)}")
    return turns


# ---------------------------------------------------------------------------
# Speaker assignment with rolling-window smoothing
# ---------------------------------------------------------------------------

def assign_speakers_smoothed(words, turns, smoothing_sec):
    if not words:
        return words
    turns_sorted = sorted(turns, key=lambda t: t["start"])
    half = smoothing_sec / 2.0
    for w in words:
        word_center = (w["abs_start"] + w["abs_end"]) / 2
        win_start = word_center - half
        win_end = word_center + half
        tally = {}
        for t in turns_sorted:
            if t["start"] > win_end:
                break
            if t["end"] < win_start:
                continue
            overlap = min(win_end, t["end"]) - max(win_start, t["start"])
            if overlap > 0:
                tally[t["speaker"]] = tally.get(t["speaker"], 0.0) + overlap
        w["speaker"] = max(tally.items(), key=lambda x: x[1])[0] if tally else "SPEAKER_UNKNOWN"
    return words


def build_segments(words):
    if not words:
        return []
    segments = []
    cur = {
        "start": words[0]["abs_start"],
        "end": words[0]["abs_end"],
        "speaker": words[0]["speaker"],
        "text": words[0]["text"].strip(),
    }
    for w in words[1:]:
        if w["speaker"] == cur["speaker"]:
            cur["end"] = w["abs_end"]
            cur["text"] = (cur["text"] + " " + w["text"].strip()).strip()
        else:
            segments.append(cur)
            cur = {
                "start": w["abs_start"],
                "end": w["abs_end"],
                "speaker": w["speaker"],
                "text": w["text"].strip(),
            }
    segments.append(cur)
    return segments


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------

def _sorted_segments(segments):
    """Defensive sort: any writer should tolerate an unsorted list without
    producing an out-of-order transcript. The tuning path used to skip merge
    (which was the only sort) and shipped segments with 266 inversions."""
    return sorted(segments, key=lambda s: (s["start"], s["end"]))


def write_transcript_txt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for seg in _sorted_segments(segments):
            text = seg["text"].strip()
            if not text:
                continue
            f.write(f"[{fmt_hms(seg['start'])}] {speaker_label(seg['speaker'])}: {text}\n\n")


def write_srt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        cue = 0
        for seg in _sorted_segments(segments):
            text = seg["text"].strip()
            if not text:
                continue
            cue += 1
            f.write(f"{cue}\n")
            f.write(f"{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}\n")
            f.write(f"{speaker_label(seg['speaker'])}: {text}\n\n")


def write_segments_json(path, segments):
    out = []
    for seg in _sorted_segments(segments):
        out.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker_raw": seg["speaker"],
            "speaker": speaker_label(seg["speaker"]),
            "text": seg["text"].strip(),
        })
    # ensure_ascii=False keeps UTF-8 characters as-is instead of \u escapes;
    # matches the plain-UTF-8 contract the human-readable outputs use.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def write_jsonl(path, segments):
    """One JSON record per turn, newline-delimited. Canonical machine-readable
    output: monotonic, sorted, plain UTF-8."""
    with open(path, "w", encoding="utf-8") as f:
        for seg in _sorted_segments(segments):
            text = seg["text"].strip()
            if not text:
                continue
            rec = {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": speaker_label(seg["speaker"]),
                "text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_speakers_json(path, segments, turns):
    """Per-speaker aggregate for hand-labeling. Speaking time comes from
    the DIARIZATION intervals (not the transcript segments, which are
    inflated by any residual duplication) — the honest denominator."""
    speaker_time = {}
    for t in turns:
        speaker_time[t["speaker"]] = speaker_time.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    speaker_words = {}
    speaker_preview = {}
    for seg in _sorted_segments(segments):
        spk = seg["speaker"]
        wc = len(seg["text"].split())
        speaker_words[spk] = speaker_words.get(spk, 0) + wc
        if spk not in speaker_preview and seg["text"].strip():
            speaker_preview[spk] = seg["text"].strip()[:400]
    out = []
    for spk in sorted(set(list(speaker_time) + list(speaker_words))):
        out.append({
            "speaker_raw": spk,
            "speaker": speaker_label(spk),
            "diarized_seconds": speaker_time.get(spk, 0.0),
            "transcript_word_count": speaker_words.get(spk, 0),
            "first_sample_text": speaker_preview.get(spk, ""),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def apply_vocab_corrections(segments, corrections):
    """Post-process segment text with a safe substitution map. Substitutions
    are case-insensitive and word-boundary anchored so partial matches inside
    other words are NOT touched.

    corrections: dict of {heard: intended}. See config.yaml `vocab_corrections`.
    """
    if not corrections:
        return segments
    compiled = []
    for heard, intended in corrections.items():
        # \b at Python re treats letter/digit boundary. Good enough for names.
        pat = re.compile(r"\b" + re.escape(heard) + r"\b", re.IGNORECASE)
        compiled.append((pat, intended))
    for seg in segments:
        text = seg["text"]
        for pat, intended in compiled:
            text = pat.sub(intended, text)
        seg["text"] = text
    return segments


def detect_semantic_inversions(segments, canary_segments):
    """Cross-model semantic-inversion detection.

    Flag places where the two ASR models disagree on a hypo/hyper (or similar
    polarity-inverting) prefix at the same time. In the Elastrin recording,
    the speaker corrected himself from `hyper`- to `hypo`- but the ASR only
    caught one variant, inverting the medical meaning. A downstream reader
    can't spot this without listening to the tape.

    Returns a list of {time, speaker, parakeet_word, canary_word} suspects.
    """
    if not canary_segments:
        return []
    # Index canary text by rough time
    canary_by_time = []
    for seg in canary_segments:
        for word in seg["text"].split():
            canary_by_time.append((seg["start"], word.lower()))
    canary_by_time.sort()

    prefixes = [
        ("hyper", "hypo"),
        ("hypo", "hyper"),
        ("in", "out"), ("out", "in"),
    ]
    interesting = ("hyper", "hypo")

    flags = []
    for seg in segments:
        for word in seg["text"].split():
            w = word.lower()
            for pref in interesting:
                if pref in w:
                    # Find canary words within ±3s carrying the opposite prefix
                    opposite = "hypo" if pref == "hyper" else "hyper"
                    for t, cw in canary_by_time:
                        if abs(t - seg["start"]) > 3:
                            continue
                        if opposite in cw and pref not in cw:
                            flags.append({
                                "time": float(seg["start"]),
                                "speaker": speaker_label(seg["speaker"]),
                                "parakeet_word": word,
                                "canary_word": cw,
                                "note": (
                                    f"parakeet emitted '{pref}...', canary emitted "
                                    f"'{opposite}...' at same time. Listen to verify."
                                ),
                            })
                            break
    return flags


def check_tail_gap(segments, audio_duration_sec, max_trailing_gap=30.0):
    """Return (ok: bool, gap_sec: float). Fails if the last transcribed segment
    ends more than `max_trailing_gap` seconds before the end of the audio.

    This exists because Elastrin's tuned outputs silently ended 5:42 before the
    audio did, and nothing in the pipeline complained. Now the pipeline logs
    a loud warning AND writes a `.warnings.json` so the caller can gate on it.
    """
    if not segments:
        return False, audio_duration_sec
    last_end = max(float(s["end"]) for s in segments)
    gap = audio_duration_sec - last_end
    return gap <= max_trailing_gap, gap


# ---------------------------------------------------------------------------
# Hallucination spot-check
# ---------------------------------------------------------------------------

def detect_hallucinations(segments, patterns):
    compiled = [re.compile(p) for p in patterns]
    flags = []
    for seg in segments:
        text = seg["text"]
        for pat in compiled:
            m = pat.search(text)
            if m:
                flags.append({
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "speaker": speaker_label(seg["speaker"]),
                    "pattern": pat.pattern,
                    "matched_text": m.group(0),
                    "segment_text": text,
                })
                break
    return flags


# ---------------------------------------------------------------------------
# Coverage audit (Option D — pyannote turns as ground truth)
# Simple O(N*M) implementation; for our sizes (hundreds of turns, thousands of
# segments) this is <1 sec. Was clever and probably wrong before.
# ---------------------------------------------------------------------------

def audit_pyannote_coverage(turns, segments, edge_grace_sec, min_gap_sec):
    if not turns:
        return {"speech_seconds": 0.0, "covered_seconds": 0.0,
                "uncovered_seconds": 0.0, "coverage_pct": 100.0, "gaps": []}

    seg_intervals = sorted([(float(s["start"]), float(s["end"])) for s in segments])
    total_speech = 0.0
    total_covered = 0.0
    gaps = []

    for t in turns:
        t_start = float(t["start"])
        t_end = float(t["end"])
        total_speech += (t_end - t_start)

        # Find all seg overlaps within this turn, union them.
        overlaps = []
        for ss, se in seg_intervals:
            if se <= t_start:
                continue
            if ss >= t_end:
                break
            overlaps.append((max(ss, t_start), min(se, t_end)))

        # Union sorted intervals.
        overlaps.sort()
        merged = []
        for s, e in overlaps:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        covered_in_turn = sum(e - s for s, e in merged)
        total_covered += covered_in_turn

        # Find gaps (uncovered runs) within this turn.
        cursor = t_start
        for s, e in merged:
            if s - cursor > edge_grace_sec:
                gap_dur = s - cursor
                if gap_dur >= min_gap_sec:
                    gaps.append({"start": cursor, "end": s,
                                 "duration": gap_dur, "speaker": t["speaker"]})
            cursor = e
        if t_end - cursor > edge_grace_sec:
            gap_dur = t_end - cursor
            if gap_dur >= min_gap_sec:
                gaps.append({"start": cursor, "end": t_end,
                             "duration": gap_dur, "speaker": t["speaker"]})

    coverage_pct = (total_covered / total_speech * 100) if total_speech else 100.0
    return {
        "speech_seconds": total_speech,
        "covered_seconds": total_covered,
        "uncovered_seconds": total_speech - total_covered,
        "coverage_pct": coverage_pct,
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def per_model_path(base, model_name, suffix):
    return Path(f"{base}.{model_name}{suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.is_file():
        print(f"ERROR: audio not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    base = str(audio_path.with_suffix(""))
    out_diarization = Path(base + config["output_suffixes"]["diarization"])
    out_coverage = Path(base + config["output_suffixes"]["coverage"])
    out_done_flag = Path(base + ".done.flag")
    per_model_suffixes = config["output_suffixes"]["per_model"]

    # Wipe any prior done flag so the orchestrator can't false-trigger on a stale one.
    try:
        out_done_flag.unlink()
    except FileNotFoundError:
        pass

    log(f"Loading audio: {audio_path}")
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    expected_sr = int(config["target_sample_rate"])
    if sr != expected_sr:
        raise RuntimeError(f"sample rate mismatch: {sr} vs expected {expected_sr}")
    duration = len(audio) / sr
    log(f"  duration: {duration:.1f} s ({duration/60:.1f} min)")

    log(f"Finding silences (threshold {config['silence_threshold_db']} dB, "
        f"min {config['silence_min_duration_ms']} ms)")
    silences = find_silences(audio, sr,
                             float(config["silence_threshold_db"]),
                             float(config["silence_min_duration_ms"]))
    log(f"  silences: {len(silences)}")
    log(f"Computing silence-anchored windows "
        f"({config['window_seconds']} s / {config['overlap_seconds']} s overlap, "
        f"min {config['overlap_min_seconds']} s)")
    windows = compute_windows(
        len(audio), sr,
        float(config["window_seconds"]),
        float(config["overlap_seconds"]),
        float(config["overlap_min_seconds"]),
        silences,
        float(config["silence_anchor_tolerance_seconds"]),
    )
    log(f"  windows: {len(windows)}")

    # Diarization, once, shared.
    turns = run_diarization(str(audio_path), hf_token,
                            config.get("diarization_model"),
                            config.get("num_speakers"))
    with open(out_diarization, "w") as f:
        json.dump(turns, f, indent=2)
    log(f"Wrote: {out_diarization}")

    transcribe_batch_size = int(config.get("transcribe_batch_size", 16))
    speaker_smoothing_sec = float(config["speaker_smoothing_window_seconds"])
    audit_edge_grace_sec = float(config["audit_edge_grace_seconds"])
    audit_min_gap_sec = float(config["audit_min_gap_seconds"])
    vocab_corrections = config.get("vocab_corrections") or {}
    coverage_min_pct = float(config.get("coverage_min_pct", 95.0))
    max_trailing_gap_sec = float(config.get("max_trailing_gap_seconds", 30.0))

    coverage_by_model = {}
    warnings = []  # aggregated across models — surfaced in .warnings.json
    segments_by_model = {}

    for model_cfg in config["models"]:
        name = model_cfg["name"]
        log(f"=== Model: {name} ({model_cfg['id']}) ===")
        model = load_asr_model(model_cfg)
        transcribe_kwargs = dict(model_cfg.get("transcribe_kwargs") or {})

        # Strip any batch_size already in transcribe_kwargs — we control it.
        transcribe_kwargs.pop("batch_size", None)

        progress_path = Path(f"{base}.{name}.progress.log")
        checkpoint_path = Path(f"{base}.{name}.checkpoint.json")

        all_words = batched_transcribe(
            model, name, audio, sr, windows, transcribe_kwargs,
            batch_size=transcribe_batch_size,
            progress_path=str(progress_path),
            checkpoint_path=str(checkpoint_path),
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        log(f"  merging overlap zones for {name}")
        words = merge_overlapping_words(all_words)
        log(f"  smoothing speaker assignment for {name}")
        words = assign_speakers_smoothed(words, turns, speaker_smoothing_sec)
        segments = build_segments(words)
        segments = apply_vocab_corrections(segments, vocab_corrections)
        log(f"  {name} segments: {len(segments)}")
        segments_by_model[name] = segments

        out_tx = per_model_path(base, name, per_model_suffixes["transcript"])
        out_srt = per_model_path(base, name, per_model_suffixes["srt"])
        out_seg = per_model_path(base, name, per_model_suffixes["segments"])
        out_hal = per_model_path(base, name, per_model_suffixes["hallucinations"])
        # Canonical outputs — deduplicated, sorted, plain-UTF-8, monotonic.
        out_final_txt = per_model_path(base, name, ".final.txt")
        out_final_jsonl = per_model_path(base, name, ".final.jsonl")
        out_speakers = per_model_path(base, name, ".speakers.json")
        write_transcript_txt(out_tx, segments)
        write_srt(out_srt, segments)
        write_segments_json(out_seg, segments)
        write_transcript_txt(out_final_txt, segments)
        write_jsonl(out_final_jsonl, segments)
        write_speakers_json(out_speakers, segments, turns)
        log(f"  wrote: {out_tx.name}, {out_srt.name}, {out_seg.name}, "
            f"{out_final_txt.name}, {out_final_jsonl.name}, {out_speakers.name}")

        hallucinations = detect_hallucinations(
            segments, config.get("hallucination_patterns", []))
        # Always write a valid JSON array (`[]`), never zero bytes. Distinguishes
        # "ran and found nothing" from "never ran."
        with open(out_hal, "w", encoding="utf-8") as f:
            json.dump(hallucinations, f, indent=2, ensure_ascii=False)
        log(f"  hallucination flags ({name}): {len(hallucinations)} -> {out_hal.name}")

        audit = audit_pyannote_coverage(
            turns, segments, audit_edge_grace_sec, audit_min_gap_sec)
        coverage_by_model[name] = audit
        log(f"  {name} coverage: {audit['coverage_pct']:.2f}% "
            f"(speech {audit['speech_seconds']:.1f}s, "
            f"covered {audit['covered_seconds']:.1f}s, "
            f"{len(audit['gaps'])} gaps >= {audit_min_gap_sec}s)")

        # Hard guards: coverage floor + trailing-gap.
        if audit["coverage_pct"] < coverage_min_pct:
            warnings.append({
                "model": name,
                "severity": "high",
                "kind": "coverage_below_threshold",
                "coverage_pct": audit["coverage_pct"],
                "threshold_pct": coverage_min_pct,
                "message": (
                    f"{name} coverage {audit['coverage_pct']:.2f}% is below the "
                    f"{coverage_min_pct}% floor. Transcript may be missing content."
                ),
            })
            log(f"  WARNING: {name} coverage below floor")
        ok, tail_gap_sec = check_tail_gap(segments, duration, max_trailing_gap_sec)
        if not ok:
            warnings.append({
                "model": name,
                "severity": "critical",
                "kind": "trailing_gap",
                "audio_duration_sec": duration,
                "last_segment_end_sec": duration - tail_gap_sec,
                "trailing_gap_sec": tail_gap_sec,
                "message": (
                    f"{name} transcript ends {tail_gap_sec:.1f}s before end of audio "
                    f"({fmt_hms(duration - tail_gap_sec)} vs {fmt_hms(duration)}). "
                    f"This was Elastrin's silent data-loss failure mode."
                ),
            })
            log(f"  CRITICAL: {name} tail gap {tail_gap_sec:.1f}s")

    # Cross-model semantic-inversion check (hypo/hyper etc.). Requires both
    # canary and parakeet segments to compare.
    inversions = []
    if "parakeet" in segments_by_model and "canary" in segments_by_model:
        inversions = detect_semantic_inversions(
            segments_by_model["parakeet"], segments_by_model["canary"])
        if inversions:
            log(f"  {len(inversions)} semantic-inversion candidates flagged")
    out_inversions = Path(base + ".semantic_inversions.json")
    with open(out_inversions, "w", encoding="utf-8") as f:
        json.dump(inversions, f, indent=2, ensure_ascii=False)

    coverage_summary = {
        "duration_seconds": duration,
        "pyannote_speech_seconds": sum(t["end"] - t["start"] for t in turns),
        "coverage_min_pct_threshold": coverage_min_pct,
        "max_trailing_gap_seconds_threshold": max_trailing_gap_sec,
        "models": coverage_by_model,
    }
    with open(out_coverage, "w", encoding="utf-8") as f:
        json.dump(coverage_summary, f, indent=2, ensure_ascii=False)
    log(f"Wrote: {out_coverage}")

    # Central warnings file — never zero bytes; always a valid JSON array.
    out_warnings = Path(base + ".warnings.json")
    with open(out_warnings, "w", encoding="utf-8") as f:
        json.dump(warnings, f, indent=2, ensure_ascii=False)
    if warnings:
        log(f"WARNINGS: {len(warnings)} issue(s) surfaced -> {out_warnings.name}")
        for w in warnings:
            log(f"  [{w['severity']}] {w['message']}")
    else:
        log("No warnings.")

    # Final flag the orchestrator polls for.
    out_done_flag.write_text(f"done at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    log("Done.")


if __name__ == "__main__":
    main()
