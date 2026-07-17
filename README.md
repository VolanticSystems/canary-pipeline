# Canary / Parakeet ASR + Diarization Pipeline

Two-pass NeMo transcription on Vast.ai with pyannote speaker diarization, designed
to fix WhisperX's silent-dropout problem on real-world recordings. Built and
hardened across one long session in late June 2026.

## What this is

A pipeline that takes a long audio recording (depositions, conferences) and
produces speaker-labeled transcripts. Two ASR models (Canary + Parakeet) run on
the same audio for cross-validation. Diarization shared across both. Outputs land
next to the input audio.

The pipeline lives in two halves:

- **Local-side** (this folder): provisioner, audio prep, local pyannote tuning,
  tests. Runs on the user's box with whatever Python they've got.
- **Remote-side** (`canary_transcribe.py`): the heavy lifting. Runs on a rented
  Vast.ai GPU inside our slim Docker image
  (`ghcr.io/bob7123/canary-asr:v1`). NeMo + pyannote + a CUDA-aligned torch.

## File layout

```
canary/
├── README.md                  # this file
├── config.yaml                # model list, window/overlap, thresholds, patterns
├── prep_audio.py              # local ffmpeg: resample to 16k mono PCM
├── vast_provision.py          # local: orchestrate Vast (provision, scp, poll, destroy)
├── canary_transcribe.py       # remote: the actual transcription/diarization
├── tune_diarization.py        # local: re-run pyannote on cached words with new settings
├── pytest.ini                 # test config
├── tests/                     # 86 pytest tests; runs in 2 sec, no GPU needed
│   ├── conftest.py
│   ├── test_audit.py
│   ├── test_formatters.py
│   ├── test_hallucinations.py
│   ├── test_merge.py
│   ├── test_provision_quoting.py
│   ├── test_guards.py            # tail-gap, vocab, semantic-inversion, sorted writers
│   ├── test_silence.py
│   ├── test_speakers.py
│   └── test_windows.py
└── .tune_venv/                # isolated venv for local tuning + tests
```

## Standard workflow (a fresh recording)

```bash
# 1. Prep audio (MP4 or WAV input — ffmpeg handles both).
python prep_audio.py "/path/to/audio.mp4"
# produces: /path/to/audio.prepped.wav

# 2. Run on Vast (provisions, uploads, transcribes, downloads, destroys).
python vast_provision.py "/path/to/audio.prepped.wav"
# produces (next to the prepped WAV, per model = canary + parakeet):
#   audio.prepped.<model>.final.txt         # CANONICAL — read this
#   audio.prepped.<model>.final.jsonl       # CANONICAL — machine-readable
#   audio.prepped.<model>.transcript.txt    # legacy alias for final.txt
#   audio.prepped.<model>.srt               # subtitles
#   audio.prepped.<model>.segments.json     # per-segment JSON with raw speaker labels
#   audio.prepped.<model>.speakers.json     # per-speaker aggregate for hand-labeling
#   audio.prepped.<model>.hallucinations.json  # [] if clean
#   audio.prepped.<model>.checkpoint.json    # pre-merge word list, for retuning
#   audio.prepped.<model>.progress.log       # per-batch timing
# and shared:
#   audio.prepped.diarization.json           # pyannote speaker turns
#   audio.prepped.coverage.json              # Option D audit
#   audio.prepped.semantic_inversions.json   # hypo/hyper cross-model disagreements
#   audio.prepped.warnings.json              # coverage/tail-gap alerts. CHECK THIS.

# 3. Optional: tune the diarization locally without touching Vast.
python tune_diarization.py "/path/to/audio.prepped.wav" \
    "/path/to/audio.prepped.parakeet.checkpoint.json" \
    --num-speakers 3 --tag tuned_3spk
# produces audio.prepped.parakeet.tuned_3spk.{final.txt,final.jsonl,transcript.txt,
#          srt,segments.json,diarization.json,coverage.json,speakers.json,
#          warnings.json}
```

## Warnings file (CHECK EVERY RUN)

`audio.prepped.warnings.json` is written on every run. It's an array of alert
objects. `[]` means clean. If it's non-empty, don't trust the transcript
without reviewing:

- `severity: critical / kind: trailing_gap` — transcript ends more than 30 sec
  before end of audio. This is the Elastrin failure mode (silent 5:42 loss).
  Do not ship the transcript without investigating.
- `severity: high / kind: coverage_below_threshold` — coverage below 95%.
  Content may be missing. Look at `coverage.json` gaps.

Thresholds are set in `config.yaml` (`max_trailing_gap_seconds`, `coverage_min_pct`).

## Bad hosts are handled automatically

Vast rents you someone else's machine, and sometimes that machine is a dud —
dies mid-pull, or retry-loops on a Docker layer forever. `vast_provision.py`
detects both patterns and auto-destroys the bad instance, then tries a fresh
offer, up to 2 retries (3 total attempts) before giving up. You don't need to
watch the run or notice a stuck host yourself. `--max-host-retries N` to
change the retry count. `--resume <id>` skips this — resuming a specific
instance never auto-destroys it.

A genuinely slow-but-progressing pull is never touched — only a confirmed-bad
host (offline, or the same status message repeating with "retrying" in it for
40+ seconds) triggers the reroll.

Host selection also prefers **datacenter-verified** hosts (`datacenter=true`)
over residential/hobbyist boxes, falling back to the broader pool only if no
datacenter offer exists for the GPU/region combo. Costs a bit more per hour,
buys noticeably higher reliability and bandwidth.

Both of these were live-validated 2026-07-15 on a real Vast run, not just
unit-tested: the stuck-detector fired on a genuine bad host and correctly
rerolled, and weight-baking was confirmed present on disk before the
transcription script even started.

## What runs where

| Step | Location | Time | Notes |
|---|---|---|---|
| ffmpeg prep | Local | ~30 sec | CPU only |
| Vast provision + SSH ready | Vast | 5-15 min | Dominated by image pull on first-time hosts |
| pyannote diarization | Vast | ~3-5 min | Single GPU run on full audio |
| Canary transcription | Vast | ~1 min | Batched (467 chunks in one call) |
| Parakeet transcription | Vast | ~20 sec | Smaller, faster |
| Render outputs + audit | Vast | <5 sec | Pure CPU |
| Download outputs | Local | ~1 min | scp |
| Vast destroy linger | Vast | 5 min | Gives the human time to ssh and inspect |
| Local tuning re-render | Local | ~2 min per experiment | On a 3060 Ti |

Typical total: ~20 min from fresh audio to outputs on disk. Vast cost per run is
small change (cents to a quarter, depending on host speed).

## Tests

```
.tune_venv/Scripts/python -m pytest tests/
```

86 tests, runs in ~2 seconds. Covers everything in `canary_transcribe.py` and
the shell-command construction in `vast_provision.py`. Specifically guards
against the two filename-with-space bugs that broke real runs (the launch
command and the polling command).

These tests don't import NeMo or pyannote — pure data-shaping logic and string
construction. GPU not required.

## Key gotchas (the long version is in memory)

1. **NeMo's transcribe() has huge per-call overhead. ALWAYS batch.** Per-window
   calls turn a 1-minute job into hours of CPU-bound stalls. Pass all chunks in
   one call with `batch_size=N`.

2. **Filenames with spaces break shell commands.** Any user-controlled string in
   an SSH command must go through `shlex.quote()`. We have two helpers
   (`build_launch_command`, `build_poll_command`) and tests that lock the
   quoting in.

3. **SSH disconnects kill foreground python.** Always launch detached
   (`nohup ... < /dev/null > log 2>&1 &` in a subshell). Poll via fresh
   short-lived SSH sessions. Wait on a `*.done.flag` file.

4. **Pyannote 3.x is incompatible with huggingface_hub ≥0.24.** Pin
   `"huggingface_hub<0.24"` when setting up a local pyannote env.

5. **`pip install "nemo_toolkit[asr]"` upgrades torch to a CPU-only build.**
   Force-reinstall a matching CUDA trio after. This is baked into the Docker
   image; only matters if you're doing local NeMo setup (we don't).

6. **File-watcher race:** checkpoint files download AFTER transcripts. If your
   local script needs the checkpoints (like `tune_diarization.py`), don't trust
   the transcript file as the readiness signal — wait for the `*.done.flag` to
   land locally.

7. **Audio quality matters massively for diarization.** Teams-compressed audio →
   pyannote sees 22-24% of duration as speech. Direct iPhone mic → 54%. Get
   uncompressed audio when speaker labels matter.

## What to do when something breaks

- **Provisioner says "SSH never became responsive"** — destroy that instance,
  re-fire the provisioner. The host's daemon is broken; you'll get a new one.
- **Provisioner polls "RUNNING" forever after work clearly finished** — see
  bug #2 above. The polling command's path probably has a space. Verify
  `tests/test_provision_quoting.py` still passes.
- **`tune_diarization.py` says "checkpoint not found" right after a Vast run** —
  see bug #6. The checkpoints downloaded after the script started. Re-run.
- **Local pyannote errors on `Pipeline.from_pretrained`** — likely the
  huggingface_hub version. See bug #4.
- **Transcripts are missing big chunks of audio** — sanity-check by running
  `tune_diarization.py` on the checkpoint files; if those are sparse too,
  the audio quality is the problem (see bug #7). If checkpoints are dense
  but rendered transcripts are sparse, suspect a bug in `build_segments` or
  `assign_speakers_smoothed`.

## Related memory files

- See user-memory `canary-tooling` for the full deep dive on every gotcha
- See `whisperx-tooling` for the older pipeline this replaces and why
