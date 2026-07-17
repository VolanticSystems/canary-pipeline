# canary-pipeline

Exhaustive speech-to-text and speaker diarization for long recordings —
depositions, business meetings, court conferences. Built after off-the-shelf
tooling (WhisperX) turned out to silently drop most of a real recording's
content, with no error and no warning.

## The problem that started this

A 78-minute deposition, transcribed with WhisperX (Whisper + VAD-gated
chunking, the standard open-source approach). The output looked plausible —
clean sentences, reasonable pacing. It was missing **57 of the 78 minutes**.

WhisperX runs a voice-activity detector *ahead of* transcription and only
feeds the model audio it classifies as speech. On real recordings — phone
compression, overlapping speakers, someone shuffling papers — that classifier
is wrong a lot, and everything it gets wrong is just gone. No gap marker, no
warning, no non-zero exit code. The transcript reads clean because the parts
it dropped are the parts you never see.

That's the failure mode this pipeline is built to make structurally
impossible: **nothing decides what counts as "speech" before transcription
runs.** The whole recording gets walked and transcribed, unconditionally.

## What it does differently

- **No VAD gate on transcription.** Audio is walked in fixed, heavily
  overlapping windows (30 s window, 20 s target overlap — every interior
  moment is covered by 2-3 independent transcription passes). Silence
  detection only chooses *where* a window boundary falls, never *whether*
  a window exists.
- **Two independent ASR models, not one.** Parakeet-TDT (CTC, less prone to
  hallucination) and Canary-1b-flash (encoder-decoder, higher raw accuracy)
  run on the same audio. Where they agree, that's high-confidence content.
  Where they disagree, the disagreement itself is surfaced for review instead
  of silently picking one.
- **Diarization is a separate concern from transcription.** pyannote labels
  who's speaking; it never gets a vote on whether a word gets transcribed.
  A word outside every detected speaker turn still ships — tagged unknown,
  not deleted.
- **Runs on rented GPU infrastructure (Vast.ai)**, orchestrated end-to-end:
  provision, transcribe, download, tear down. A full ~30-minute recording
  costs single-digit cents and finishes in well under 10 minutes with the
  current image.

## Two repos, one system

- **`canary-pipeline`** (this repo) — everything that runs on your own
  machine: audio prep, provisioning and monitoring the rented GPU, pulling
  results back, local diarization tuning, the test suite. Changes here are
  application logic and ship instantly — no rebuild required.
- **[`canary-asr`](https://github.com/VolanticSystems/canary-asr)** — the
  Docker image that runs *on* the rented GPU. NeMo, pyannote, a
  version-pinned CUDA-aligned PyTorch, and both models' weights baked in at
  build time. Changes here mean a new image and a ~20-minute CI build.

Splitting them means a config tweak or a bugfix in the orchestration logic
never triggers a 13 GB image rebuild, and a dependency bump in the ASR stack
never requires touching the Python that runs locally. Two different rates of
change, two different repos.

## How this got built: what off-the-shelf got wrong, and what got learned fixing it

This wasn't designed clean on a whiteboard. It was built, run against real
recordings, broken by real recordings, and hardened one incident at a time.
That's the part worth reading if you want to know how this actually holds up
under load, not just what it claims to do.

**1. The naive version didn't scale — at all.**
First cut called the ASR model's `transcribe()` once per audio window. NeMo's
per-call setup overhead (decoder init, kernel launch) turned an 80-minute
recording into a job that was still running after 50 minutes with no output.
Fix: batch every window into a single `transcribe()` call. Same 80 minutes of
audio, same two models, **under 3 minutes total** on a rented RTX 4090.

**2. A later run silently lost 5 minutes 42 seconds of audio — with a clean
exit code.**
The transcription loop checkpointed its progress every 5 batches but had no
final write after the loop finished. If the last batch wasn't a multiple of
5, its output lived only in memory and never reached disk. The bug shipped
once before anyone noticed, because nothing failed — the process just quietly
returned less than it should have.
Fix: checkpoint every batch, plus an explicit final write, plus a **hard
guard** — if the last transcribed segment ends more than 30 seconds before
the audio does, the run now writes a `critical` entry to a warnings file
instead of exiting clean. Silent data loss became structurally loud.

**3. Overlapping windows were duplicating text — up to 56% inflation.**
The overlap that makes the pipeline exhaustive (point above) creates a
correctness problem: the same word gets transcribed by 2-3 windows. The
first dedup pass matched on normalized text within a time window, which
missed paraphrases ("didn't" vs "did not") and broke down entirely under
3-way overlap.
Fix: rewrote it as **timestamp-ownership partitioning** — each window owns
the exact time range where its center is closest, and a word is kept only if
it falls inside its own window's owned range. No text comparison, no
paraphrase blind spot, correct by construction regardless of overlap depth.

**4. A domain-specific error class that's actually dangerous.**
On a medical/biotech recording, one model transcribed a term with a
`hyper-` prefix at a timestamp where the other model transcribed the same
term with `hypo-` — an ASR error that inverts clinical meaning rather than
just garbling a word. Neither model was "wrong" in an obviously detectable
way; you'd only catch it by listening.
Fix: a cross-model check that flags exactly this pattern — polarity-prefix
disagreement between the two models at the same timestamp — into its own
review file, rather than trusting either transcript by default.

**5. Infrastructure reliability was the last mile, and it mattered as much as
the model logic.**
Real production runs surfaced a run of smaller-but-real problems: shell
commands breaking on filenames with spaces (twice, in two different code
paths — both now regression-tested); an SSH-disconnect killing the whole job
because the transcription ran in the foreground of the same session; every
run burning 10-15 minutes re-downloading ~6 GB of model weights from a
throttled HuggingFace endpoint; and — the recurring one — **roughly half the
rented GPU hosts turning out to be bad**, either dying mid-pull or stuck in a
Docker retry loop with no forward progress.

Fixed, respectively: strict `shlex.quote()`-based command construction;
detached execution (`nohup` + polled completion flag, immune to SSH drops);
baking both models' weights into the Docker image at build time so runtime
never touches HuggingFace; and a **stuck-host detector** that distinguishes
"genuinely slow but progressing" from "confirmed dead" (same failure message
repeating with zero forward progress for 40+ seconds) and automatically
destroys and re-provisions on a fresh host — no human has to notice and
intervene. This was live-validated against real Vast.ai infrastructure, not
just simulated: it caught and correctly rerolled a real bad host mid-session.

## Instrumentation

**Why this matters, concretely:** a transcript that reads cleanly tells you
nothing about whether it's *complete*. That's the entire premise this repo
started from (see the problem statement at the top) — WhisperX's output also
read cleanly, right up until someone checked it against the source audio and
found 57 missing minutes. A pipeline that can silently drop content and
still exit 0 hasn't actually solved that problem; it's just moved the same
failure mode somewhere less visible. So every run here produces
machine-readable diagnostics alongside the transcript, and every one of them
traces back to a specific incident from the build journey above — this
isn't a checklist added for appearances, it's the accumulated scar tissue
from things that actually went wrong once:

| File | Purpose | Born from |
|---|---|---|
| `warnings.json` | Coverage floor / trailing-gap alerts. `[]` = clean run. | Incident #2 — the silent 5:42 tail-loss |
| `coverage.json` | % of detected speech actually covered by the transcript, per model | Same incident — the metric the guard checks against |
| `semantic_inversions.json` | Cross-model polarity-prefix disagreements | Incident #4 — the hypo/hyper meaning-inversion risk |
| `hallucinations.json` | Pattern-matched ASR training-data leakage ("thank you for watching," etc.) | General ASR failure mode, checked on every run regardless of incident history |
| `*.speakers.json` | Per-speaker word count + diarized time, for hand-labeling | Practical need — mapping anonymous speaker IDs to real names |

If you're evaluating whether to trust a transcript from this pipeline, start
with `warnings.json`. `[]` means the two hard guards (coverage floor,
trailing-gap detector) didn't trip. Anything else means read it before you
read the transcript.

**118 automated tests**, ~3 seconds to run, zero GPU/NeMo/pyannote
dependency — pure logic and string-construction tests, including a dedicated
regression test for every incident above: `test_guards.py` (#2), `test_merge.py`
(#3), `test_provision_quoting.py` (#5's shell-quoting half), and
`test_stuck_host.py` (#5's bad-host half). The image these tests validate
against has its own build history worth reading — see
[`canary-asr`'s evolution](https://github.com/VolanticSystems/canary-asr#how-this-image-evolved)
for the environment-level half of this story.

## Quickstart

```bash
# 1. Prep audio (MP4 or WAV — ffmpeg handles both).
python prep_audio.py "/path/to/audio.mp4"

# 2. Run end-to-end on Vast: provision, transcribe, download, destroy.
python vast_provision.py "/path/to/audio.prepped.wav"
# outputs land next to the input WAV — per model (canary + parakeet):
#   *.final.txt / *.final.jsonl   — canonical transcript
#   *.srt                          — subtitles
#   *.segments.json                — structured, per-turn
#   *.speakers.json                — per-speaker summary
# plus shared: *.diarization.json, *.coverage.json,
#              *.semantic_inversions.json, *.warnings.json  <- check this first

# 3. Optional: re-tune diarization locally, no GPU rental needed.
python tune_diarization.py "/path/to/audio.prepped.wav" \
    "/path/to/audio.prepped.parakeet.checkpoint.json" \
    --num-speakers 3 --tag tuned_3spk
```

A ~30-minute recording: ~5 minutes to provision, well under 3 minutes to
transcribe both models, ~1 minute to download results. Bad-host detection
and datacenter-preferred host selection run automatically — no babysitting
required.

## Tests

```bash
python -m pytest tests/
```

118 tests across 10 files, ~3 second runtime. See `tests/` for the full
breakdown — window planning, overlap partitioning, speaker assignment,
segment building, coverage audit, hallucination detection, shell-command
construction, and stuck-host detection, all exercised with synthetic
fixtures so the suite runs with no external dependencies at all.

## Repository layout

```
canary-pipeline/
├── canary_transcribe.py    # remote-side driver — runs inside the Docker image
├── vast_provision.py       # local orchestrator — provision/upload/poll/download/destroy
├── tune_diarization.py     # local — re-tune diarization from cached checkpoints
├── prep_audio.py           # local — ffmpeg audio prep
├── config.yaml             # models, windowing, thresholds, vocabulary corrections
└── tests/                  # 118 tests, no GPU required
```

## License

MIT — see `LICENSE`.
