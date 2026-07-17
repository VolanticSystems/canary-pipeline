"""Render speaker-diarized segments to transcript / SRT / JSON."""
import json


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


def write_transcript_txt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            f.write(f"[{fmt_hms(seg['start'])}] {speaker_label(seg['speaker'])}: {text}\n\n")


def write_srt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        cue = 0
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            cue += 1
            f.write(f"{cue}\n")
            f.write(f"{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}\n")
            f.write(f"{speaker_label(seg['speaker'])}: {text}\n\n")


def write_json(path, segments):
    out = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        out.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": speaker_label(seg["speaker"]),
            "speaker_raw": seg["speaker"],
            "text": text,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
