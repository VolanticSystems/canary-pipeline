"""
Coverage audit. Uses webrtcvad to flag which frames are speech-like, then
checks what fraction of those frames are inside a transcribed segment.

VAD here is AUDIT-ONLY. It NEVER decides what reaches the transcriber.
"""
import wave


def audit(audio_path, segments, vad_aggressiveness=2, min_gap=2.0):
    import webrtcvad
    vad = webrtcvad.Vad(int(vad_aggressiveness))

    with wave.open(audio_path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        n_frames = w.getnframes()
        pcm = w.readframes(n_frames)

    if nch != 1 or sw != 2 or sr not in (8000, 16000, 32000, 48000):
        raise RuntimeError(
            f"webrtcvad needs mono 16-bit at 8/16/32/48 kHz, got "
            f"{nch}ch {sw*8}bit {sr}Hz"
        )

    frame_ms = 30
    frame_bytes = int(sr * (frame_ms / 1000) * 2)  # 16-bit mono
    n_full = len(pcm) // frame_bytes

    speech_mask = [False] * n_full
    for i in range(n_full):
        frame = pcm[i * frame_bytes:(i + 1) * frame_bytes]
        try:
            speech_mask[i] = vad.is_speech(frame, sr)
        except Exception:
            pass

    frame_sec = frame_ms / 1000.0
    total_seconds = n_full * frame_sec
    speech_frames = sum(speech_mask)
    speech_seconds = speech_frames * frame_sec

    cov_mask = [False] * n_full
    for seg in segments:
        s = max(0, int(seg["start"] / frame_sec))
        e = min(n_full, int(seg["end"] / frame_sec) + 1)
        for i in range(s, e):
            cov_mask[i] = True

    covered = sum(1 for i in range(n_full) if speech_mask[i] and cov_mask[i])
    uncovered_speech = speech_frames - covered

    coverage_pct = (covered / speech_frames * 100) if speech_frames else 100.0

    gaps = []
    in_gap = False
    gap_start = 0.0
    for i in range(n_full):
        if speech_mask[i] and not cov_mask[i]:
            if not in_gap:
                in_gap = True
                gap_start = i * frame_sec
        else:
            if in_gap:
                gap_end = i * frame_sec
                if gap_end - gap_start >= min_gap:
                    gaps.append({
                        "start": gap_start,
                        "end": gap_end,
                        "duration": gap_end - gap_start,
                    })
                in_gap = False
    if in_gap:
        gap_end = n_full * frame_sec
        if gap_end - gap_start >= min_gap:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "duration": gap_end - gap_start,
            })

    return {
        "total_seconds": total_seconds,
        "speech_seconds": speech_seconds,
        "speech_covered_seconds": covered * frame_sec,
        "speech_uncovered_seconds": uncovered_speech * frame_sec,
        "coverage_pct": coverage_pct,
        "vad_aggressiveness": int(vad_aggressiveness),
        "gaps": gaps,
    }
