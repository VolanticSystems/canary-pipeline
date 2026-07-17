"""Tests for energy-based silence detection."""
import numpy as np
import pytest

import canary_transcribe as ct


SR = 16000  # all tests use 16 kHz mono


def make_audio(segments):
    """Build a 1-D float32 audio array from a list of (duration_sec, amplitude) tuples.

    amplitude is 0 for silence, ~0.3 for "noisy/voice-like" content (well above the
    -45 dB threshold which is ~0.0056 linear).
    """
    chunks = []
    for dur, amp in segments:
        n = int(dur * SR)
        if amp == 0:
            chunks.append(np.zeros(n, dtype=np.float32))
        else:
            # Sine wave at 440 Hz scaled to amp — comfortably above -45 dB threshold
            t = np.arange(n) / SR
            chunks.append((amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    return np.concatenate(chunks)


class TestFindSilences:
    def test_no_silence(self):
        # 5 seconds of continuous tone, no quiet regions
        audio = make_audio([(5.0, 0.3)])
        silences = ct.find_silences(audio, SR, threshold_db=-45, min_duration_ms=200)
        assert silences == []

    def test_one_clear_silence(self):
        # 1 sec tone, 1 sec silence, 1 sec tone
        audio = make_audio([(1.0, 0.3), (1.0, 0.0), (1.0, 0.3)])
        silences = ct.find_silences(audio, SR, threshold_db=-45, min_duration_ms=200)
        assert len(silences) == 1
        start, end = silences[0]
        # Should be in the second second (allow some slack for frame quantization)
        assert SR * 0.9 <= start <= SR * 1.1
        assert SR * 1.9 <= end <= SR * 2.1

    def test_multiple_silences(self):
        audio = make_audio([
            (0.5, 0.3),  # tone
            (0.5, 0.0),  # silence #1
            (0.5, 0.3),  # tone
            (0.5, 0.0),  # silence #2
            (0.5, 0.3),  # tone
        ])
        silences = ct.find_silences(audio, SR, threshold_db=-45, min_duration_ms=200)
        assert len(silences) == 2

    def test_short_silence_below_min_duration_ignored(self):
        # 100 ms of silence, below the 200 ms threshold — should be ignored
        audio = make_audio([(1.0, 0.3), (0.1, 0.0), (1.0, 0.3)])
        silences = ct.find_silences(audio, SR, threshold_db=-45, min_duration_ms=200)
        assert silences == []

    def test_silence_at_end_of_file(self):
        # tone + silence that runs to end of file
        audio = make_audio([(1.0, 0.3), (1.0, 0.0)])
        silences = ct.find_silences(audio, SR, threshold_db=-45, min_duration_ms=200)
        assert len(silences) == 1
        # End of silence should be at the end of the audio (or very close)
        _, end = silences[0]
        assert end >= len(audio) - SR * 0.1  # within ~100 ms of end

    def test_threshold_lower_classifies_more_as_signal(self):
        # Very quiet tone (-50 dB-ish): with threshold=-45 it's silence; with -60 it's signal
        quiet = make_audio([(2.0, 0.003)])  # ~-50 dB
        loud_threshold = ct.find_silences(quiet, SR, threshold_db=-45, min_duration_ms=200)
        strict_threshold = ct.find_silences(quiet, SR, threshold_db=-60, min_duration_ms=200)
        # At -45 dB threshold, this is silence
        assert len(loud_threshold) >= 1
        # At -60 dB threshold (more permissive), this looks like signal
        assert len(strict_threshold) == 0

    def test_returns_tuples_of_ints(self):
        audio = make_audio([(0.5, 0.3), (0.5, 0.0), (0.5, 0.3)])
        silences = ct.find_silences(audio, SR, threshold_db=-45, min_duration_ms=200)
        for s, e in silences:
            assert isinstance(s, int) or isinstance(s, np.integer)
            assert isinstance(e, int) or isinstance(e, np.integer)
            assert s < e
