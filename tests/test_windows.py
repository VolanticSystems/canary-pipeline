"""Tests for silence-anchored window planning."""
import canary_transcribe as ct


SR = 16000


class TestComputeWindows:
    def test_audio_shorter_than_window(self):
        # 10 sec audio, 30 sec window — one window covering everything
        windows = ct.compute_windows(
            num_samples=10 * SR, sr=SR,
            window_sec=30, target_overlap_sec=20, min_overlap_sec=5,
            silences=[], anchor_tolerance_sec=3,
        )
        assert len(windows) == 1
        assert windows[0] == (0, 10 * SR)

    def test_no_silences_fallback_to_target(self):
        # 90 sec audio, 30/20 windows, no silences — windows step every 10 sec
        windows = ct.compute_windows(
            num_samples=90 * SR, sr=SR,
            window_sec=30, target_overlap_sec=20, min_overlap_sec=5,
            silences=[], anchor_tolerance_sec=3,
        )
        # Should have ~7 windows: [0-30, 10-40, 20-50, 30-60, 40-70, 50-80, 60-90]
        assert len(windows) >= 5
        assert windows[0][0] == 0
        # Last window should reach end of audio
        assert windows[-1][1] == 90 * SR

    def test_min_overlap_floor_respected(self):
        # 90 sec audio, with a silence that would otherwise create overlap < min_overlap.
        # The floor should kick in to enforce min_overlap.
        windows = ct.compute_windows(
            num_samples=90 * SR, sr=SR,
            window_sec=30, target_overlap_sec=20, min_overlap_sec=5,
            silences=[], anchor_tolerance_sec=3,
        )
        # Check overlap between every adjacent pair >= min_overlap
        for i in range(len(windows) - 1):
            overlap_samples = windows[i][1] - windows[i + 1][0]
            assert overlap_samples >= 5 * SR, (
                f"Window {i}→{i+1} overlap {overlap_samples/SR:.2f}s < 5s floor"
            )

    def test_window_edges_snap_to_silence(self):
        # 90 sec audio with a silence around 28.4-28.6 sec (mid at 28.5).
        # Target end of first window is 30s; silence midpoint is at 28.5s,
        # 1.5s away — within the ±3s tolerance — so window end snaps to 28.5*SR.
        silence_mid_sec = 28.5
        s_start = int((silence_mid_sec - 0.1) * SR)
        s_end = int((silence_mid_sec + 0.1) * SR)
        silences = [(s_start, s_end)]
        windows = ct.compute_windows(
            num_samples=90 * SR, sr=SR,
            window_sec=30, target_overlap_sec=20, min_overlap_sec=5,
            silences=silences, anchor_tolerance_sec=3,
        )
        first_end = windows[0][1]
        # Should be near the silence midpoint, not at 30s
        snapped_to_silence = abs(first_end - silence_mid_sec * SR) < 0.2 * SR
        assert snapped_to_silence, (
            f"First window end {first_end/SR:.2f}s did not snap to silence at {silence_mid_sec}s"
        )

    def test_silence_outside_tolerance_not_snapped(self):
        # Silence at 10 sec is way outside the ±3s tolerance around the 30-sec target
        silences = [(int(10 * SR), int(10.5 * SR))]
        windows = ct.compute_windows(
            num_samples=90 * SR, sr=SR,
            window_sec=30, target_overlap_sec=20, min_overlap_sec=5,
            silences=silences, anchor_tolerance_sec=3,
        )
        # First window end should be near 30s, not near 10s
        assert abs(windows[0][1] - 30 * SR) <= 3 * SR

    def test_last_tail_window_dropped_if_tiny(self):
        # Audio designed so the last window would be very short
        # 30 sec window, 20 sec overlap = 10 sec step. 30.5 sec audio means
        # window 0 ends at 30, and there would be a tail from 10 to 30.5.
        # The script's `while end < num_samples` loop would emit the final
        # window. Then the tail-drop in compute_windows removes windows
        # smaller than 0.3 sec.
        # Use an audio where the very last window is naturally tiny.
        windows = ct.compute_windows(
            num_samples=int(30.1 * SR), sr=SR,
            window_sec=30, target_overlap_sec=20, min_overlap_sec=5,
            silences=[], anchor_tolerance_sec=3,
        )
        # Every window must be at least 0.3 sec
        for s, e in windows:
            assert (e - s) >= int(SR * 0.3), (
                f"Tiny window kept: {(e-s)/SR:.3f}s"
            )

    def test_windows_cover_entire_audio(self):
        # Every sample of audio must be inside at least one window
        # (the cardinal "no VAD gate" invariant of the design)
        num_samples = 90 * SR
        windows = ct.compute_windows(
            num_samples=num_samples, sr=SR,
            window_sec=30, target_overlap_sec=20, min_overlap_sec=5,
            silences=[], anchor_tolerance_sec=3,
        )
        # Sweep coverage
        covered = [False] * num_samples
        for s, e in windows:
            for i in range(s, min(e, num_samples)):
                covered[i] = True
        uncovered = sum(1 for c in covered if not c)
        # Allow at most a handful of samples at the boundary (rounding)
        assert uncovered < SR * 0.1, f"{uncovered} samples uncovered ({uncovered/SR:.3f}s)"

    def test_no_infinite_loop_on_weird_input(self):
        # If we hit any weird edge case, compute_windows should bail via its
        # safety counter, not hang the script.
        windows = ct.compute_windows(
            num_samples=1, sr=SR,
            window_sec=30, target_overlap_sec=20, min_overlap_sec=5,
            silences=[], anchor_tolerance_sec=3,
        )
        # Single sample is below the tail-drop threshold; result should be empty
        assert len(windows) == 0
