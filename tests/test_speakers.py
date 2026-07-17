"""Tests for speaker assignment and segment building."""
import canary_transcribe as ct


def w(abs_start, abs_end, text="word"):
    return {
        "abs_start": abs_start,
        "abs_end": abs_end,
        "text": text,
        "window_idx": 0,
        "window_center": 0,
    }


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


class TestAssignSpeakersSmoothed:
    def test_empty_words(self):
        assert ct.assign_speakers_smoothed([], [], smoothing_sec=2.0) == []

    def test_word_in_middle_of_turn(self):
        words = [w(10.0, 10.3)]
        turns = [turn(0, 60, "SPEAKER_00")]
        result = ct.assign_speakers_smoothed(words, turns, smoothing_sec=2.0)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_word_with_no_overlapping_turn(self):
        words = [w(100.0, 100.3)]
        turns = [turn(0, 60, "SPEAKER_00")]
        result = ct.assign_speakers_smoothed(words, turns, smoothing_sec=2.0)
        assert result[0]["speaker"] == "SPEAKER_UNKNOWN"

    def test_no_turns_at_all(self):
        # Pyannote returned nothing — every word becomes UNKNOWN
        words = [w(10, 10.3), w(20, 20.3)]
        result = ct.assign_speakers_smoothed(words, [], smoothing_sec=2.0)
        assert all(x["speaker"] == "SPEAKER_UNKNOWN" for x in result)

    def test_dominant_speaker_in_window_wins(self):
        # Word at t=30. Smoothing window ±1 sec = [29, 31].
        # SPEAKER_00 covers 29-30.6 = 1.6s within window
        # SPEAKER_01 covers 30.6-31 = 0.4s within window
        # SPEAKER_00 should win.
        words = [w(30.0, 30.3)]
        turns = [
            turn(0, 30.6, "SPEAKER_00"),
            turn(30.6, 60, "SPEAKER_01"),
        ]
        result = ct.assign_speakers_smoothed(words, turns, smoothing_sec=2.0)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_speakers_dont_carry_across_words(self):
        # Two words at very different times in different turns — each gets its own
        words = [w(10, 10.3), w(50, 50.3)]
        turns = [
            turn(0, 30, "SPEAKER_00"),
            turn(40, 60, "SPEAKER_01"),
        ]
        result = ct.assign_speakers_smoothed(words, turns, smoothing_sec=2.0)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"


class TestBuildSegments:
    def test_empty(self):
        assert ct.build_segments([]) == []

    def test_single_word(self):
        words = [{"abs_start": 1.0, "abs_end": 1.3,
                  "speaker": "SPEAKER_00", "text": "hello"}]
        segs = ct.build_segments(words)
        assert len(segs) == 1
        assert segs[0]["start"] == 1.0
        assert segs[0]["end"] == 1.3
        assert segs[0]["text"] == "hello"

    def test_consecutive_same_speaker_merged(self):
        words = [
            {"abs_start": 1.0, "abs_end": 1.3, "speaker": "SPEAKER_00", "text": "hello"},
            {"abs_start": 1.4, "abs_end": 1.6, "speaker": "SPEAKER_00", "text": "world"},
        ]
        segs = ct.build_segments(words)
        assert len(segs) == 1
        assert segs[0]["text"] == "hello world"
        assert segs[0]["start"] == 1.0
        assert segs[0]["end"] == 1.6

    def test_speaker_change_creates_new_segment(self):
        words = [
            {"abs_start": 1.0, "abs_end": 1.3, "speaker": "SPEAKER_00", "text": "hello"},
            {"abs_start": 1.4, "abs_end": 1.6, "speaker": "SPEAKER_01", "text": "world"},
        ]
        segs = ct.build_segments(words)
        assert len(segs) == 2
        assert segs[0]["speaker"] == "SPEAKER_00"
        assert segs[1]["speaker"] == "SPEAKER_01"

    def test_three_speakers_alternating(self):
        words = [
            {"abs_start": 1.0, "abs_end": 1.3, "speaker": "A", "text": "a1"},
            {"abs_start": 2.0, "abs_end": 2.3, "speaker": "B", "text": "b1"},
            {"abs_start": 3.0, "abs_end": 3.3, "speaker": "A", "text": "a2"},
            {"abs_start": 4.0, "abs_end": 4.3, "speaker": "C", "text": "c1"},
        ]
        segs = ct.build_segments(words)
        assert len(segs) == 4
        assert [s["speaker"] for s in segs] == ["A", "B", "A", "C"]
