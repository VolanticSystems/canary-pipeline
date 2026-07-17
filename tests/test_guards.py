"""Tests for the tail-gap guard, vocab corrections, semantic-inversion
detection, and sorted-writer defensive checks.

All of these were added 2026-07-10 in response to the Elastrin First Meeting
review that surfaced silent data loss and unsorted output.
"""
import json
import tempfile
from pathlib import Path

import canary_transcribe as ct


def seg(start, end, speaker, text):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


class TestCheckTailGap:
    def test_no_segments_fails(self):
        ok, gap = ct.check_tail_gap([], 6917.0, max_trailing_gap=30.0)
        assert not ok
        assert gap == 6917.0

    def test_last_segment_ends_at_audio_end(self):
        segs = [seg(0, 6917.0, "S0", "text")]
        ok, gap = ct.check_tail_gap(segs, 6917.0, max_trailing_gap=30.0)
        assert ok
        assert gap <= 0.001

    def test_small_gap_ok(self):
        # 25 second gap, under the 30 second threshold
        segs = [seg(0, 6892.0, "S0", "text")]
        ok, gap = ct.check_tail_gap(segs, 6917.0, max_trailing_gap=30.0)
        assert ok
        assert abs(gap - 25.0) < 0.001

    def test_big_gap_fails(self):
        # Regression for Elastrin: 5:42 (342 sec) trailing gap. Must fail.
        segs = [seg(0, 6575.0, "S0", "text")]
        ok, gap = ct.check_tail_gap(segs, 6917.0, max_trailing_gap=30.0)
        assert not ok
        assert 340 < gap < 345

    def test_uses_last_end_not_last_start(self):
        # Guard should look at max END, not last START — segments may be
        # unsorted (defensive).
        segs = [
            seg(100, 6900.0, "S0", "spans long"),
            seg(0, 50.0, "S0", "early short one appears last in list"),
        ]
        ok, gap = ct.check_tail_gap(segs, 6917.0, max_trailing_gap=30.0)
        assert ok


class TestApplyVocabCorrections:
    def test_empty_corrections_is_noop(self):
        segs = [seg(0, 1, "S0", "Hello world")]
        result = ct.apply_vocab_corrections(segs, {})
        assert result[0]["text"] == "Hello world"

    def test_simple_substitution(self):
        segs = [seg(0, 1, "S0", "We spoke with Elastin about the trial")]
        result = ct.apply_vocab_corrections(segs, {"Elastin": "Elastrin"})
        assert "Elastrin" in result[0]["text"]
        assert "Elastin" not in result[0]["text"]

    def test_case_insensitive(self):
        segs = [seg(0, 1, "S0", "elastin, ELASTIN, and Elastin")]
        result = ct.apply_vocab_corrections(segs, {"Elastin": "Elastrin"})
        # All three variants should be replaced
        assert "Elastrin" in result[0]["text"]
        assert result[0]["text"].count("Elastrin") == 3

    def test_word_boundary_anchored(self):
        # Must NOT match inside other words — "in" should not match inside
        # "insight" or "sing".
        segs = [seg(0, 1, "S0", "This is insightful, thanks for singing.")]
        result = ct.apply_vocab_corrections(segs, {"in": "OUT"})
        assert "insightful" in result[0]["text"]
        assert "singing" in result[0]["text"]

    def test_multi_word_key(self):
        segs = [seg(0, 1, "S0", "The E N N T B enzyme is critical.")]
        result = ct.apply_vocab_corrections(segs, {"E N N T B": "ENPP1"})
        assert "ENPP1" in result[0]["text"]

    def test_multiple_segments_all_processed(self):
        segs = [
            seg(0, 1, "S0", "Talking about Elastin therapy"),
            seg(1, 2, "S1", "and Spearing's group in Germany"),
        ]
        result = ct.apply_vocab_corrections(segs, {
            "Elastin": "Elastrin",
            "Spearing": "Spiering",
        })
        assert "Elastrin" in result[0]["text"]
        assert "Spiering" in result[1]["text"]


class TestDetectSemanticInversions:
    def test_no_canary_segments_returns_empty(self):
        parakeet = [seg(0, 1, "S0", "hyperphosphatemia patients")]
        result = ct.detect_semantic_inversions(parakeet, [])
        assert result == []

    def test_agreement_no_flag(self):
        # Both models say "hyper" at the same time → no flag
        parakeet = [seg(10.0, 11.0, "S0", "hyperphosphatemia patients")]
        canary = [seg(10.0, 11.0, "S0", "hyperphosphatemia patients")]
        result = ct.detect_semantic_inversions(parakeet, canary)
        assert result == []

    def test_hyper_hypo_disagreement_flagged(self):
        # Parakeet says "hyper", Canary says "hypo" at the same time → flag
        # (This is the actual Elastrin failure mode.)
        parakeet = [seg(10.0, 11.0, "S0", "hyperphosphatemia patients here")]
        canary = [seg(10.0, 11.0, "S0", "hypophosphatemia patients here")]
        result = ct.detect_semantic_inversions(parakeet, canary)
        assert len(result) >= 1
        assert result[0]["time"] == 10.0
        assert "hyper" in result[0]["parakeet_word"].lower()
        assert "hypo" in result[0]["canary_word"].lower()

    def test_time_window_matters(self):
        # If canary says "hypo" 10 seconds away, that's outside the ±3s window
        # → no flag.
        parakeet = [seg(10.0, 11.0, "S0", "hyperphosphatemia patients")]
        canary = [seg(30.0, 31.0, "S0", "hypophosphatemia different topic")]
        result = ct.detect_semantic_inversions(parakeet, canary)
        assert result == []


class TestSortedWriters:
    def test_transcript_txt_output_sorted(self):
        # Regression for the tuned-transcript 266-inversions bug.
        # Even with unsorted input, output must be time-ordered.
        segs = [
            seg(67.7, 68.0, "S0", "second"),
            seg(49.1, 49.4, "S0", "first"),
            seg(90.0, 90.3, "S0", "third"),
        ]
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as fp:
            path = Path(fp.name)
        try:
            ct.write_transcript_txt(path, segs)
            content = path.read_text(encoding="utf-8")
        finally:
            path.unlink()
        # first, second, third should appear in that order
        assert content.index("first") < content.index("second") < content.index("third")

    def test_srt_output_sorted(self):
        segs = [
            seg(67.7, 68.0, "S0", "second"),
            seg(49.1, 49.4, "S0", "first"),
        ]
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".srt") as fp:
            path = Path(fp.name)
        try:
            ct.write_srt(path, segs)
            content = path.read_text(encoding="utf-8")
        finally:
            path.unlink()
        assert content.index("first") < content.index("second")

    def test_jsonl_output_sorted(self):
        segs = [
            seg(67.7, 68.0, "S0", "second"),
            seg(49.1, 49.4, "S0", "first"),
        ]
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as fp:
            path = Path(fp.name)
        try:
            ct.write_jsonl(path, segs)
            lines = path.read_text(encoding="utf-8").splitlines()
        finally:
            path.unlink()
        recs = [json.loads(l) for l in lines]
        assert recs[0]["text"] == "first"
        assert recs[1]["text"] == "second"

    def test_speakers_json_output(self):
        # write_speakers_json needs turns too (uses diarization for time).
        segs = [
            seg(0, 10, "SPEAKER_00", "hello world"),
            seg(10, 20, "SPEAKER_01", "goodbye now"),
        ]
        turns = [
            {"start": 0, "end": 10, "speaker": "SPEAKER_00"},
            {"start": 10, "end": 20, "speaker": "SPEAKER_01"},
        ]
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as fp:
            path = Path(fp.name)
        try:
            ct.write_speakers_json(path, segs, turns)
            data = json.loads(path.read_text(encoding="utf-8"))
        finally:
            path.unlink()
        assert len(data) == 2
        by_raw = {d["speaker_raw"]: d for d in data}
        assert by_raw["SPEAKER_00"]["diarized_seconds"] == 10.0
        assert by_raw["SPEAKER_00"]["transcript_word_count"] == 2
        assert "hello" in by_raw["SPEAKER_00"]["first_sample_text"]
