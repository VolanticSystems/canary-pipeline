"""Tests for the pyannote-coverage audit (Option D).

This function was rewritten after the original "clever" version was suspect.
These tests lock in the correctness of the rewrite.
"""
import canary_transcribe as ct


def turn(start, end, speaker="S"):
    return {"start": start, "end": end, "speaker": speaker}


def seg(start, end):
    return {"start": start, "end": end}


class TestAuditPyannoteCoverage:
    def test_no_turns_returns_100_pct(self):
        result = ct.audit_pyannote_coverage(turns=[], segments=[seg(0, 10)],
                                            edge_grace_sec=0.5, min_gap_sec=2.0)
        assert result["coverage_pct"] == 100.0
        assert result["speech_seconds"] == 0.0
        assert result["gaps"] == []

    def test_full_coverage(self):
        # One turn 0-10s, one segment 0-10s — perfect coverage
        result = ct.audit_pyannote_coverage(
            turns=[turn(0, 10)], segments=[seg(0, 10)],
            edge_grace_sec=0.5, min_gap_sec=2.0,
        )
        assert result["coverage_pct"] == 100.0
        assert result["speech_seconds"] == 10.0
        assert result["covered_seconds"] == 10.0
        assert result["gaps"] == []

    def test_zero_coverage(self):
        # Turn 0-10s, segment way outside at 100-110s — no overlap
        result = ct.audit_pyannote_coverage(
            turns=[turn(0, 10)], segments=[seg(100, 110)],
            edge_grace_sec=0.5, min_gap_sec=2.0,
        )
        assert result["coverage_pct"] == 0.0
        assert result["uncovered_seconds"] == 10.0
        # One gap covering the entire turn
        assert len(result["gaps"]) == 1
        assert result["gaps"][0]["duration"] == 10.0

    def test_partial_coverage(self):
        # Turn 0-10s. Segment 0-5s covers first half.
        result = ct.audit_pyannote_coverage(
            turns=[turn(0, 10)], segments=[seg(0, 5)],
            edge_grace_sec=0.5, min_gap_sec=2.0,
        )
        assert result["coverage_pct"] == 50.0
        assert result["covered_seconds"] == 5.0
        # Gap from 5 to 10
        assert len(result["gaps"]) == 1
        assert result["gaps"][0]["duration"] == 5.0
        assert result["gaps"][0]["start"] == 5.0
        assert result["gaps"][0]["end"] == 10.0

    def test_short_gap_below_min_gap_not_reported(self):
        # Turn 0-10s, segment 0-9.5s — gap is 0.5s, below min_gap=2.0
        # But edge_grace=0.5 means anything <=0.5s gap is grace
        result = ct.audit_pyannote_coverage(
            turns=[turn(0, 10)], segments=[seg(0, 9.5)],
            edge_grace_sec=0.5, min_gap_sec=2.0,
        )
        # No gap reported (within edge grace)
        assert len(result["gaps"]) == 0

    def test_gap_above_edge_grace_but_below_min_gap_not_reported(self):
        # Gap of 1s — above edge grace (0.5) but below min_gap (2.0)
        # Coverage % reflects the missing 1s; no gap entry in the list
        result = ct.audit_pyannote_coverage(
            turns=[turn(0, 10)], segments=[seg(0, 9)],
            edge_grace_sec=0.5, min_gap_sec=2.0,
        )
        assert result["covered_seconds"] == 9.0
        assert len(result["gaps"]) == 0

    def test_multiple_turns_summed(self):
        # Two turns, both perfectly covered
        result = ct.audit_pyannote_coverage(
            turns=[turn(0, 10), turn(20, 30)],
            segments=[seg(0, 10), seg(20, 30)],
            edge_grace_sec=0.5, min_gap_sec=2.0,
        )
        assert result["speech_seconds"] == 20.0
        assert result["covered_seconds"] == 20.0
        assert result["coverage_pct"] == 100.0

    def test_segment_spans_multiple_turns(self):
        # Two turns (0-10, 20-30) but one big segment 0-30 covering everything
        # including the silent gap in between
        result = ct.audit_pyannote_coverage(
            turns=[turn(0, 10), turn(20, 30)],
            segments=[seg(0, 30)],
            edge_grace_sec=0.5, min_gap_sec=2.0,
        )
        assert result["coverage_pct"] == 100.0
        assert result["covered_seconds"] == 20.0  # only counts what's in turns

    def test_overlapping_segments_dont_double_count(self):
        # Two overlapping segments inside one turn shouldn't count their overlap twice
        result = ct.audit_pyannote_coverage(
            turns=[turn(0, 10)],
            segments=[seg(0, 6), seg(4, 10)],
            edge_grace_sec=0.5, min_gap_sec=2.0,
        )
        assert result["covered_seconds"] == 10.0
        assert result["coverage_pct"] == 100.0
