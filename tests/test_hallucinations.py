"""Tests for the hallucination spot-check pattern matcher."""
import canary_transcribe as ct


def seg(text, start=0.0, end=1.0, speaker="SPEAKER_00"):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


# Same patterns as config.yaml
PATTERNS = [
    r"(?i)thank you for watching",
    r"(?i)subtitles by",
    r"(?i)amara\.org",
    r"(?i)\[music\]",
]


class TestDetectHallucinations:
    def test_no_match(self):
        segments = [seg("Real testimony content here.")]
        assert ct.detect_hallucinations(segments, PATTERNS) == []

    def test_match_thank_you_for_watching(self):
        segments = [seg("Thank you for watching this video.")]
        result = ct.detect_hallucinations(segments, PATTERNS)
        assert len(result) == 1
        assert "thank you for watching" in result[0]["matched_text"].lower()

    def test_case_insensitive(self):
        segments = [seg("THANK YOU FOR WATCHING")]
        result = ct.detect_hallucinations(segments, PATTERNS)
        assert len(result) == 1

    def test_first_match_wins_per_segment(self):
        # Segment matching MULTIPLE patterns — only one entry per segment
        segments = [seg("Thank you for watching, subtitles by ACME.")]
        result = ct.detect_hallucinations(segments, PATTERNS)
        assert len(result) == 1

    def test_multiple_segments_each_flagged_independently(self):
        segments = [
            seg("Thank you for watching", start=0),
            seg("Real content", start=10),
            seg("subtitles by ACME", start=20),
        ]
        result = ct.detect_hallucinations(segments, PATTERNS)
        assert len(result) == 2

    def test_amara_dot_org_escaped_dot(self):
        # The pattern uses \. which must match literal "." not any char
        segments = [seg("uploaded to amara.org for translation")]
        result = ct.detect_hallucinations(segments, PATTERNS)
        assert len(result) == 1

    def test_music_bracket_pattern(self):
        segments = [seg("[Music] plays in background")]
        result = ct.detect_hallucinations(segments, PATTERNS)
        assert len(result) == 1

    def test_empty_segments(self):
        assert ct.detect_hallucinations([], PATTERNS) == []

    def test_match_includes_speaker_and_timestamp(self):
        segments = [seg("Thank you for watching", start=12.5, end=14.0)]
        result = ct.detect_hallucinations(segments, PATTERNS)
        assert result[0]["start"] == 12.5
        assert result[0]["end"] == 14.0
        assert result[0]["speaker"] == "Speaker 1"  # SPEAKER_00 → "Speaker 1"
