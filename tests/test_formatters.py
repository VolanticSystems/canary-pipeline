"""Tests for the small formatter functions."""
import canary_transcribe as ct


class TestFmtHms:
    def test_zero(self):
        assert ct.fmt_hms(0) == "00:00:00"

    def test_one_second(self):
        assert ct.fmt_hms(1) == "00:00:01"

    def test_one_minute(self):
        assert ct.fmt_hms(60) == "00:01:00"

    def test_one_hour(self):
        assert ct.fmt_hms(3600) == "01:00:00"

    def test_mixed(self):
        assert ct.fmt_hms(3661) == "01:01:01"

    def test_fractional_truncated(self):
        # fmt_hms uses int floor; 59.9 sec is still "00:00:59"
        assert ct.fmt_hms(59.9) == "00:00:59"


class TestFmtSrt:
    def test_zero(self):
        assert ct.fmt_srt(0) == "00:00:00,000"

    def test_milliseconds(self):
        assert ct.fmt_srt(1.5) == "00:00:01,500"

    def test_one_hour_with_ms(self):
        assert ct.fmt_srt(3600.250) == "01:00:00,250"

    def test_comma_separator(self):
        # SRT format requires comma, not period, as decimal sep
        assert "," in ct.fmt_srt(1.5)
        assert "." not in ct.fmt_srt(1.5)


class TestSpeakerLabel:
    def test_none(self):
        assert ct.speaker_label(None) == "Speaker ?"

    def test_unknown(self):
        assert ct.speaker_label("SPEAKER_UNKNOWN") == "Speaker ?"

    def test_speaker_00_is_one(self):
        # SPEAKER_00 is the first pyannote cluster, displayed as "Speaker 1"
        assert ct.speaker_label("SPEAKER_00") == "Speaker 1"

    def test_speaker_03_is_four(self):
        assert ct.speaker_label("SPEAKER_03") == "Speaker 4"

    def test_empty_string(self):
        assert ct.speaker_label("") == "Speaker ?"

    def test_malformed_falls_through(self):
        # If pyannote ever returns something weird, don't crash — return raw
        assert ct.speaker_label("WEIRD_LABEL") == "WEIRD_LABEL"


class TestNormalizeText:
    def test_lowercase(self):
        assert ct.normalize_text("Hello") == "hello"

    def test_strips_punctuation(self):
        assert ct.normalize_text("Hello, world!") == "helloworld"

    def test_alphanumeric_kept(self):
        assert ct.normalize_text("test123") == "test123"

    def test_empty(self):
        assert ct.normalize_text("") == ""

    def test_only_punctuation(self):
        assert ct.normalize_text("!?,.") == ""
