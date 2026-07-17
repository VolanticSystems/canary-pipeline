"""Tests for overlap-zone word merging.

Semantics (2026-07-10 onward): the merge is a **timestamp-ownership partition**,
not a text-match dedup. Each window owns the interval where its center is
closest to any time. A word is kept iff its abs_start lies inside its emitting
window's ownership interval.

This replaces an earlier text-match approach that missed paraphrases and
duplicated timestamps across three overlapping windows.
"""
import canary_transcribe as ct


def w(text, abs_start, abs_end, window_idx, window_center):
    """Build a word dict for testing."""
    return {
        "abs_start": abs_start,
        "abs_end": abs_end,
        "text": text,
        "window_idx": window_idx,
        "window_center": window_center,
    }


class TestMergeOverlappingWords:
    def test_empty_input(self):
        assert ct.merge_overlapping_words([]) == []

    def test_single_window_all_words_kept(self):
        # One window, its ownership range is (-inf, +inf), so every word is kept.
        words = [
            w("alpha", 1.0, 1.3, 0, 15.0),
            w("bravo", 2.0, 2.3, 0, 15.0),
            w("charlie", 3.0, 3.3, 0, 15.0),
        ]
        merged = ct.merge_overlapping_words(words)
        assert len(merged) == 3

    def test_same_word_at_same_time_from_two_windows(self):
        # Both windows saw the same word at t=25. Ownership boundary between
        # window 0 (center 15) and window 1 (center 30) is at 22.5. The word
        # at t=25 belongs to window 1's territory. Only window 1's copy
        # survives. Window 0's copy is dropped.
        words = [
            w("hello", 25.1, 25.4, 0, 15.0),
            w("hello", 25.1, 25.4, 1, 30.0),
        ]
        merged = ct.merge_overlapping_words(words)
        assert len(merged) == 1
        assert merged[0]["window_idx"] == 1

    def test_word_in_window_0_territory(self):
        # Same word at t=17. Boundary is 22.5. 17 < 22.5 → window 0's territory.
        # Only window 0's copy survives.
        words = [
            w("hello", 17.0, 17.3, 0, 15.0),
            w("hello", 17.0, 17.3, 1, 30.0),  # rare but possible: window 1 also emitted
        ]
        merged = ct.merge_overlapping_words(words)
        assert len(merged) == 1
        assert merged[0]["window_idx"] == 0

    def test_paraphrase_dedup(self):
        # The old text-match algorithm missed paraphrases like "did not" vs
        # "didn't" at the same timestamp. The partition algorithm doesn't
        # compare text — it partitions by ownership. Both copies would still
        # collapse to one if they fall in the same window's territory.
        words = [
            w("didn't", 25.1, 25.4, 0, 15.0),
            w("did not", 25.1, 25.4, 1, 30.0),
        ]
        merged = ct.merge_overlapping_words(words)
        # Ownership boundary is at 22.5; t=25.1 belongs to window 1.
        # Window 0's "didn't" is dropped, window 1's "did not" survives.
        assert len(merged) == 1
        assert merged[0]["text"] == "did not"

    def test_three_window_overlap(self):
        # With 67% overlap on 30s windows, three windows share every interior
        # region. window centers at 15, 25, 35, 45 ...
        # Word at t=27: ownership goes to window 1 (nearest center).
        words = [
            w("hello", 27.0, 27.3, 0, 15.0),
            w("hello", 27.0, 27.3, 1, 25.0),
            w("hello", 27.0, 27.3, 2, 35.0),
        ]
        merged = ct.merge_overlapping_words(words)
        # window 1 owns [20, 30]. t=27 is in window 1's range → wins.
        assert len(merged) == 1
        assert merged[0]["window_idx"] == 1

    def test_words_at_different_times_from_own_territories_all_kept(self):
        # Two windows, two words each in their own territory. All four survive.
        # Window 0 owns < 22.5, window 1 owns >= 22.5.
        words = [
            w("hello", 17.0, 17.3, 0, 15.0),   # in window 0's territory
            w("hello", 20.0, 20.3, 0, 15.0),   # still in window 0's territory
            w("world", 25.0, 25.3, 1, 30.0),   # in window 1's territory
            w("world", 28.0, 28.3, 1, 30.0),   # still in window 1's territory
        ]
        merged = ct.merge_overlapping_words(words)
        assert len(merged) == 4

    def test_output_sorted_by_start(self):
        # Even if input isn't sorted, output must be sorted by abs_start.
        words = [
            w("c", 3.0, 3.3, 0, 15.0),
            w("a", 1.0, 1.3, 0, 15.0),
            w("b", 2.0, 2.3, 0, 15.0),
        ]
        merged = ct.merge_overlapping_words(words)
        assert [m["text"] for m in merged] == ["a", "b", "c"]

    def test_no_sliding_window_doubling(self):
        # Regression test for the Elastrin bug: word-level doubling like
        # "didn't didn't like like it it" from overlapping windows emitting
        # the same word twice.
        # Setup: three overlapping windows all emit "didn't like it" at t=27.
        # Only ONE copy of each word should survive.
        words = []
        for wi, wc in [(0, 15.0), (1, 25.0), (2, 35.0)]:
            words.append(w("didn't", 27.0, 27.4, wi, wc))
            words.append(w("like",   27.5, 27.9, wi, wc))
            words.append(w("it",     28.0, 28.4, wi, wc))
        merged = ct.merge_overlapping_words(words)
        texts = [m["text"] for m in merged]
        # Exactly three words, one of each, in order.
        assert texts == ["didn't", "like", "it"]
        # All from the owner window (window 1 owns [20, 30]).
        assert all(m["window_idx"] == 1 for m in merged)

    def test_dedup_across_paraphrase_and_case(self):
        # Old test: punctuation & case shouldn't prevent dedup. Under partition
        # semantics, dedup is timestamp-based, not text-based — case and
        # punctuation are irrelevant. This test just confirms the partition
        # still collapses same-timestamp duplicates regardless of text.
        words = [
            w("Hello,", 25.0, 25.3, 0, 15.0),
            w("hello",  25.0, 25.3, 1, 30.0),
        ]
        merged = ct.merge_overlapping_words(words)
        assert len(merged) == 1


class TestMergeSortInvariant:
    def test_unsorted_input_produces_sorted_output(self):
        # Regression for the tuned-transcript out-of-order bug: 266 inversions.
        # Input words from checkpoint are not sorted by time. Output must be.
        words = [
            w("second",  67.7, 68.0, 3, 65.0),
            w("first",   49.1, 49.4, 2, 55.0),
            w("third",   90.0, 90.3, 4, 85.0),
        ]
        merged = ct.merge_overlapping_words(words)
        starts = [m["abs_start"] for m in merged]
        assert starts == sorted(starts)
