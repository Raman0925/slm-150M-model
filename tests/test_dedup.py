"""Tests for dedup.py — pure helpers for Phase 2 (dedup + decontamination)."""

from __future__ import annotations

import dedup


def test_normalize_collapses_whitespace_and_lowercases():
    assert dedup.normalize("  Hello   WORLD\n\tfoo ") == "hello world foo"


def test_words_extracts_alnum_tokens():
    assert dedup.words("Hello, World! 123 test-case") == ["hello", "world", "123", "test", "case"]


def test_exact_hash_stable_for_normalization_equivalent_text():
    a = dedup.exact_hash("Hello   World")
    b = dedup.exact_hash("hello world")
    assert a == b


def test_exact_hash_differs_for_different_text():
    assert dedup.exact_hash("hello world") != dedup.exact_hash("goodbye world")


def test_word_ngrams_correct_count_and_uniqueness():
    tokens = ["a", "b", "c", "d", "e"]
    grams = dedup.word_ngrams(tokens, 3)
    # (a,b,c) (b,c,d) (c,d,e) -> 3 distinct 3-grams
    assert len(grams) == 3


def test_word_ngrams_empty_when_too_short():
    assert dedup.word_ngrams(["a", "b"], 3) == set()


def test_word_ngrams_shared_across_overlapping_texts():
    grams_a = dedup.word_ngrams(dedup.words("the quick brown fox jumps"), 4)
    grams_b = dedup.word_ngrams(dedup.words("the quick brown fox jumps over the dog"), 4)
    assert grams_a & grams_b  # shared 4-gram means overlap detected


def test_shingles_k_sized_windows():
    tokens = ["a", "b", "c", "d"]
    sh = dedup.shingles(tokens, 2)
    assert sh == {b"a b", b"b c", b"c d"}


def test_shingles_fallback_when_shorter_than_k():
    assert dedup.shingles(["a", "b"], 5) == {b"a b"}


def test_shingles_empty_for_empty_tokens():
    assert dedup.shingles([], 5) == set()
