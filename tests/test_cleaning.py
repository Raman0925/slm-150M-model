"""Tests for cleaning.py — the deterministic 6-step cleaning chain."""

from __future__ import annotations

import cleaning
from config import CLEAN


def test_filter_lines_drops_short_lines():
    short = "too short"
    long_enough = "x" * CLEAN.min_line_chars
    out = cleaning.filter_lines(f"{short}\n{long_enough}")
    assert short not in out
    assert long_enough in out


def test_filter_lines_drops_high_nonalnum_ratio():
    noisy = "!" * 50 + "ab"  # ~96% non-alnum, well over the 30% cap
    wordy = "a" * CLEAN.min_line_chars  # 100% alnum
    out = cleaning.filter_lines(f"{noisy}\n{wordy}")
    assert noisy not in out
    assert wordy in out


def test_filter_lines_collapses_whitespace_and_strips():
    # Collapsed form must still clear min_line_chars, or filter_lines drops it as too short.
    words = " ".join(f"word{i}" for i in range(10))  # collapsed form, for comparison
    line = "   " + "   ".join(f"word{i}" for i in range(10)) + "  \t "
    out = cleaning.filter_lines(line)
    assert out == words


def test_strip_boilerplate_removes_known_patterns():
    text = "\n".join([
        "Form 10-K annual report",
        "Table of Contents",
        "Page 3 of 45",
        "/s/ Jane Doe, Secretary",
        "All Rights Reserved.",
        "This is real substantive content that should survive.",
    ])
    out = cleaning.strip_boilerplate(text)
    assert "real substantive content" in out
    assert "Table of Contents" not in out
    assert "Page 3 of 45" not in out
    assert "Form 10-K" not in out


def test_is_repetitive_true_for_repeated_ngram():
    words = (["the quick brown fox"] * 20)
    text = " ".join(words)
    assert cleaning.is_repetitive(text) is True


def test_is_repetitive_false_for_varied_text():
    text = " ".join(f"word{i} filler{i} extra{i} more{i}" for i in range(50))
    assert cleaning.is_repetitive(text) is False


def test_is_repetitive_false_when_too_short():
    assert cleaning.is_repetitive("only three words") is False


def test_is_english_true_for_pure_ascii():
    assert cleaning.is_english("This is plain English text." * 5) is True


def test_is_english_false_for_low_ascii_ratio():
    non_ascii = "文字化け" * 50  # far below the 90% ascii threshold
    assert cleaning.is_english(non_ascii) is False


def test_nonword_ratio_uses_injected_dictionary(monkeypatch):
    monkeypatch.setattr(cleaning, "_ENGLISH_WORDS", frozenset({"the", "quick", "brown", "fox", "jumps"}))
    # 30 known tokens + 30 unknown tokens = 60 total (>= ocr_min_tokens=50) -> ratio 0.5
    text = " ".join(["the", "quick", "brown", "fox", "jumps"] * 6 + ["zzqx"] * 30)
    ratio = cleaning.nonword_ratio(text)
    assert 0.45 < ratio < 0.55


def test_nonword_ratio_zero_below_min_tokens(monkeypatch):
    monkeypatch.setattr(cleaning, "_ENGLISH_WORDS", frozenset({"the"}))
    assert cleaning.nonword_ratio("zzq wwq xxq") == 0.0


def test_is_ocr_garble_respects_threshold(monkeypatch):
    monkeypatch.setattr(cleaning, "_ENGLISH_WORDS", frozenset({"the"}))
    garbled = " ".join(["zzq"] * 60)
    assert cleaning.is_ocr_garble(garbled) is True
    clean_text = " ".join(["the"] * 60)
    assert cleaning.is_ocr_garble(clean_text) is False


def test_clean_document_too_short():
    result = cleaning.clean_document("short text")
    assert result.kept is False
    assert result.reason == "too_short"


def test_clean_document_kept_for_normal_prose():
    # Every token carries a unique per-iteration suffix so no 4-gram repeats,
    # keeping this well clear of the repetition filter (which is tested separately).
    body = " ".join(
        f"paragraph{i} discusses{i} topic{i} number{i} with{i} unique{i} content{i} here{i}"
        for i in range(80)
    )
    result = cleaning.clean_document(body)
    assert result.kept is True
    assert result.reason == "kept"
    assert result.clean_chars > 0


def test_clean_document_repetitive():
    body = ("the plaintiff hereby moves the court " * 60)
    result = cleaning.clean_document(body)
    assert result.kept is False
    assert result.reason == "repetitive"
