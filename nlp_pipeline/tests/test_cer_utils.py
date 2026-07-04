"""Tests pour cer_utils.py (CER et WER).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cer_utils import cer, wer


def test_wer_is_zero_for_identical_text():
    assert wer("que dame prist", "que dame prist") == 0.0


def test_wer_counts_word_level_errors_not_char_level():
    # Un seul mot differe ("dame" -> "dama"), quel que soit le nombre de
    # caracteres qui changent a l'interieur de ce mot : WER = 1/2.
    assert wer("que dame", "que dama") == 0.5


def test_wer_denominator_uses_reference_word_count():
    # Un mot en moins dans l'hypothese : distance = 1, ref = 3 mots -> 1/3.
    result = wer("que dame prist", "que dame")
    assert abs(result - (1 / 3)) < 1e-9


def test_wer_matches_cer_semantics_denominator_convention():
    # cer() et wer() suivent la meme convention : edit_distance / max(1, len(reference)).
    # Sur un texte reference vide, les deux valent 0 (aucune erreur mesurable).
    assert cer("", "") == 0.0
    assert wer("", "") == 0.0