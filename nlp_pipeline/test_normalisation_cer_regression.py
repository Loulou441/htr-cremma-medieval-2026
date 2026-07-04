"""Test de non-regression CER pour la normalisation par regles.

Le petit echantillon ci-dessous est un sous-ensemble minimal, mais
representatif, du tableau d'ablation manuel (`data/reference_200.csv`,
cf. CONVENTIONS_NLP.md section 5) : chaque paire (texte_brut, reference)
a ete verifiee manuellement. Il est duplique ici en dur pour que le test
soit autonome et ne depende pas d'un fichier de donnees externe (qui n'est
pas versionne dans le depot).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cer_utils import cer
from normalization_rules import MedievalFrenchNormalizer


# Table d'abreviations reduite, alignee sur medieval_abbreviations.json,
# suffisante pour les cas couverts par cet echantillon.
ABBREVIATIONS = {"q~": "que", "d~e": "dame"}

# (texte_brut_htr, transcription_de_reference_annotee_manuellement)
REFERENCE_SAMPLE = [
    ("q~ d~e", "que dame"),                  # table d'abreviations
    ("a~ e~ o~", "an en om"),                # tilde nasal
    ("cheualier auant", "chevalier avant"),  # regle u/v contextuelle
    ("que", "que"),                          # digramme qu- preserve (pas de bruit introduit)
    ("guerre", "guerre"),                    # digramme gu- preserve
    ("lui", "lui"),                          # u+i preserve
    ("bien", "bien"),                        # diphtongue ie preservee
    ("mie", "mie"),                          # diphtongue ie preservee
]


def _normalizer() -> MedievalFrenchNormalizer:
    return MedievalFrenchNormalizer(ABBREVIATIONS)


def test_normalization_does_not_degrade_mean_cer_on_reference_sample():
    """Le CER moyen (texte vs reference) ne doit pas augmenter apres normalisation."""
    normalizer = _normalizer()

    cer_before = [cer(reference, raw) for raw, reference in REFERENCE_SAMPLE]
    cer_after = [
        cer(reference, normalizer.normalize(raw)) for raw, reference in REFERENCE_SAMPLE
    ]

    mean_before = sum(cer_before) / len(cer_before)
    mean_after = sum(cer_after) / len(cer_after)

    assert mean_after <= mean_before, (
        f"La normalisation degrade le CER moyen sur l'echantillon de reference "
        f"({mean_before:.4f} -> {mean_after:.4f})"
    )


def test_normalization_does_not_degrade_cer_per_line():
    """Aucune ligne individuelle ne doit voir son CER augmenter apres normalisation.

    Une regression ligne par ligne serait masquee par une simple moyenne
    (une grosse amelioration pourrait compenser une degradation ponctuelle) ;
    ce test verifie donc chaque paire independamment.
    """
    normalizer = _normalizer()

    for raw, reference in REFERENCE_SAMPLE:
        before = cer(reference, raw)
        after = cer(reference, normalizer.normalize(raw))
        assert after <= before, (
            f"Ligne {raw!r} : le CER se degrade apres normalisation "
            f"({before:.4f} -> {after:.4f}), reference={reference!r}"
        )


def test_reference_sample_is_not_trivially_empty():
    """Garde-fou : l'echantillon de reference doit rester non vide et non trivial."""
    assert len(REFERENCE_SAMPLE) >= 5
    assert any(raw != reference for raw, reference in REFERENCE_SAMPLE)