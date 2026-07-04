# HTR + NLP — Manuscrits médiévaux du XIIIe siècle

Projet en deux volets sur la reconnaissance et le traitement de manuscrits médiévaux
(ancien français + latin, XIIIe siècle), réalisé dans le cadre du Mastère Data & IA
d'HETIC :

- **Volet 1 — HTR** : fine-tuning d'un modèle de reconnaissance d'écriture manuscrite
  (Kraken) sur 33 manuscrits CREMMA/HTRomance.
- **Volet 2 — NLP** : normalisation, correction et structuration du texte produit par
  ce modèle sur 16 manuscrits supplémentaires transcrits depuis Gallica/BnF.

> **Ce README est un récapitulatif du projet dans son ensemble.** Le code de chaque
> volet vit sur sa propre branche (voir [section 2](#2-où-trouver-le-code)) — `main`
> ne contient pour l'instant que la documentation. Pour le détail complet de chaque
> volet (commandes, code, résultats intégraux), suivez les liens vers les README de
> branche : ils restent la source de vérité la plus à jour et la plus détaillée.

**Équipe** : Ouazar, Djamal · Tessier, Manon · El Mortada, Hamza
**Dépôt** : [github.com/Loulou441/htr-manuscrits-XIIIe-siecle](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle)

---

## Sommaire

1. [Vue d'ensemble : de l'image au texte exploitable](#1-vue-densemble--de-limage-au-texte-exploitable)
2. [Où trouver le code](#2-où-trouver-le-code)
3. [Volet 1 — HTR : fine-tuning Kraken](#3-volet-1--htr--fine-tuning-kraken)
4. [Volet 2 — NLP : normalisation, correction, structuration](#4-volet-2--nlp--normalisation-correction-structuration)
5. [Résultats globaux du projet](#5-résultats-globaux-du-projet)
6. [Limitations transversales](#6-limitations-transversales)
7. [Prochaines étapes](#7-prochaines-étapes)
8. [Installation rapide](#8-installation-rapide)
9. [Références](#9-références)
10. [Citation](#10-citation)

---

## 1. Vue d'ensemble : de l'image au texte exploitable

```
Volet 1 — HTR (branche fine_tuning)
════════════════════════════════════
33 manuscrits CREMMA/HTRomance (XIIIe s., ancien français + latin, 22 858 lignes)
        ↓  dataset.py (agrégation) → pre_traitement.py (deskew, CLAHE, filtres → mode L)
        ↓  compile_arrow.py (filtrage zones bruit) → ketos train (fine-tuning Kraken)
Modèle HTR fine-tuné — CER 26.3% (validation, Run 4)
        │
        │  appliqué à 16 NOUVEAUX manuscrits (Gallica/BnF, hors corpus d'entraînement)
        ↓
Volet 2 — NLP (branche nlp-pipeline-completed)
════════════════════════════════════
129 documents transcrits, 16 336 lignes → data contract JSON (texte + confiances + candidats)
        ↓  validate → eda → review-queue (triage confiance)
        ↓  normalisation par règles (Unicode, u/v, i/j, tilde, abréviations)
        ↓  correction guidée par confiance (CamemBERT MLM)
        ↓  détection lexicale + évaluation relative (CER pairwise) + split stratifié scellé
Texte médiéval normalisé, corrigé, prêt pour NER/POS/graphe/TEI (phase suivante, non démarrée)
```

Point important : **le Volet 2 ne retraite pas les 33 manuscrits d'entraînement du
Volet 1**. Il traite 16 manuscrits *supplémentaires*, transcrits avec le modèle une
fois celui-ci entraîné — deux corpus distincts, à ne pas confondre.

---

## 2. Où trouver le code

| Volet | Branche | Contenu | README détaillé |
|---|---|---|---|
| HTR (entraînement, prétraitement, démo) | [`fine_tuning`](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/tree/fine_tuning) | `src/`, `notebooks/`, `app.py` (démo Streamlit), `article/`, `docs/` | [README](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/blob/fine_tuning/README.md) |
| NLP (normalisation, correction, CLI) | [`nlp-pipeline-completed`](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/tree/nlp-pipeline-completed) | `nlp_pipeline/`, `notebooks/`, `docs/` | [README](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/blob/nlp-pipeline-completed/README.md) |

D'autres branches de travail existent (`hamza/entrainement`, `manon`, `legb`,
`nlp_comprehension_legb`, `bugfix/model_test_correction`,
`feature/caracter_counter_fine_tuning`) — les deux ci-dessus sont les branches de
référence, les plus abouties et documentées à ce jour pour chaque volet.

---

## 3. Volet 1 — HTR : fine-tuning Kraken

*(Résumé — détail complet, méthodologie pas à pas, courbes d'apprentissage et
diagnostic sur la [branche `fine_tuning`](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/blob/fine_tuning/README.md))*

### Objectif

Les modèles HTR génériques CREMMA (~95% de précision sur leur propre corpus de
validation) généralisent mal à un corpus non vu. Ce volet fine-tune `cremma-generic`
avec Kraken 7.x sur un corpus élargi de 33 manuscrits du XIIIe siècle.

| Niveau | CER | val_accuracy |
|---|:---:|:---:|
| Meilleure run actuelle (Run 4) | **26.3%** | 73.7% |
| Objectif validation | < 15% | > 85% |
| Objectif excellence | < 8% | > 92% |

### Corpus d'entraînement

| Indicateur | Valeur |
|---|---|
| Manuscrits | 33 (XIIIe siècle, 4 corpus HTR-United agrégés) |
| Langues | Ancien français et latin |
| Lignes totales (brut) | 22 858 |
| Lignes train (filtré, zones bruit exclues) | 18 769 |
| Script dominant | Gothic Textualis (92.5% des lignes) |
| Sources | CREMMA-Medieval, CREMMA-Medieval-LAT, HTRomance Medieval FR/LAT (HTR-United) |

Un 34e manuscrit (BnF fr. 25516) sert uniquement de test de généralisation, hors
corpus d'entraînement.

### Méthodologie

```
33 manuscrits (ALTO XML + JPEG)
    ├─ dataset.py ──────────── Agrégation des 4 corpus HTR-United + manifest.json
    ├─ pre_traitement.py ───── Deskew + CLAHE + filtres → mode L (grayscale)
    ├─ ketos compile ───────── Arrow binaire (train.arrow / dev.arrow)
    ├─ compile_arrow.py ────── Filtrage zones bruit (Music/DropCapital/Interlinear)
    └─ ketos train ─────────── Fine-tuning depuis cremma-generic-1.0.1 (GPU cloud)
```

### Résultats des runs

| Run | Plateforme | CER | Statut |
|---|---|:---:|---|
| 1–3 | Local / Kaggle / Colab | 27–30% | Runs préliminaires, logs partiels |
| **4** | **Kaggle T4 x2** | **26.3%** | **Meilleure run** (stage 27/37) |
| 5 | Kaggle T4 x2 | 26.3% | Identique Run 4 (confirme le plafond) |
| 6 | Colab T4 | ~26.5% | Aborté — mismatch confirmé |

**Diagnostic** : toutes les runs 1–6 utilisent des données **binarisées (mode 1)**
alors que `cremma-generic` a été entraîné en **grayscale (mode L)** — ce mismatch crée
un plafond artificiel à ~74% de val_accuracy. L'expérience en cours (Exp 3, Arrow
grayscale filtré) vise à lever ce plafond (+10–15 pts CER estimés).

Modèles publiés : [legb/htr-cremma-medieval](https://huggingface.co/legb/htr-cremma-medieval) (Hugging Face, CC-BY 4.0).

### Démo interactive

Une application Streamlit (`app.py`, branche `fine_tuning`) permet de tester les
modèles sur une image de manuscrit : upload, segmentation BLLA, transcription,
export du data contract JSON — le même format consommé par le Volet 2 NLP.

```bash
git checkout fine_tuning
pip install -r requirements.txt
streamlit run app.py
```

### Limitations connues (Volet 1)

- Mismatch mode L/1 non résolu sur toutes les runs sauf Exp 3 (en cours).
- Corpus limité (33 manuscrits) — sous-représentation de certains scribes et des
  scripts minoritaires (*Semitextualis Currens*, *Textualis Currens*).
- 22 caractères présents dans le train set mais absents de l'alphabet du modèle de
  base (gérés via `--resize union`, non comptabilisés dans l'accuracy officielle).
- Biais linguistique : majoritairement ancien français parisien.
- Reproductibilité GPU : Kraken ne supporte pas de seed globale fixe (±0.2% CER
  entre runs identiques).

---

## 4. Volet 2 — NLP : normalisation, correction, structuration

*(Résumé — détail complet, référence des 11 commandes CLI et section
reproductibilité complète sur la [branche `nlp-pipeline-completed`](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/blob/nlp-pipeline-completed/README.md))*

### Objectif

Le modèle HTR du Volet 1 a été utilisé pour transcrire 16 manuscrits supplémentaires
depuis Gallica/BnF (129 documents, 16 336 lignes), produisant pour chaque page un
**data contract JSON** (texte, confiance par caractère, candidats alternatifs). Ce
volet transforme cette sortie brute en texte exploitable, sans vérité terrain
complète disponible — l'évaluation y est donc majoritairement **relative** (CER entre
variantes) plutôt qu'absolue.

### Pipeline

```
Data contract HTR (JSON brut, 129 documents)
        ↓ validate (schéma JSON strict) → eda (métriques exploratoires)
        ↓ review-queue (triage direct / review / exclude par confiance)
        ↓ normalisation par règles : NFC, u/v, i/j, tilde nasal, table d'abréviations
        ↓ correction guidée par confiance : CamemBERT MLM (almanach/camembert-base)
        ↓ détection lexicale (dictionnaire ancien français) + évaluation relative (CER pairwise)
        ↓ split stratifié (siècle × type de document) + scellement SHA-256 du test set
```

Toutes les étapes sont exposées via un CLI unifié :
`python nlp_pipeline/nlp_cli.py <commande> [options]` — 11 commandes au total
(`validate`, `eda`, `review-queue`, `normalize`, `normalize-contract`, `correct`,
`ablation`, `relative-eval`, `detect-normalization`, `lexical-check`, `split`).

### Résultats (run du 18 juin 2026, 129 documents, 16 336 lignes)

| Mesure | Valeur |
|---|---|
| Documents validés contre le schéma | 129 / 129 (100%) |
| Confiance HTR moyenne | 0.793 |
| Lignes signalées pour révision | 36.8% |
| Paires de mots corrigées par les règles de normalisation | 3725 |
| CER pairwise moyen (raw / normalisé / corrigé) | 0.0667 |
| Tokens couverts par le dictionnaire ancien français | 4.4% |
| Tests unitaires | 22 / 22 |

### Reproductibilité

Points déjà solides : seed fixée (`--seed 67`) pour le split stratifié, test set
scellé et vérifiable par hash SHA-256, correction MLM déterministe (modèle en mode
évaluation par construction, pas d'échantillonnage), 22 tests unitaires entièrement
autonomes (aucune dépendance au corpus réel).

Points encore ouverts : dépendances non verrouillées (bornes basses uniquement dans
`requirements.txt`), version de Python non documentée formellement, modèle CamemBERT
non épinglé à une révision Hugging Face précise. Détail complet dans la section
*Reproductibilité* du [README de la branche NLP](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/blob/nlp-pipeline-completed/README.md#14-reproductibilité).

### Limitations connues (Volet 2)

- **Détection lexicale (4.4% de couverture)** : limite de la ressource externe (mots-outils absents du dictionnaire), pas un échec de la normalisation.
- **Correction guidée par confiance** : `candidates` est `null` sur la quasi-totalité des lignes réelles → le scorer CamemBERT, bien qu'actif par défaut, n'a rien à arbitrer sur ce run (0 correction). Le mécanisme est fonctionnel et a été vérifié sur données synthétiques.
- **Règle u/v** : compromis assumé qui empêche la correction de `deuient→devient` pour éviter davantage de faux positifs ailleurs.

---

## 5. Résultats globaux du projet

| | Volet 1 — HTR | Volet 2 — NLP |
|---|---|---|
| Corpus | 33 manuscrits (entraînement) | 16 manuscrits / 129 documents (transcrits en aval) |
| Lignes | 22 858 (brut), 18 769 (filtré train) | 16 336 |
| Métrique clé | CER 26.3% (objectif < 15%) | CER pairwise 0.0667 (relatif, pas de vérité terrain) |
| Statut | Exp 3 (grayscale) en cours pour lever le plafond à 74% | Normalisation + triage opérationnels ; NER/POS/graphe/TEI non démarrés |
| Tests automatisés | `pytest tests/` (Volet HTR) | `pytest nlp_pipeline/tests/ -q` — 22/22 |

---

## 6. Limitations transversales

- **Aucun des deux volets ne dispose d'une vérité terrain complète** sur son corpus de production (Volet 1 : set de test scellé mais évaluation détaillée par script encore *à compléter* ; Volet 2 : évaluation relative uniquement). Les deux README détaillés documentent précisément ce qui est mesuré et comment.
- **Reproductibilité des dépendances** : les deux volets utilisent des `requirements.txt` avec bornes basses uniquement (`>=`), sans lockfile — un point d'attention commun avant tout rendu final ou publication.
- **Données lourdes non versionnées** (corpus, modèles, Arrow, dictionnaire ancien français) : gérées via S3 (Volet 1) et exclues par `.gitignore` (Volet 2) — cloner le dépôt reproduit le code et les tests, pas les données de production.

---

## 7. Prochaines étapes

**Volet 1 (HTR)** : valider l'hypothèse grayscale (Exp 3), évaluer sur le set de test scellé, analyse différentielle par script paléographique, comparaison optionnelle TrOCR (LoRA) vs Kraken.

**Volet 2 (NLP)** : générer de vraies entrées `candidates` pour exercer réellement le scorer CamemBERT, enrichir le dictionnaire de référence, verrouiller les dépendances (`pyproject.toml`/lockfile). Plan « after » en 4 phases séquentielles : baseline NER, fine-tuning léger NER, POS + extraction de relations par règles, graphe NetworkX + export TEI-XML — évaluation à chaque phase via CER relatif, faute de vérité terrain.

---

## 8. Installation rapide

```bash
git clone https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle.git
cd htr-manuscrits-XIIIe-siecle

# Volet 1 — HTR (entraînement, prétraitement, démo Streamlit)
git checkout fine_tuning
pip install -r requirements.txt

# Volet 2 — NLP (normalisation, correction, CLI)
git checkout nlp-pipeline-completed
pip install -r requirements.txt --break-system-packages
pytest nlp_pipeline/tests/ -q
```

Chaque branche a son propre `requirements.txt` et sa propre suite de tests — voir les
README de branche pour le détail des commandes et options.

---

## 9. Références

### Corpus et données

- **CREMMA-Medieval** / **CREMMA-Medieval-LAT** — HTR-United / ENC-PSL.
- **HTRomance Medieval French / Latin** — HTRomance-Project.

### Modèles et frameworks

- **cremma_generic** — Pinche, A. (2022). Zenodo. DOI: [10.5281/zenodo.7234166](https://doi.org/10.5281/zenodo.7234166)
- **Kraken** — Kiessling, B. (2019). *Kraken — an Universal Text Recognizer for the Humanities*. DH2019.
- **almanach/camembert-base** — modèle de langue français (Hugging Face), utilisé pour la correction guidée par confiance (Volet 2).
- **SegmOnto** — schéma d'annotation de zones, [segmonto.github.io](https://segmonto.github.io).

### Conventions et méthodologie

- **Pinche, A.** (2022). *Guide de transcription pour les manuscrits du Xe au XVe siècle*. HAL.
- **Sauvola & Pietikäinen** (2000). *Adaptive Document Image Binarization*. Pattern Recognition, 33(2).
- **Zuiderveld, K.** (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV.

### Documentation complète

- [README — Volet 1 HTR (branche `fine_tuning`)](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/blob/fine_tuning/README.md)
- [README — Volet 2 NLP (branche `nlp-pipeline-completed`)](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle/blob/nlp-pipeline-completed/README.md)
- Article scientifique complet (format IEEE) : `article/HTR_Manuscrits_XIIIe_siecle.pdf` sur la branche `fine_tuning`.

---

## 10. Citation

```bibtex
@misc{htr-manuscrits-xiiie-siecle-2026,
  title  = {HTR + NLP sur manuscrits médiévaux du XIIIe siècle : fine-tuning Kraken et pipeline de normalisation},
  author = {Ouazar, Djamal and Tessier, Manon and El Mortada, Hamza},
  year   = {2026},
  url    = {https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle}
}
```