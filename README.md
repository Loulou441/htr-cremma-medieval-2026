# Pipeline NLP — Manuscrits médiévaux XIIIe siècle

Volet NLP du projet HTR Manuscrits XIIIe Siècle. Ce document couvre **uniquement le pipeline NLP** de cette branche : ingestion du data contract HTR, normalisation par règles, correction guidée par confiance (CamemBERT MLM), détection lexicale, évaluation relative (CER pairwise) et split stratifié.

> Le pipeline NLP prend en entrée la sortie du modèle HTR (Kraken) développé dans le Volet 1 du projet (`nlp_pipeline/batch_transcribe.py`). L'entraînement HTR lui-même (CER, architecture, expériences) n'est pas couvert ici.

---

## Sommaire

1. [Contexte : d'où vient le NLP dans ce projet](#1-contexte--doù-vient-le-nlp-dans-ce-projet)
2. [Le data contract HTR](#2-le-data-contract-htr)
3. [Architecture du pipeline NLP](#3-architecture-du-pipeline-nlp)
4. [Validation et EDA](#4-validation-et-eda)
5. [Stratégie de triage / review queue](#5-stratégie-de-triage--review-queue)
6. [Normalisation par règles](#6-normalisation-par-règles)
7. [Correction guidée par confiance (CamemBERT MLM)](#7-correction-guidée-par-confiance-camembert-mlm)
8. [Détection lexicale](#8-détection-lexicale)
9. [Évaluation : CER pairwise à chaque étape](#9-évaluation--cer-pairwise-à-chaque-étape)
10. [Split stratifié et test set scellé](#10-split-stratifié-et-test-set-scellé)
11. [Référence des commandes CLI](#11-référence-des-commandes-cli)
12. [Structure des fichiers de la branche](#12-structure-des-fichiers-de-la-branche)
13. [Tests](#13-tests)
14. [Reproductibilité](#14-reproductibilité)
15. [Résultats sur le corpus complet](#15-résultats-sur-le-corpus-complet)
16. [Limites connues](#16-limites-connues)
17. [Prochaines étapes](#17-prochaines-étapes)

---

## 1. Contexte : d'où vient le NLP dans ce projet

Le Volet 1 du projet a entraîné un modèle **Kraken** (fine-tuning de `cremma-generic`) sur des corpus issus de HTR United (ancien français + latin, XIIIe siècle), atteignant un **CER de 26.3%** en validation. Ce modèle a ensuite été utilisé pour transcrire des manuscrits supplémentaires depuis Gallica/BnF (129 documents, 16 336 lignes sur le run de référence du 18 juin), produisant pour chaque page un JSON structuré — le **data contract HTR** — contenant le texte transcrit, les scores de confiance par caractère, et des candidats alternatifs pour les positions ambiguës.

Cette branche prend cette sortie HTR brute et la transforme en texte exploitable :

```
Sortie HTR (JSON brut)
        ↓
1. Validation du data contract (jsonschema)
        ↓
2. EDA (confiance, longueur, taux needs_review, abréviations)
        ↓
3. Triage / review queue (direct / review / exclude)
        ↓
4. Normalisation par règles (Unicode, u/v, i/j, tilde, abréviations)
        ↓
5. Correction guidée par confiance (CamemBERT MLM sur les positions ambiguës)
        ↓
6. Détection lexicale (dictionnaire ancien français, fourni séparément — voir section 8)
        ↓
7. Évaluation relative (CER pairwise) + split stratifié scellé
```

Comme pour le HTR, **aucune vérité terrain complète** n'existe pour ces manuscrits : l'évaluation est donc **relative** (comparaison entre variantes du même texte) plutôt qu'absolue, sauf sur un échantillon annoté manuellement pour le tableau d'ablation.

---

## 2. Le data contract HTR

Chaque page transcrite par le modèle HTR est un JSON validé par un schéma strict (`nlp_pipeline/json_files/htr_data_contract_schema.json`) :

```json
{
  "document_id": "gallica_bnf_fr_12483_f003",
  "metadata": {
    "source": "Gallica",
    "century_estimate": "XIII",
    "document_type": "roman",
    "sha256": "a3f2b1..."
  },
  "pages": [{
    "page_id": "f003r",
    "image_path": "f003r.tif",
    "lines": [{
      "line_id": "f003r_l001",
      "text": "Et li cuens Artus prist la d~e par la main",
      "confidence": 0.87,
      "char_confidences": [0.99, 0.99, 0.88, 0.72, 0.43, ...],
      "candidates": {"position": 4, "options": ["a", "e", "o"]},
      "needs_review": false,
      "polygon": [[120, 45], [892, 45], [892, 68], [120, 68]],
      "reading_order": 1
    }]
  }]
}
```

### Champs critiques pour le NLP

| Champ | Rôle |
|---|---|
| `text` | Transcription semi-diplomatique (abréviations non résolues conservées). |
| `confidence` | Score global de la ligne (0–1), utilisé par le triage (section 5). |
| `char_confidences` | Liste par caractère, de même longueur que `text`. |
| `candidates` | Pour les positions ambiguës, propositions alternatives (objet ou liste d'objets `{position, options}`). Peut être `null`. |
| `needs_review` | Booléen ; recalculé après correction (voir [section 7](#7-correction-guidée-par-confiance-camembert-mlm)). |
| `polygon` | Ancrage spatial (utile pour relier entités nommées et image en aval). |

**129/129 documents valides** contre le schéma sur le corpus complet du run de référence.

Ce même format est produit par `nlp_pipeline/batch_transcribe.py` (`build_data_contract()`) à partir des prédictions Kraken — c'est le point de jonction exact entre le Volet HTR et ce pipeline NLP.

---

## 3. Architecture du pipeline NLP

Tout le code de cette branche est regroupé dans un seul package `nlp_pipeline/` :

```
nlp_pipeline/
├── htr_data_contract.py         # validation, EDA, triage, split stratifié + scellement
├── normalization_rules.py       # normaliseur par règles + détection lexicale/abréviations
├── confidence_correction.py     # correction guidée par confiance (CamemBERT MLM)
├── cer_utils.py                  # CER, WER, CER pairwise moyen
├── nlp_cli.py                    # CLI unifié exposant toutes les commandes
├── json_files/                   # schéma + table d'abréviations (config, pas du code)
│   ├── htr_data_contract_schema.json
│   └── medieval_abbreviations.json
└── tests/                        # suite pytest (voir section 13)
```

Toutes les commandes NLP sont exposées via un seul point d'entrée :

```bash
python nlp_pipeline/nlp_cli.py <commande> [options]
```

Les commandes disponibles : `validate`, `eda`, `review-queue`, `normalize`, `normalize-contract`, `correct`, `ablation`, `relative-eval`, `detect-normalization`, `lexical-check`, `split`.

**Chemins par défaut** : `nlp_cli.py` calcule `DEFAULT_SCHEMA` et `DEFAULT_ABBR` relativement à son propre emplacement (`Path(__file__).parent / "json_files" / ...`), donc les commandes fonctionnent sans `--schema`/`--abbreviations` explicite quel que soit le répertoire depuis lequel on les lance — voir [test_nlp_cli_defaults.py](#13-tests) qui verrouille cette garantie.

> **Note** : `nlp_pipeline/` contient aussi trois scripts qui ne font pas partie de la logique NLP elle-même — `batch_transcribe.py` et `evaluate_model.py` (Volet HTR : extraction/transcription batch et évaluation CER/WER du modèle Kraken) et `sync_to_s3.py` (synchronisation infra vers S3). Les trois sont vérifiés fonctionnels (imports propres) mais restent hors du périmètre NLP décrit dans ce document — leur présence dans ce dossier est un choix d'organisation du dépôt, pas une dépendance technique envers le reste du package.

---

## 4. Validation et EDA

### Validation du schéma

```bash
python nlp_pipeline/nlp_cli.py validate --input data/nlp_output
```

Vérifie la conformité au schéma JSON **et** des règles logiques additionnelles : `len(char_confidences) == len(text)`, présence et validité du `polygon`.

### Analyse exploratoire (EDA)

```bash
python nlp_pipeline/nlp_cli.py eda --input data/nlp_output --output reports/eda_report.json
```

Métriques calculées :
- confiance moyenne et quartiles,
- médiane de longueur de ligne,
- taux de lignes `needs_review`,
- taux de lignes courtes (< 10 caractères),
- abréviations résiduelles par ligne (`~`, `⁊`, `ꝑ`, `ꝗ`, `ꝓ`, `ꝙ`).

**Repères typiques sur un corpus CREMMA moyen-français** : taux `needs_review` 15–30%, confiance moyenne 0.78–0.88, longueur médiane 45–65 caractères. Au-delà de 40% de `needs_review`, le corpus est jugé difficile (scan de mauvaise qualité ou modèle HTR sous-performant).

---

## 5. Stratégie de triage / review queue

Logique à trois niveaux, ajustable par seuils :

| Niveau | Confiance | Action |
|---|---|---|
| Validation automatique | `> 0.90` | Ingestion directe. |
| File de révision | `0.60–0.90` ou `needs_review = true` | Export CSV pour correction humaine, trié par confiance croissante. |
| Exclusion | `< 0.60` | Retranscription manuelle ou vérification de l'image source. |

Une ligne est **forcée en révision** même si sa confiance moyenne est bonne (`≥ 0.90`) si l'écart-type de ses `char_confidences` dépasse `0.2` — une moyenne flatteuse peut masquer un caractère critique très incertain (ex. *Louis* vs *Louise*).

```bash
python nlp_pipeline/nlp_cli.py review-queue \
  --input data/nlp_output \
  --csv-output data/review/review_queue.csv \
  --json-output data/review/review_buckets.json \
  --direct-threshold 0.9 --exclude-threshold 0.6 --char-std-threshold 0.2
```

---

## 6. Normalisation par règles

Avant toute correction statistique/IA, six règles déterministes sont appliquées, dans cet ordre, via une classe indépendante et toggleable (`MedievalFrenchNormalizer`) :

1. **NFC Unicode** — formes combinées → précomposées.
2. **lowercase** — les majuscules médiévales sont rares et inconsistantes.
3. **u/v** — résolution contextuelle (`auant→avant`, `cheualier→chevalier`).
   *Exception* : les digrammes `qu`/`gu` et `u+i` (`lui`) gardent leur `u` vocalique — compromis qui empêche `deuient→devient`, accepté car les faux positifs corrigés sont bien plus fréquents que ce faux négatif.
4. **i/j** — `i` consonantique devant voyelle (sauf digramme `ie`/`ien`/`ier`).
5. **tilde nasal** — `a~/e~/o~` → `an/en/on`.
6. **table d'abréviations** (`nlp_pipeline/json_files/medieval_abbreviations.json`) — 14 marqueurs scribaux et latinismes (`⁊→et`, `ꝑ→per`, `dñs→dominus`, etc.), appliqués du plus long au plus court pour éviter les collisions.

```bash
# Sur du texte brut
python nlp_pipeline/nlp_cli.py normalize --text "Et li cuens prist la d~e"

# Sur un data contract complet (ajoute normalized_text à chaque ligne)
python nlp_pipeline/nlp_cli.py normalize-contract \
  --input data/nlp_output \
  --output-dir data/nlp_output_normalized \
  --cer-output data/review/normalize_cer_report.json
```

### Impact mesuré sur le corpus complet

| Avant | Après | Occurrences |
|---|---|---|
| `qve` | `que` | 651 |
| `qvi` | `qui` | 420 |
| `bjen` | `bien` | 182 |
| `lvi` | `lui` | 182 |
| `qvil` | `quil` | 146 |
| `pvis` | `puis` | 112 |

→ **3725 paires de mots distinctes** corrigées par les règles sur le corpus complet.

---

## 7. Correction guidée par confiance (CamemBERT MLM)

### Principe (4 étapes)

1. **Identification** : parcourir les `char_confidences`, repérer les positions sous le seuil (défaut `0.7`) qui possèdent une entrée `candidates`.
2. **Arbitrage par modèle de langue** : pour chaque position ambiguë, générer une variante du texte par option candidate, masquer la position et faire scorer chaque variante par un modèle de langue masqué.
3. **Application** : retenir l'option au score le plus élevé, modifier le texte, journaliser la correction (`old_char`, `new_char`, `old_confidence`, position) dans un fichier `.jsonl`.
4. **Réinjection** : mettre à jour `needs_review` dans le contrat corrigé, et mesurer le CER pairwise introduit par la correction.

### Scorer : CamemBERT MLM activé par défaut

`ConfidenceGuidedCorrector` utilise par défaut `almanach/camembert-base` en mode *Masked Language Model* (`MaskedLMVariantScorer`) : chaque variante est évaluée en masquant la position ambiguë et en comparant la log-probabilité de chaque caractère candidat sous le modèle. Un scorer heuristique léger (`HeuristicVariantScorer`, basé sur la continuité alphabétique/vocalique) reste disponible en repli explicite via `--no-mlm`, pour les environnements sans GPU ou sans `transformers` installé.

```bash
# CamemBERT MLM (par défaut)
python nlp_pipeline/nlp_cli.py correct \
  --input data/nlp_output \
  --output-dir data/nlp_output_corrected \
  --log-output data/review/correction_log.jsonl \
  --cer-output data/review/correction_cer_report.json

# Scorer heuristique de repli
python nlp_pipeline/nlp_cli.py correct --input data/nlp_output --output-dir data/nlp_output_corrected --no-mlm
```

### Réinjection de `needs_review`

Après correction, `needs_review` est recalculé ligne par ligne (désactivable via `--no-review-update`) selon une logique prudente :

- une ligne ne repasse à `false` que si **toutes** les positions ambiguës connues (`candidates`) ont été traitées par le correcteur **et** que sa confiance globale (`≥ 0.9`) et l'écart-type de ses `char_confidences` (`≤ 0.2`) restent dans les seuils acceptables ;
- une ligne sans aucune entrée `candidates` garde son statut inchangé : le correcteur n'a aucune prise sur elle ;
- une ligne ne passe **jamais** de `false` à `true` (la réinjection ne peut qu'améliorer le statut, jamais le dégrader).

### Pourquoi 0 correction est attendu sur ce run

Sur le corpus actuel, `candidates` est **`null` sur la quasi-totalité des lignes** : le HTR ne produit pas de variantes alternatives. Même avec CamemBERT actif, le correcteur n'a donc rien à arbitrer aujourd'hui — ce n'est pas un bug, c'est documenté en [section 16](#16-limites-connues). Le mécanisme est néanmoins pleinement opérationnel et prêt à s'activer dès qu'une source de `candidates` existera (HTR multi-hypothèses, ou heuristique de substitution par fréquence, voir [section 17](#17-prochaines-étapes)).

---

## 8. Détection lexicale

Deux niveaux de détection, complémentaires, tous deux implémentés dans `normalization_rules.py` :

- **`detect-normalization`** : repère les marqueurs typographiques d'abréviation résiduels (`~`, `⁊`, `ꝑ`...) et propose des expansions automatiques pour les tokens non encore couverts par la table d'abréviations.
- **`lexical-check`** : vérifie si chaque token (normalisé) existe dans un dictionnaire ancien français fourni via `--dictionary` (format `mot → {wiktionary_en, cltk_fr}`). Les tokens absents sont des candidats à une vraie erreur lexicale — mot mal transcrit, abréviation non résolue, ou terme hors corpus.

Le dictionnaire (`resources/dictionnaire_ancien_francais.json`) est **versionné dans cette branche** — **172 734 entrées**, construites à partir de deux sources réelles :
- *Old French-English Wiktionary dictionary* (StarDict, [Vuizur/Wiktionary-Dictionaries](https://github.com/Vuizur/Wiktionary-Dictionaries)) — 5691 entrées + 122 667 variantes orthographiques médiévales rattachées via son fichier `.syn` ;
- *Lexique de l'ancien français* (Godefroy, [cltk/french_lexicon_cltk](https://github.com/cltk/french_lexicon_cltk)) — 53 145 entrées en français moderne.

`nlp_cli.py` pointe vers ce fichier par défaut (`DEFAULT_DICTIONARY`) — `--dictionary` reste disponible pour en fournir un autre.

```bash
python nlp_pipeline/nlp_cli.py detect-normalization --output-dir data/nlp_output --top-n 50
python nlp_pipeline/nlp_cli.py lexical-check --output-dir data/nlp_output --top-n 30
```

### Couverture mesurée — deux résultats, pas encore de mesure sur le corpus complet

| Corpus testé | Couverture | Contexte |
|---|---|---|
| Corpus de démonstration synthétique (propre, 14 lignes) | **66.7%** | Texte fabriqué mais grammaticalement correct une fois normalisé |
| Page réelle Gallica (*Roman de Troie*, f5, 511 tokens) | **27.2%** | Transcription réelle mais dégradée (confiance moyenne 0.62) — la moitié des tokens inconnus les plus fréquents sont des fragments d'1-2 lettres, pas de vrais mots |

Le chiffre de **4.4%** cité dans le run de référence du 18 juin (section 15) provient d'un dictionnaire différent (~55k entrées, jamais versionné, origine non documentée) appliqué au corpus complet de 129 documents. Ce nouveau dictionnaire n'a **pas encore été passé sur ce même corpus de 129 documents** (non disponible dans cet environnement) — les deux chiffres ci-dessus ne sont donc pas directement comparables au 4.4% historique. Ce qu'ils montrent : la couverture dépend fortement de la qualité de la transcription source, pas seulement de la taille du dictionnaire.

---


## 9. Évaluation : CER pairwise à chaque étape

Faute de vérité terrain complète, l'évaluation est **relative** : on mesure le CER entre variantes du même texte plutôt qu'un CER absolu.

Le CER pairwise est calculé à **chaque étape qui modifie le texte** :

| Étape | Calcul | Où |
|---|---|---|
| `normalize-contract` | CER(`raw`, `normalized_text`) par ligne + moyenne | `--cer-output` |
| `correct` | CER(`raw`, `corrected`) par ligne + moyenne | `--cer-output` |
| `relative-eval` | CER pairwise moyen entre N variantes au choix (`raw`, `text_normalized`, `corrected`...) sur un CSV | sortie console |

```bash
python nlp_pipeline/nlp_cli.py relative-eval \
  --csv-input data/review/relative_eval_sample.csv \
  --variant-cols raw,text_normalized,corrected
```

**Important** : `ablation` (CER avant/après) ne doit être utilisé qu'avec une vraie colonne de référence (transcription validée manuellement, ex. ALTO/PageXML). Sans cela, le `CER before` est trivialement nul (comparaison du texte brut à lui-même) et le `CER after` ne mesure que l'ampleur des changements introduits — pas un gain de qualité réel.

```bash
python nlp_pipeline/nlp_cli.py ablation \
  --csv-input data/reference_200.csv \
  --reference-col reference --hypothesis-col text --limit 200
```

`nlp_pipeline/evaluate_model.py` complète ce dispositif côté **HTR** : il calcule CER *et* WER (`cer_utils.wer`) d'un modèle Kraken face à une vérité terrain ALTO/PageXML.

---

## 10. Split stratifié et test set scellé

Le split `train/val/test` est **stratifié** sur deux dimensions simultanément — siècle (`century_estimate`) et type de document (`document_type`) — pour qu'aucune partition ne soit biaisée vers une seule période ou un seul genre de texte.

**Règle absolue** : une fois constitué, le test set est **scellé** (hash SHA-256 calculé et consigné immédiatement) et ne doit plus être consulté jusqu'au rendu final. Toutes les décisions d'architecture et d'hyperparamètres se prennent en observant uniquement la performance sur le jeu de validation.

```bash
python nlp_pipeline/nlp_cli.py split \
  --records data/documents_metadata.json \
  --output-dir data/splits_nlp \
  --train-ratio 0.8 --val-ratio 0.1 --seed 67
```

Produit `train.json`, `val.json`, `test_sealed.json` et `test_set.sha256`. Le `--seed 67` est la valeur par défaut du CLI — voir [section 14](#14-reproductibilité) pour ce que ça garantit exactement.

---

## 11. Référence des commandes CLI

| Commande | Rôle |
|---|---|
| `validate` | Valide un ou plusieurs data contracts contre le schéma JSON + règles logiques. |
| `eda` | Calcule les métriques exploratoires (confiance, longueur, abréviations, `needs_review`). |
| `review-queue` | Construit les buckets `direct/review/exclude` et exporte la file de révision en CSV. |
| `normalize` | Normalise du texte brut ou un CSV via les règles déterministes. |
| `normalize-contract` | Normalise un data contract complet (ajoute `normalized_text`), calcule le CER pairwise. |
| `correct` | Correction guidée par confiance (CamemBERT MLM par défaut), réinjecte `needs_review`, calcule le CER pairwise. |
| `ablation` | CER avant/après normalisation sur un échantillon de référence **annoté manuellement**. |
| `relative-eval` | CER pairwise moyen entre plusieurs variantes d'un même texte (CSV). |
| `detect-normalization` | Repère les tokens suspects (marqueurs d'abréviation) et propose des expansions. |
| `lexical-check` | Flague les tokens absents du dictionnaire ancien français. |
| `split` | Split stratifié + scellement du test set (SHA-256). |

Options communes à `normalize-contract` et `correct` : `--input` (fichier ou dossier de data contracts), `--output` / `--output-dir`, `--cer-output` (rapport JSON du CER pairwise).

Options spécifiques à `correct` : `--threshold` (seuil de confiance, défaut `0.7`), `--mlm-model` (défaut `almanach/camembert-base`), `--mlm-device` (défaut `auto`), `--no-mlm` (scorer heuristique de repli), `--no-review-update` (désactive la réinjection de `needs_review`), `--log-output` (journal `.jsonl` des corrections).

Tous les chemins par défaut (`--schema`, `--abbreviations`) sont résolus relativement à `nlp_pipeline/json_files/`, indépendamment du répertoire courant.

---

## 12. Structure des fichiers de la branche

Arborescence réellement versionnée sur `nlp-pipeline-completed` :

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── docs/
│   ├── CONVENTIONS_NLP.md         # conventions détaillées (normalisation, schéma BIO, etc.)
│   ├── PRESENTATION_NLP.md        # support de présentation
│   ├── RAPPORT_NLP_2026-06-18.md  # rapport détaillé daté (chronologie, résultats mesurés)
│   ├── RAPPORT_NLP_2026_07_04.md  # rapport de démonstration (corpus synthétique, scorer heuristique)
│   └── RAPPORT_NLP_2026_07_05.md  # rapport sur un vrai document Gallica (CamemBERT MLM réel)
├── notebooks/
│   ├── kaggle_eval_test.ipynb      # évaluation CER d'un modèle HTR (Volet HTR, sans lien avec nlp_cli.py)
│   ├── nlp_cli_kaggle.ipynb        # pipeline NLP bout-en-bout sur un document, environnement Kaggle + S3
│   └── nlp_cli_local.ipynb        # même démo, en local, sans clone Git ni sync S3
├── resources/
│   ├── manuscrits_xiii_siecle.txt         # liste source des manuscrits Gallica/BnF
│   └── dictionnaire_ancien_francais.json  # dictionnaire ancien français, 172 734 entrées (section 8)
└── nlp_pipeline/
    ├── htr_data_contract.py
    ├── normalization_rules.py
    ├── confidence_correction.py
    ├── cer_utils.py
    ├── nlp_cli.py
    ├── batch_transcribe.py         # hors périmètre NLP — extraction/transcription batch (Volet HTR)
    ├── evaluate_model.py           # hors périmètre NLP — évaluation HTR (CER/WER)
    ├── sync_to_s3.py                # hors périmètre NLP — infra
    ├── json_files/
    │   ├── htr_data_contract_schema.json
    │   └── medieval_abbreviations.json
    └── tests/
        ├── test_cer_utils.py
        ├── test_htr_data_contract.py
        ├── test_nlp_cli.py
        ├── test_nlp_cli_defaults.py
        ├── test_normalization_rules.py
        └── test_normalization_cer_regression.py
```

`nlp_cli_kaggle.ipynb` et `nlp_cli_local.ipynb` exécutent 7 des 11 commandes du CLI sur un document HTR unique : `validate`, `eda`, `review-queue`, `normalize-contract`, `correct`, `relative-eval`, `lexical-check`. `split` n'y figure pas volontairement : il opère sur une liste de métadonnées de documents (`--records documents_metadata.json`), pas sur un data contract individuel comme les étapes démontrées — voir [section 10](#10-split-stratifié-et-test-set-scellé) pour son usage réel sur le corpus complet.

**Répertoires runtime, non versionnés** (générés localement par le pipeline, exclus via `.gitignore` : `data/`, `reports/`, `models/`) — chemins utilisés dans les exemples de commandes ci-dessus :

```
data/
├── nlp_output/                    # data contracts bruts (sortie HTR)
├── nlp_output_normalized/         # après normalize-contract
├── nlp_output_corrected/          # après correct
├── review/                        # CSV, JSON, logs, rapports CER
├── splits_nlp/                    # train/val/test + scellement
└── downloads/                     # pages Gallica téléchargées par batch_transcribe.py
```

Le dictionnaire ancien français n'est **plus** dans cette liste : il est désormais versionné (`resources/dictionnaire_ancien_francais.json`, voir section 8), et `nlp_cli.py` l'utilise comme valeur par défaut de `--dictionary`.

> **Point d'attention `.gitignore`** : les commandes CLI écrivent leurs sorties par défaut sous `data/...` et leur journal d'exécution dans `data/review/nlp_cli_run_log.jsonl` — chemins **relatifs au répertoire courant**, pas à l'emplacement des scripts. Lancer une commande depuis `nlp_pipeline/` plutôt que depuis la racine du dépôt créerait donc un `nlp_pipeline/data/` parasite (déjà vu pendant la vérification de cette branche). Comme `.gitignore` ignore `data` sans égard à la profondeur, ce dossier resterait bien ignoré par git où qu'il apparaisse — mais toujours lancer les commandes `nlp_cli.py` depuis la racine du dépôt pour éviter la confusion.

---

## 13. Tests

```bash
pip install -r requirements.txt --break-system-packages
pytest nlp_pipeline/tests/ -q
```

**23 tests**, répartis sur 6 fichiers dans `nlp_pipeline/tests/` :
- `test_cer_utils.py` (4) — CER, WER, cohérence des deux conventions de calcul.
- `test_htr_data_contract.py` (4) — validation de schéma, EDA, triage.
- `test_nlp_cli.py` (1) — CER pairwise moyen.
- `test_nlp_cli_defaults.py` (3) — les chemins par défaut du CLI (`DEFAULT_SCHEMA`, `DEFAULT_ABBR`, `DEFAULT_DICTIONARY`) pointent vers des fichiers qui existent réellement sur le disque. Ajouté après deux bugs réels successifs : le déplacement du schéma et de la table d'abréviations vers `json_files/`, puis l'introduction de `DEFAULT_DICTIONARY` pointant vers `nlp_pipeline/resources/` alors que le dictionnaire vit à la racine du dépôt (`resources/`) — les deux avaient cassé leur commande respective (`validate`/`correct`, puis `lexical-check`) sans qu'aucun test ne le détecte.
- `test_normalization_rules.py` (8) — règles de normalisation, détection d'abréviations, erreurs lexicales.
- `test_normalization_cer_regression.py` (3) — non-régression du CER sur un échantillon de référence après normalisation (moyenne + ligne par ligne).

Ces 23 tests sont **entièrement autonomes** : aucun n'a besoin des données du corpus réel (`data/`, non versionnées) — ils tournent sur des exemples codés en dur, sur les fichiers réels de `json_files/`/`resources/`, ou dans des répertoires temporaires (`tmp_path`). C'est ce qui garantit qu'ils passent de manière identique sur n'importe quelle machine, y compris sans accès aux 129 manuscrits transcrits.

Dépendances NLP ajoutées : `jsonschema>=4.21`, `pytest>=8.0`, `transformers>=4.0`, `sentencepiece>=0.1.0` (CamemBERT MLM), `Pillow>=10.0` (requis par `evaluate_model.py`).

**Vérifier que le MLM est bien actif** (sans relancer tout le run) :

```bash
python nlp_pipeline/nlp_cli.py correct --input data/nlp_output --output /tmp/out.json
# → doit afficher : "Scorer : CamemBERT MLM (almanach/camembert-base)"
```

Si `transformers`/`torch` ne sont pas installés, la commande échoue avec un message explicite renvoyant vers `--no-mlm`, plutôt qu'un traceback brut.

---

## 14. Reproductibilité

Cette section liste, honnêtement, ce qui est garanti reproductible aujourd'hui et ce qui ne l'est pas encore.

### ✅ Ce qui est déjà solide

- **Seed fixée pour le split** : `stratified_split_records(..., seed=67)`, valeur par défaut du CLI (`--seed 67`). Deux exécutions de `split` sur les mêmes `--records` produisent exactement la même répartition `train/val/test`.
- **Test set scellé et vérifiable** : `seal_test_set()` calcule un SHA-256 du contenu JSON du test set au moment de sa création. Toute modification ultérieure (volontaire ou accidentelle) du fichier `test_sealed.json` est détectable en recalculant ce hash — c'est la garantie qu'aucune fuite de données de test n'a eu lieu pendant le développement.
- **Correction MLM déterministe par construction** : `AutoModelForMaskedLM.from_pretrained()` place le modèle en mode évaluation (`model.eval()`, dropout désactivé) dès le chargement — comportement par défaut de `transformers` depuis plusieurs versions majeures. Le scoring se fait uniquement par un `forward` sous `torch.no_grad()`, sans échantillonnage : à entrée et device identiques, la sortie est strictement identique d'une exécution à l'autre. Aucun `torch.manual_seed()` n'est donc nécessaire pour cette étape.
- **Chemins par défaut découplés du répertoire d'exécution** : `DEFAULT_SCHEMA`/`DEFAULT_ABBR` sont calculés via `Path(__file__).parent`, donc résolus par rapport à l'emplacement des fichiers Python et non au répertoire courant — les commandes se comportent de façon identique qu'on les lance depuis la racine du dépôt ou d'ailleurs. Ce point est maintenant verrouillé par `test_nlp_cli_defaults.py` (voir [section 13](#13-tests)) suite à une régression réelle où ces chemins pointaient vers un emplacement obsolète après réorganisation des fichiers JSON.
- **Tests unitaires 100% autonomes** (voir [section 13](#13-tests)) : reproductibles sur n'importe quelle machine sans accès au corpus réel.
- **Schéma JSON strict** (`json_files/htr_data_contract_schema.json`) : garantit que la structure de données consommée par le pipeline reste stable dans le temps, indépendamment de qui produit les data contracts.

### ⚠️ Ce qui n'est pas encore garanti — à corriger avant un rendu final

- **Dépendances non pinnées** : `requirements.txt` ne fixe que des bornes basses (`transformers>=4.0`, `torch>=2.4.0`, etc.), sans borne haute ni fichier de verrouillage (`pyproject.toml` + lockfile, ou `pip freeze`). Un `pip install -r requirements.txt` refait dans 6 mois peut résoudre des versions différentes de celles utilisées pour produire les résultats de la [section 15](#15-résultats-sur-le-corpus-complet). **Recommandation** : générer un `requirements-lock.txt` (`pip freeze`) au moment du rendu, ou migrer vers `pyproject.toml` avec versions exactes, comme demandé par les consignes du projet.
- **Version de Python non documentée** : aucun `.python-version` ni `requires-python` dans le dépôt. Ce document a été vérifié avec Python 3.12 ; à préciser explicitement pour que l'environnement soit reconstructible à l'identique.
- **Modèle CamemBERT non épinglé à une révision** : `almanach/camembert-base` est référencé par son seul nom de dépôt Hugging Face, sans `revision=<commit_sha>`. Le poids du modèle de base a très peu de chances de changer, mais épingler une révision précise (`from_pretrained(model_name, revision="...")`) supprimerait ce doute pour un rendu qui doit rester reproductible à long terme.
- **`--mlm-device auto` peut varier d'une machine à l'autre** : sur une machine avec GPU, `auto` choisira `cuda` ; sur une machine sans GPU, `cpu`. Les résultats du MLM sont déterministes *sur un device donné*, mais de très légers écarts de calcul flottant peuvent apparaître entre CPU et GPU. Pour comparer des runs bit-à-bit, fixer explicitement `--mlm-device cpu` (ou `cuda`) plutôt que `auto`.
- **Reproductibilité des résultats du corpus complet (section 15) ≠ reproductibilité du code** : les 129 documents et l'échantillon `reference_200.csv` ne sont **pas versionnés** dans cette branche (`.gitignore` exclut `data/`) — le dictionnaire ancien français, lui, l'est désormais (`resources/dictionnaire_ancien_francais.json`, section 8). Cloner ce dépôt et lancer `pytest` reproduit fidèlement le comportement du code (23/23 tests), mais **ne reproduit pas** à lui seul les chiffres de la section 15 — il faut disposer séparément des données sources (sortie HTR sur le corpus complet).
- **`stratified_split_records()` modifie l'état global de `random`** : elle appelle `random.seed(seed)` sur le module `random` global plutôt que d'utiliser une instance locale (`random.Random(seed)`). Sans conséquence pour un appel CLI isolé (process à usage unique), mais à corriger si cette fonction est un jour appelée plusieurs fois dans un même processus Python (notebook, pipeline orchestré).
- **Toujours lancer `nlp_cli.py` depuis la racine du dépôt** : les sorties par défaut (`data/review/...`) sont relatives au répertoire courant, pas à l'emplacement du script — voir l'encadré en [section 12](#12-structure-des-fichiers-de-la-branche).

---

## 15. Résultats sur le corpus complet

*Run du 18 juin 2026, 129 documents, 16 336 lignes :*

| Mesure | Valeur |
|---|---|
| Documents validés | 129 / 129 (100%) |
| Lignes analysées (EDA) | 16 336 |
| Confiance HTR moyenne | 0.793 |
| Lignes signalées pour révision | 36.8% |
| Tests unitaires (suite actuelle, pas figée au 18 juin) | 23 / 23 |
| CER pairwise moyen (raw / normalisé / corrigé) | 0.0667 |
| Tokens couverts par le dictionnaire ancien français (dictionnaire de l'époque, ~55k entrées — voir [section 8](#8-détection-lexicale)) | 4.4% |
| Paires de mots corrigées par les règles | 3725 |

> Voir [section 14](#14-reproductibilité) : ces chiffres proviennent du run du 18 juin sur des données non versionnées dans ce dépôt ; ils ne sont pas regénérables par un simple `git clone` + `pytest`.

---

## 16. Limites connues

- **Détection lexicale** : le dictionnaire de 172 734 entrées (section 8) n'a été mesuré que sur un corpus synthétique propre (66.7%) et une page réelle dégradée (27.2%) — pas encore sur le corpus complet de 129 documents, faute d'y avoir accès dans cet environnement. Le chiffre historique de 4.4% (section 15) utilisait un dictionnaire différent et plus petit ; il n'est pas comparable directement.
- **Correction guidée par confiance/MLM** : le scorer CamemBERT est actif par défaut, mais `candidates` est `null` sur la quasi-totalité des données réelles → 0 correction appliquée sur ce run. Le CER pairwise de l'étape `correct` est donc proche de 0 (texte inchangé), ce qui est attendu.
- **Réinjection `needs_review`** : pour la même raison, son effet n'est pas encore observable sur le corpus actuel.
- **Règle u/v** : exclut volontairement `u+i` pour éviter les faux positifs, empêchant `deuient→devient` — compromis documenté et accepté.
- **`ablation` vs `relative-eval`** : ne pas confondre les deux. `ablation` nécessite une vraie référence manuelle ; `relative-eval` ne mesure qu'une stabilité relative entre variantes, sans préjuger d'un gain de qualité.
- **Dépendances non verrouillées** : voir [section 14](#14-reproductibilité) — risque à moyen terme, pas un bug actuel.

---

## 17. Prochaines étapes

- Générer de vraies entrées `candidates` (HTR multi-hypothèses, ou heuristique de substitution par fréquence à partir du dictionnaire/lexique de référence) pour que le scorer CamemBERT, déjà actif, ait effectivement des variantes à arbitrer.
- Enrichir le dictionnaire de référence avec les mots-outils pour rendre `lexical-check` plus discriminant.
- Verrouiller les dépendances (`requirements-lock.txt` ou `pyproject.toml`) et documenter la version de Python utilisée — voir [section 14](#14-reproductibilité).
- **Plan « after »** (NER, POS, graphe, TEI) : 4 phases séquentielles non démarrées — baseline NER (`magistermilitum/roberta-multilingual-medieval-ner`), fine-tuning NER (CamemBERT + seqeval), POS (Stanza `frm`) + extraction de relations par règles, graphe NetworkX + export TEI-XML. Évaluation à chaque phase via CER **relatif** uniquement, toujours faute de vérité terrain.

---

## Références

- Pinche, A. — conventions graphématiques pour l'édition de manuscrits médiévaux (MUFI, SegmOnto, ChocoMufin).
- HTR United / CREMMA Medieval / CATMuS Medieval — corpus et modèles de référence HTR.
- `almanach/camembert-base` — modèle de langue français utilisé pour la correction guidée par confiance (Hugging Face).