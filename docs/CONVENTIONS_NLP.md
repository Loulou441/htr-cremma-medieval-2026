# NLP - Implementation dans le projet

Ce document explicite comment l'utilisation du NLP a ete applique au projet, sans suppression de fichiers existants.

## 1. Validation du data contract HTR

- Schema JSON ajoute: `nlp_pipeline/json_files/htr_data_contract_schema.json`
- Validation schema + controles logiques (taille `char_confidences` == taille `text`):
  - `nlp_pipeline/htr_data_contract.py` -> `validate_contract()`
- Commandes:

```bash
python nlp_pipeline/nlp_cli.py validate --input data/contracts/htr_contract.json
python nlp_pipeline/nlp_cli.py validate --input data/nlp_output
```

## 2. EDA corpus HTR

Metriques implementees (cours J1):

- confiance moyenne
- mediane de longueur de ligne
- taux `needs_review`
- taux de lignes courtes `< 10`
- abreviations residuelles par ligne (`~`, `ꝑ`, `ꝗ`, `ꝓ`, `ꝙ`)

Code:

- `nlp_pipeline/htr_data_contract.py` -> `compute_eda()`

Commandes:

```bash
python nlp_pipeline/nlp_cli.py eda --input data/contracts/htr_contract.json --output reports/eda_day1.json
python nlp_pipeline/nlp_cli.py eda --input data/nlp_output --output reports/eda_nlp_output.json
```

## 3. Strategie de triage confidence / needs_review

Regles implementees:

- `confidence < 0.60` -> exclusion auto
- `0.60 <= confidence < 0.90` -> review
- `confidence >= 0.90` -> ingestion directe
- override review si:
  - `needs_review == true`
  - ecart-type `char_confidences > 0.2`

Sorties:

- CSV de revue humain-in-the-loop
- JSON des buckets `direct/review/exclude`

Code:

- `nlp_pipeline/htr_data_contract.py` -> `split_review_buckets()`, `export_review_csv()`

Commandes:

```bash
python nlp_pipeline/nlp_cli.py review-queue --input data/contracts/htr_contract.json
python nlp_pipeline/nlp_cli.py review-queue --input data/nlp_output
```

## 4. Normalisation par regles

Normaliseur en classe independante, regles activables/desactivables (ablation possible):

- NFC Unicode
- minuscule
- regles `u/v`
- regles `i/j`
- expansion tilde (`a~`, `e~`, `o~`)
- table d'abreviations JSON

Code:

- `nlp_pipeline/normalization_rules.py` -> `NormalizerConfig`, `MedievalFrenchNormalizer`
- table par defaut: `nlp_pipeline/json_files/medieval_abbreviations.json`

Commandes:

```bash
python nlp_pipeline/nlp_cli.py normalize --text "Et li cuens prist la d~e"
python nlp_pipeline/nlp_cli.py normalize --csv-input data/input.csv --csv-output data/normalized/output.csv
```

Note : la commande `normalize-contract` (qui applique le normaliseur a un data contract complet, par opposition a `normalize` sur du texte brut/CSV) calcule egalement, depuis cette mise a jour, le **CER pairwise** (`raw` vs `normalized_text`) ligne par ligne et en moyenne, exporte via `--cer-output` — meme principe que pour `correct` (section 6).

```bash
python nlp_pipeline/nlp_cli.py normalize-contract \
  --input data/nlp_output \
  --output-dir data/nlp_output_normalized \
  --cer-output data/review/normalize_cer_report.json
```

## 5. CER et tableau d'ablation

Code CER:

- `nlp_pipeline/cer_utils.py` -> `cer()`

Ablation (avant/apres normalisation):

```bash
python nlp_pipeline/nlp_cli.py ablation --csv-input data/reference_200.csv --reference-col reference --hypothesis-col text
```

Evaluation relative (sans verite terrain, comparaison entre variantes) :

```bash
python nlp_pipeline/nlp_cli.py relative-eval \
  --csv-input data/review/relative_eval_sample.csv \
  --variant-cols raw,text_normalized,corrected
```

## 6. Correction contextuelle guidee par confiance

Implementation operationnelle pour J1 (mise a jour : MLM active par defaut + reinjection) :

- detection des positions ambiguës selon `char_confidences` + `candidates`
- selection de variante par scorer contextuel : **CamemBERT en mode MLM par defaut** (`almanach/camembert-base`), scorer heuristique disponible en repli via `--no-mlm`
- trace JSONL des corrections
- mise a jour du contrat corrige
- **reinjection** : recalcul de `needs_review` ligne par ligne apres correction (desactivable via `--no-review-update`)
- **CER pairwise par ligne et moyen** (raw vs corrige), exporte via `--cer-output`

Code:

- `nlp_pipeline/confidence_correction.py` -> `ConfidenceGuidedCorrector`, `MaskedLMVariantScorer`, `HeuristicVariantScorer`, `LineCorrectionResult`

Commandes:

```bash
# MLM actif par defaut (CamemBERT) :
python nlp_pipeline/nlp_cli.py correct --input data/contracts/htr_contract.json --output data/contracts/htr_contract.corrected.json --log-output data/review/correction_log.jsonl --cer-output data/review/correction_cer_report.json
python nlp_pipeline/nlp_cli.py correct --input data/nlp_output --output-dir data/nlp_output_corrected --log-output data/review/correction_log.jsonl

# Scorer heuristique de repli (sans transformers/torch) :
python nlp_pipeline/nlp_cli.py correct --input data/contracts/htr_contract.json --output data/contracts/htr_contract.corrected.json --no-mlm
```

Note:
- Le cours mentionne CamemBERT MLM pour le scoring (J2 detaille) : c'est desormais le comportement par defaut de `correct`, et non plus une option a activer manuellement. Le scorer heuristique reste disponible (`--no-mlm`) pour les environnements sans GPU/sans `transformers` installe.
- Sur les donnees reelles de ce corpus, `candidates` est presque toujours `null` (cf. `PRESENTATION_NLP.md` section 5) : meme avec le MLM actif, 0 correction est appliquee faute de variantes a arbitrer. Le MLM est neanmoins le chemin par defaut, pret a s'activer dès que des `candidates` existent (HTR ou heuristique de substitution par frequence, cf. section 6 de `PRESENTATION_NLP.md`).

## 7. Split stratifie + test set scelle

Implementation:

- stratification sur `(century_estimate, document_type)`
- generation `train/val/test`
- scellement du test set (`test_sealed.json`) et hash SHA-256 (`test_set.sha256`)

Code:

- `nlp_pipeline/htr_data_contract.py` -> `stratified_split_records()`, `seal_test_set()`

Commande:

```bash
python nlp_pipeline/nlp_cli.py split --records data/documents_metadata.json --output-dir data/splits_nlp
```

## 8. Detection lexicale

Deux commandes complementaires, toutes deux implementees dans `normalization_rules.py` :

- `detect-normalization` : repere les tokens porteurs de marqueurs d'abreviation residuels et propose des expansions.
- `lexical-check` : flague les tokens (normalises) absents du dictionnaire ancien francais fourni via `--dictionary`.

Code:

- `nlp_pipeline/normalization_rules.py` -> `detect_normalization_candidates()`, `find_lexical_errors()`

Commandes:

```bash
python nlp_pipeline/nlp_cli.py detect-normalization --output-dir data/nlp_output --top-n 50
python nlp_pipeline/nlp_cli.py lexical-check --dictionary data/dictionary/dictionnaire_ancien_francais.json --output-dir data/nlp_output --top-n 30
```

## 9. Tests automatiques

Tests (`nlp_pipeline/tests/`):

- `test_cer_utils.py` — CER, WER
- `test_htr_data_contract.py` — validation de schema, EDA, triage
- `test_nlp_cli.py` — CER pairwise moyen
- `test_nlp_cli_defaults.py` — les chemins par defaut (`DEFAULT_SCHEMA`, `DEFAULT_ABBR`) pointent vers des fichiers reels
- `test_normalization_rules.py` — regles de normalisation, detection d'abreviations, erreurs lexicales
- `test_normalization_cer_regression.py` — non-regression du CER sur un echantillon de reference

Lancer:

```bash
pytest nlp_pipeline/tests/ -q
```

## 10. Dependances ajoutees

`requirements.txt`:

- `jsonschema>=4.21`
- `pytest>=8.0`
- `transformers>=4.0`, `sentencepiece>=0.1.0` (CamemBERT MLM, section 6)
- `Pillow>=10.0` (requis par `evaluate_model.py`)