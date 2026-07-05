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

## 11. Pourquoi ces choix (rationale)

Les sections precedentes documentent *quoi* a ete implemente et *ou*. Cette section
documente *pourquoi*, pour les decisions qui ne sont pas evidentes a la lecture du
code seul.

### Pourquoi des regles deterministes avant tout traitement statistique/IA ?

La consigne du projet donne explicitement cet ordre de priorite : "la normalisation
est la brique la plus facile a mettre en oeuvre et celle qui apporte le plus de gain
en CER immediat [...] Commencez par les regles deterministes [...] avant la
correction guidee par confiance". Trois raisons pratiques a ce choix :

1. **Cout/benefice** : les regles (NFC, u/v, i/j, tilde, abreviations) ne demandent
   aucune donnee d'entrainement ni modele externe, et corrigent a elles seules 3725
   paires de mots distinctes sur le corpus complet (cf. README section 6) — un gain
   immediat et gratuit avant meme de brancher un modele de langue.
2. **Explicabilite** : une regle ratee est facile a diagnostiquer et corriger (un
   `if`/`else` de plus) ; une erreur de scoring MLM est beaucoup plus opaque a
   deboguer.
3. **Le traitement statistique (CamemBERT MLM) n'a de sens que sur ce qui reste
   ambigu apres les regles** : appliquer les regles d'abord reduit le nombre de
   positions que le correcteur guide par confiance a effectivement besoin
   d'arbitrer.

### Pourquoi NFC en premiere etape ?

Les manuscrits medievaux transcrits par HTR melangent des caracteres Unicode
combines et precomposes pour un meme signe diacritique (ex. un `o` suivi d'un tilde
combinant U+0303, versus un `õ` precompose). Sans normalisation NFC prealable, les
regles suivantes (u/v, i/j, expansion du tilde) devraient chacune gerer les deux
representations separement, ce qui double leur complexite et leur surface de bug
pour aucun gain. Faire NFC en premier garantit qu'une seule forme canonique arrive
aux regles suivantes.

### Pourquoi exclure `qu`/`gu` et `u+i` de la regle u/v ?

La regle u/v resout les variantes graphiques medievales par contexte (`auant` →
`avant`, `cheualier` → `chevalier`). Mais dans les digrammes `qu`/`gu` (`que`,
`qui`, `guerre`) et dans `u+i` (`lui`), le `u` est deja vocalique en francais moderne
— le convertir en `v` produirait des formes fausses (`qve`, `gverre`, `lvi`).
**Compromis assume** : cette exception empeche aussi de corriger `deuient` en
`devient` (un vrai cas de `u` consonantique suivi de `i`), mais les faux positifs
qu'elle evite (sur `que`, `qui`, `guerre`, `lui`, tres frequents) sont bien plus
nombreux que ce faux negatif ponctuel. Verifie et documente empiriquement dans
`test_uv_rule_keeps_qu_gu_and_ui_digraphs` (`tests/test_normalization_rules.py`).

### Pourquoi le seuil de confiance a 0.7 pour la correction guidee ?

Le seuil `--threshold 0.7` (par defaut dans `correct`) determine a partir de quelle
confiance caractere une position est consideree assez incertaine pour justifier
l'arbitrage d'un modele de langue. Il est volontairement different des seuils du
triage `review-queue` (0.60 exclusion / 0.90 ingestion directe, section 3) : ces
derniers decident si une **ligne entiere** merite une revue humaine, alors que 0.7
decide, **caractere par caractere**, s'il vaut la peine de solliciter CamemBERT.
Un seuil plus bas (ex. 0.5) laisserait passer des positions deja suffisamment
fiables sans les arbitrer ; un seuil plus haut (ex. 0.9) solliciterait le modele sur
des positions presque certaines, sans gain attendu et pour un cout de calcul inutile.
0.7 se situe dans l'intervalle de la strategie de triage (0.60-0.90) ou la confiance
est ambigue mais pas franchement mauvaise — la zone ou un arbitrage a le plus de
chances d'apporter un vrai gain.

### Pourquoi CamemBERT MLM plutot qu'un scorer heuristique par defaut ?

Le scorer heuristique (`HeuristicVariantScorer`) ne juge une variante que sur des
criteres de surface (le caractere est-il alphabetique ? une voyelle ? entoure de
lettres ?) — il ne "comprend" pas le mot ni la phrase. CamemBERT en mode *Masked
Language Model* evalue au contraire la probabilite de chaque candidat dans son
contexte linguistique reel, ce qui est strictement plus informatif des lors qu'un
GPU (ou meme un CPU raisonnable) est disponible. Le scorer heuristique reste
neanmoins le repli par defaut recommande (`--no-mlm`) pour les environnements sans
`transformers`/`torch installes, ou sans connectivite vers Hugging Face.

### Pourquoi le schema BIO n'est pas encore documente ici

Le schema d'annotation BIO (Beginning/Inside/Outside, standard pour la
reconnaissance d'entites nommees token par token) concerne la phase NER du plan
"after" du projet (README section 17, "Prochaines etapes") — **cette phase n'a pas
encore demarre**. Aucun code de ce depot ne produit ou ne consomme d'annotations BIO
a ce jour ; documenter un choix de schema maintenant serait premature et risquerait
de ne pas correspondre au modele de base finalement retenu (voir README section 17 :
`magistermilitum/roberta-multilingual-medieval-ner` ou equivalent CREMMA/CATMuS).
Cette section sera completee avec la justification du schema BIO (et de tout
regroupement de classes, ex. ajout d'une classe `TITLE` a cote de `PER`/`LOC`/`ORG`
comme suggere par la consigne du projet) au moment ou la phase NER demarrera
reellement, plutot que d'anticiper un choix non encore teste.

### Pourquoi une evaluation relative (CER pairwise) plutot qu'absolue ?

Aucune verite terrain complete n'existe pour les manuscrits transcrits (129
documents, section 1) — impossible de calculer un CER absolu par comparaison a une
transcription de reference humaine sur l'ensemble du corpus. La mesure retenue
(CER pairwise entre variantes successives d'une meme ligne : brute, normalisee,
corrigee) donne une courbe d'evolution sans necessiter cette verite terrain,
au prix de ne mesurer qu'un changement relatif et non un gain de qualite absolu.
Sur l'echantillon annote manuellement (`ablation`, section 5), un vrai CER
avant/apres reste calculable et vient completer cette mesure relative.

### Pourquoi stratifier le split sur `(century_estimate, document_type)` ?

Le corpus melange plusieurs siecles et plusieurs types de documents (roman,
chronique...). Un split aleatoire simple risquerait de concentrer par hasard un
siecle ou un type de document dans le train et un autre dans le test, biaisant
l'evaluation. Stratifier sur ces deux dimensions garantit que chaque strate est
representee proportionnellement dans train/val/test — condition necessaire pour
qu'une performance mesuree sur le test set soit representative du corpus dans son
ensemble, et pas seulement d'une de ses sous-populations.

### Pourquoi sceller le test set par SHA-256 ?

Le principe (consigne du projet, section 2) est de ne plus regarder le test set une
fois constitue, pour que les decisions d'architecture/hyperparametres restent
prises uniquement sur le jeu de validation. Le hash SHA-256 rend cette regle
verifiable plutot que declarative : si le contenu de `test_sealed.json` est modifie
(intentionnellement ou par erreur) apres scellement, le hash recalcule ne correspond
plus a `test_set.sha256`, et la fuite potentielle de donnees de test devient
detectable au lieu de reposer sur la seule discipline de l'equipe.