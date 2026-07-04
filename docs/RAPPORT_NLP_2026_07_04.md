# Rapport — Run de démonstration du pipeline NLP (4 juillet 2026)

## ⚠️ Avertissement — nature de cette run

**Ceci n'est pas une run sur le corpus réel.** Les 129 manuscrits du run du 18 juin
(`RAPPORT_NLP_2026-06-18.md`), le dictionnaire ancien français (55k entrées) et
l'échantillon `reference_200.csv` sont des données non versionnées sur la branche
GitHub (exclues par `.gitignore`) — Le test ayant été fait sur un hardware n'y ayant pas accès depuis cet environnement.

Pour produire un rapport basé sur une **exécution réelle** plutôt que des chiffres
inventés, nous avons :
1. généré un petit corpus HTR **synthétique** (4 documents, 14 lignes, graphies
   médiévales plausibles mais fabriquées) ;
2. construit un dictionnaire ancien français factice (31 entrées) et un échantillon
   de référence pour l'ablation (4 lignes annotées à la main) ;
3. exécuté réellement les 11 commandes du CLI dessus, dans l'ordre du pipeline.

Tous les chiffres ci-dessous proviennent de cette exécution réelle sur ces données
synthétiques — ils démontrent que le pipeline fonctionne de bout en bout, mais **ne
remplacent pas** et ne sont **pas comparables** aux résultats du run du 18 juin sur
le vrai corpus (129 documents, 16 336 lignes).

Limite additionnelle de cet environnement : pas d'accès réseau à Hugging Face, donc
l'étape `correct` a été exécutée avec le scorer heuristique (`--no-mlm`), pas
CamemBERT MLM.

---

## 1. Corpus synthétique utilisé

| Document | Siècle | Type | Lignes |
|---|---|---|---|
| `demo_roman_xiii_01` | XIII | roman | 5 |
| `demo_roman_xiii_02` | XIII | roman | 3 |
| `demo_chronique_xiii_01` | XIII | chronique | 3 |
| `demo_chronique_xiv_01` | XIV | chronique | 3 |
| **Total** | | | **14** |

Les textes imitent des graphies médiévales réalistes : abréviations (`~`, `⁊`),
variantes u/v et i/j (`cheualier`, `auant`, `bjen`), et deux positions volontairement
ambiguës (`candidates` non nul, confiance caractère forcée < 0.5) pour exercer la
correction guidée par confiance.

Commande de génération : script `generate_demo_corpus.py` (seed fixée à 67, comme le
`--seed` par défaut du CLI).

---

## 2. Validation du data contract

```
python nlp_pipeline/nlp_cli.py validate --input data/nlp_output
```

**4/4 documents valides** contre `nlp_pipeline/json_files/htr_data_contract_schema.json`, aucune erreur de schéma ni de cohérence logique (`len(char_confidences) == len(text)`, `polygon` présent sur chaque ligne).

---

## 3. EDA

```
python nlp_pipeline/nlp_cli.py eda --input data/nlp_output --output data/review/eda_report.json
```

| Métrique | Valeur |
|---|---|
| Lignes analysées | 14 |
| Confiance moyenne | 0.9134 |
| Longueur médiane de ligne | 29 caractères |
| Taux `needs_review` | 35.7% (5/14) |
| Abréviations résiduelles / ligne | 0.357 |
| Lignes courtes (< 10 car.) | 0% |
| Répartition confiance | 12 lignes ≥ 0.9, 2 lignes entre 0.8 et 0.9, aucune < 0.8 |

Le taux `needs_review` élevé (35.7%) reflète le corpus synthétique volontairement
construit avec plusieurs lignes ambiguës ou marquées à revoir — pas un signal de
qualité HTR réelle.

---

## 4. Triage / review queue

```
python nlp_pipeline/nlp_cli.py review-queue --input data/nlp_output --csv-output data/review/review_queue.csv --json-output data/review/review_buckets.json
```

**Direct : 7 — Review : 7 — Exclude : 0**

Aucune ligne exclue (confiance minimale du corpus synthétique : 0.897, au-dessus du
seuil d'exclusion à 0.60). Les 7 lignes en révision combinent confiance modérée
(0.85–0.92) et écart-type de `char_confidences` élevé (positions ambiguës forcées).

---

## 5. Normalisation par règles

```
python nlp_pipeline/nlp_cli.py normalize-contract --input data/nlp_output --output-dir data/nlp_output_normalized --cer-output data/review/normalize_cer_report.json
```

**CER pairwise moyen (raw vs normalisé) : 0.0483**

Exemples de transformations observées sur ce corpus :
- `d~e` → `dame` (table d'abréviations)
- `cheualier auant` → `chevalier avant` (règle u/v)
- `⁊` → `et` (table d'abréviations)
- `bjen` → non corrigé par les règles (pas de règle j→i ; relève de la correction
  lexicale, pas de la normalisation déterministe)

---

## 6. Correction guidée par confiance

```
python nlp_pipeline/nlp_cli.py correct --input data/nlp_output --output-dir data/nlp_output_corrected --log-output data/review/correction_log.jsonl --cer-output data/review/correction_cer_report.json --no-mlm
```

**Scorer utilisé : heuristique (`--no-mlm`)** — pas d'accès réseau à Hugging Face
dans cet environnement pour télécharger `almanach/camembert-base`. Sur le vrai
corpus, `correct` utilise CamemBERT MLM par défaut (voir README section 7).

| Mesure | Valeur |
|---|---|
| Lignes traitées | 14 |
| Corrections appliquées | 2 (sur 2 lignes) |
| CER pairwise moyen (raw vs corrigé) | 0.0051 |
| `needs_review` après correction | 4 (contre 5 avant) |

Corrections réellement appliquées (journal `correction_log.jsonl`) :

| Ligne | Position | Avant | Après | Confiance d'origine |
|---|---|---|---|---|
| `demo_roman_xiii_01_p1_l2` | 0 | `x` | `q` | 0.3506 |
| `demo_chronique_xiv_01_p1_l2` | 0 | `x` | `q` | 0.3554 |

Ces deux lignes contenaient une position à confiance < 0.7 avec des `candidates`
n'incluant pas le caractère reconnu par le HTR — un cas où le mécanisme peut
réellement arbitrer. **Constat méthodologique** : sur un premier essai avec des
`options` composées de deux consonnes proches (`["q","k"]`) sans caractère fautif
clairement extérieur aux options, le scorer heuristique aboutit à une égalité de
score et ne corrige rien — comportement cohérent avec ce que documente déjà le
rapport du 18 juin sur le corpus réel (`candidates` peu exploitable sans un scorer
plus fin que l'heuristique).

---

## 7. CER par ablation (échantillon de référence annoté)

```
python nlp_pipeline/nlp_cli.py ablation --csv-input data/reference_sample.csv --reference-col reference --hypothesis-col text
```

Échantillon (4 lignes annotées à la main, construit pour cette démonstration) :

| Référence | Texte HTR |
|---|---|
| que dame | q~ d~e |
| chevalier avant | cheualier auant |
| et li roi | ⁊ li roi |
| bien savoir | bjen sauoir |

**CER before : 0.2593 — CER after : 0.0227 — Gain : 0.2366**

Le gain de CER (~24 points) illustre l'effet de la normalisation par règles sur des
graphies avec abréviations et variantes u/v/i, sur un échantillon volontairement
choisi pour les exercer. Sur le vrai corpus, `data/reference_200.csv` (200 lignes
annotées) donnerait une mesure représentative ; ces 4 lignes ne le sont pas.

---

## 8. Détection lexicale

```
python nlp_pipeline/nlp_cli.py detect-normalization --output-dir data/nlp_output --top-n 20
python nlp_pipeline/nlp_cli.py lexical-check --dictionary data/dictionary/dictionnaire_ancien_francais.json --output-dir data/nlp_output_normalized --top-n 20
```

**`detect-normalization`** : 71 tokens totaux, 4 tokens suspects (`⁊` ×2, `d~e`,
`x~`, `~e`) — cohérent avec les quelques marqueurs résiduels du corpus synthétique.

**`lexical-check`** : dictionnaire factice de 31 entrées, **46/71 tokens inconnus**
(coverage ≈ 35%). Ce taux n'est **pas comparable** au 4.4% du vrai corpus : mon
dictionnaire synthétique ne couvre qu'une trentaine de mots choisis à la main,
contre 55 000 entrées réelles construites depuis Wiktionary + CLTK.

**Constat technique confirmé en exécutant le code** : `lexical-check` tokenise
toujours le champ `text` brut des data contracts, jamais `normalized_text` — pointer
la commande vers `data/nlp_output_normalized` au lieu de `data/nlp_output` ne change
donc rigoureusement rien au résultat (vérifié : sorties identiques). Ce comportement
explique pourquoi le rapport du 18 juin voit encore des formes non normalisées
(`cheualier`, `auant`) dans son top des tokens inconnus — ce n'est pas un oubli
d'exécution, c'est le comportement actuel du code. À corriger si l'intention est de
vérifier le texte *normalisé* plutôt que le texte brut.

---

## 9. Évaluation relative (CER pairwise)

```
python nlp_pipeline/nlp_cli.py relative-eval --csv-input data/review/relative_eval_sample.csv --variant-cols raw,text_normalized,corrected
```

CSV construit à partir des 5 lignes du document `demo_roman_xiii_01` (colonnes
`raw`, `text_normalized`, `corrected`).

**CER pairwise moyen entre les 3 variantes : 0.0446** (5 lignes évaluées).

---

## 10. Split stratifié et test set scellé

```
python nlp_pipeline/nlp_cli.py split --records data/documents_metadata.json --output-dir data/splits_nlp --train-ratio 0.5 --val-ratio 0.25 --seed 67
```

**train=3, val=1, test=0**

Avec seulement 4 documents synthétiques répartis en 3 strates
(`(XIII,roman)`×2, `(XIII,chronique)`×1, `(XIV,chronique)`×1), le test set ressort
**vide** — la logique de `stratified_split_records()` retient au moins 1 document en
train et en val par strate avant d'allouer le reste au test, donc les strates à 1
seul document n'en fournissent aucun au test. **C'est une limite du corpus
synthétique (trop petit), pas un bug** : sur le vrai corpus (129 documents, plusieurs
dizaines par strate), ce problème ne se pose pas — voir le split réel documenté en
README section 10.

Hash du test set scellé (vide) : `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.

---

## 11. Tests unitaires

```
pytest nlp_pipeline/tests/ -q
```

**22/22 tests passés**, indépendamment de ce corpus synthétique (les tests utilisent
leurs propres fixtures, cf. README section 13).

---

## 12. Résumé

| Étape | Statut | Résultat clé |
|---|---|---|
| `validate` | ✅ | 4/4 documents valides |
| `eda` | ✅ | confiance moy. 0.913, needs_review 35.7% |
| `review-queue` | ✅ | 7 direct / 7 review / 0 exclude |
| `normalize-contract` | ✅ | CER pairwise moyen 0.0483 |
| `correct` (heuristique) | ✅ | 2 corrections appliquées, CER pairwise 0.0051 |
| `ablation` | ✅ | CER 0.2593 → 0.0227 (gain 0.2366) |
| `detect-normalization` | ✅ | 4 tokens suspects sur 71 |
| `lexical-check` | ✅ | 46/71 tokens inconnus (dico factice 31 entrées) |
| `relative-eval` | ✅ | CER pairwise moyen 0.0446 |
| `split` | ✅ (dégénéré) | train=3 / val=1 / test=0 — corpus trop petit |
| `pytest` | ✅ | 22/22 |

**Conclusion** : les 11 commandes du CLI s'exécutent correctement de bout en bout, y
compris leur enchaînement (`normalize-contract` → `correct` → `relative-eval`). Le
seul point de vigilance découvert pendant cette run (`lexical-check` qui ignore
`normalized_text`) est un comportement réel du code à connaître, pas un défaut de
cette démonstration. Pour un rapport avec des chiffres représentatifs du projet,
il faut relancer ces mêmes commandes sur le vrai corpus — ce que je ne peux pas
faire depuis cet environnement.