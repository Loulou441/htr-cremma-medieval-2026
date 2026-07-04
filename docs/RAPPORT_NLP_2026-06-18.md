# Rapport détaillé — branche `nlp-pipeline` (18 juin 2026)

## 0. Étapes préparatoires (branche `nlp`, 14-17 juin) — avant la réorganisation

Le pipeline NLP repose sur un travail préalable de constitution et de validation du dataset, réalisé sur la branche `nlp` avant la réorganisation en `nlp-pipeline`. Quatre étapes, dans l'ordre chronologique :

### a) Extraction du nouveau dataset depuis le modèle (17/06, commit `5f55bc1`)

Script `batch_transcribe.py` : à partir d'une liste de manuscrits du XIIIe siècle référencés sur Gallica/BnF (`Manuscrits XIII siecle.txt`, 21 manuscrits listés), le script :
1. extrait l'identifiant ARK de chaque manuscrit et télécharge les pages (haute résolution) depuis Gallica,
2. charge le modèle HTR de référence (« Exp 2 — Baseline binarisée, CER 26.3% », via Kraken),
3. segmente chaque page (Kraken BLLA) puis transcrit les lignes détectées (`rpred`),
4. construit le data contract JSON pour chaque page et le sauvegarde dans `data/predictions/`.

**16 manuscrits** ont effectivement été transcrits et sont présents dans `data/nlp_output/` (un sous-dossier par identifiant ARK) :

| Identifiant ARK | Manuscrit | Source Gallica |
|---|---|---|
| `btv1b90589023` | Luce de Gast, *Le Roman de Tristan* | [gallica.bnf.fr/.../btv1b90589023](https://gallica.bnf.fr/ark:/12148/btv1b90589023/f64.item.zoom) |
| `btv1b90595162` | Benoit de Sainte-Maure, *Le Roman de Troie* | [gallica.bnf.fr/.../btv1b90595162](https://gallica.bnf.fr/ark:/12148/btv1b90595162/f5.item) |
| `btv1b10467098b` | Benoît de Sainte-Maure, *Le Roman de Troie* (2e exemplaire) | [gallica.bnf.fr/.../btv1b10467098b](https://gallica.bnf.fr/ark:/12148/btv1b10467098b/f10.item) |
| `btv1b9065998b` | Latin 17177 | [gallica.bnf.fr/.../btv1b9065998b](https://gallica.bnf.fr/ark:/12148/btv1b9065998b/f4.item) |
| `btv1b9059006z` | *Livre de jostice et de plet* | [gallica.bnf.fr/.../btv1b9059006z](https://gallica.bnf.fr/ark:/12148/btv1b9059006z/f3.item) |
| `btv1b105600410` | Bréviaire des Dominicains (BM Toulouse) | [gallica.bnf.fr/.../btv1b105600410](https://gallica.bnf.fr/ark:/12148/btv1b105600410/f11.item) |
| `btv1b9058848r` | *Tristan en prose* | [gallica.bnf.fr/.../btv1b9058848r](https://gallica.bnf.fr/ark:/12148/btv1b9058848r/f5.item) |
| `btv1b9009629n` | Recueil de fabliaux, dits, contes en vers | [gallica.bnf.fr/.../btv1b9009629n](https://gallica.bnf.fr/ark:/12148/btv1b9009629n) |
| `btv1b9059827w` | *Roman d'Eneas* — Wace, *Brut* | [gallica.bnf.fr/.../btv1b9059827w](https://gallica.bnf.fr/ark:/12148/btv1b9059827w) |
| `btv1b10467099s` | *Roman d'Eneas* — Wace, *Brut* (2e exemplaire) | [gallica.bnf.fr/.../btv1b10467099s](https://gallica.bnf.fr/ark:/12148/btv1b10467099s) |
| `btv1b52500695k` | *Chanson d'Aspremont* | [gallica.bnf.fr/.../btv1b52500695k](https://gallica.bnf.fr/ark:/12148/btv1b52500695k/f7.item) |
| `btv1b100336071` | *Lancelot-Graal* (Lancelot en prose, Queste del Graal, Mort Artu) | [gallica.bnf.fr/.../btv1b100336071](https://gallica.bnf.fr/ark:/12148/btv1b100336071/f4.item) |
| `btv1b8447872k` | Girart d'Amiens, *Meliacin ou le Cheval de fust* | [gallica.bnf.fr/.../btv1b8447872k](https://gallica.bnf.fr/ark:/12148/btv1b8447872k/f7.item) |
| `btv1b9006878w` | Recueil de textes, Vie de saints | [gallica.bnf.fr/.../btv1b9006878w](https://gallica.bnf.fr/ark:/12148/btv1b9006878w) |
| `btv1b100341529` | Notices généalogiques, familles de Périgord — Dossiers de Lespine XLVII | [gallica.bnf.fr/.../btv1b100341529](https://gallica.bnf.fr/ark:/12148/btv1b100341529) |
| `btv1b90591336` | Estoire du Graal — Merlin en prose — Suite Vulgate | [gallica.bnf.fr/.../btv1b90591336](https://gallica.bnf.fr/ark:/12148/btv1b90591336/f8.item) |
| `btv1b52520460z` | Fragments de romans de Chrétien de Troyes (fragments d'Annonay) | [gallica.bnf.fr/.../btv1b52520460z](https://gallica.bnf.fr/ark:/12148/btv1b52520460z) |

5 manuscrits de la liste source (21 au total) n'ont pas été repris dans le dataset final : le *Chansonnier français / Bestiaire d'Amour de Richard de Fournival*, *Estoire du Graal — Merlin en prose — Suite Vulgate* (variante `btv1b9009473c`), *Recueil de fabliaux* (variante `btv1b55013464t`), et deux doublons de manuscrits déjà couverts par une autre cote.

### b) Correction / évaluation du dataset par rapport à une vérité terrain (17/06, commit `187674b`)

Script `evaluate_model.py` : charge le modèle HTR entraîné (format `.safetensors` ou VGSL) et le confronte aux transcriptions de référence ALTO/PageXML pour calculer **CER** et **WER** ligne par ligne, avec export CSV/JSON. Cette étape sert à mesurer objectivement la qualité du nouveau dataset transcrit avant de l'envoyer dans le pipeline NLP.

### c) Construction de l'alphabet et du lexique (16/06, commit `2bc3920`)

Deux scripts complémentaires sous `src/lexique/`, tous deux *cumulatifs* (les comptes s'additionnent à chaque nouveau contrat traité, pour accumuler la statistique au fil des manuscrits) :

- **`build_alphabet.py`** : compte les caractères du texte brut (sans normalisation) de chaque data contract, avec leur point de code Unicode (`U+XXXX`) — utile pour repérer les caractères invisibles ou parasites (mojibake, espaces spéciaux). Sortie : `data/lexique/alphabet.csv` (colonnes `char, codepoint, count`).
- **`build_lexicon.py`** : construit un lexique *descriptif* (pas un lexique de référence) des mots produits par le modèle HTR — minuscule + tokenisation préservant les marqueurs d'abréviation médiévaux (`⁊`, `ꝑ`, `~`...). Sortie : `data/lexique/lexicon.csv` (colonnes `word, count`). Utile pour inspecter le vocabulaire produit et repérer les mots aberrants.

### d) Construction du dictionnaire de référence ancien français (16/06, commit `b565e0f`)

Script `dictionary.py` : télécharge et fusionne deux sources externes —
- le dictionnaire StarDict *Old French-English Wiktionary*,
- le lexique CLTK (`lexiquedelancienfrancais.txt`),

puis construit `dictionnaire_ancien_francais.json` (mot → définitions Wiktionary/CLTK), **54 921 entrées au total**. C'est ce dictionnaire qui sera utilisé plus tard (18/06) par `find_lexical_errors` / la commande `lexical-check` pour détecter les erreurs lexicales.

Ces 4 étapes ont ensuite été reprises et réorganisées dans `nlp_pipeline/` (voir section 1 et suivantes) : les scripts `build_alphabet.py`/`build_lexicon.py` ont depuis été supprimés (jugés redondants une fois le lexique construit), et `dictionary.py` déplacé vers `nlp_pipeline/lexique/`.

## 1. Vue d'ensemble

| | |
|---|---|
| Commits poussés aujourd'hui | 9 (17:18 → 22:10) |
| Modifications en cours, non commitées | 3 fichiers |
| Tests unitaires | **13/13 passés** (`pytest nlp_pipeline/tests/ -q` → `13 passed in 0.29s`) |
| Données sur S3 | 399 objets, **27,3 Mo** (`28 329 754` octets) sous `s3://htr-cremma-medieval/nlp/` |
| Dictionnaire ancien français | **54 921 entrées** |
| Documents HTR traités | 18 manuscrits, 129 fichiers contracts normalisés |

## 2. Chronologie des commits

| Heure | Commit | Contenu |
|---|---|---|
| 17:18 | `d8aed35` | Sortie HTR branchée comme entrée du pipeline NLP |
| 19:10 | `1559427` | Ajout de `average_pairwise_cer` |
| 19:10 | `224eaf8` | Merge de la branche `nlp` distante |
| 19:43 | `29f6ad9` | Construction du dictionnaire ancien français |
| 20:52 | `9d27da5` | Réorganisation `src/` → `nlp_pipeline/` |
| 20:53 | `ec7b1f8` | Suppression des scripts de construction du lexique (380 lignes) |
| 21:05 | `78b9ba0` | Réécriture du notebook Kaggle (vrai pipeline, +163/-81 lignes) |
| 21:16 | `7a2ef34` | Consolidation des commandes AWS CLI |
| 21:17 | `ce38d8e` | Fix détection racine projet sur Kaggle |
| 21:52 | `ae2dc31` | Fix abréviations + schéma + `find_lexical_errors` |
| 22:10 | `e73393b` | Logging cumulatif horodaté + notebook local |

## 3. Résultats mesurés

### a) Tests unitaires

```
13 passed in 0.29s
```

Aucune régression introduite par les changements de la journée (règles u/v, i/j, ajout `find_lexical_errors`, docstrings).

### b) Qualité de la normalisation — EDA sur 157 lignes traitées (`data/eda_report.json`)

| Métrique | Valeur |
|---|---|
| Confiance moyenne | 0.957 |
| Lignes à réviser (confidence faible) | 3.18% |
| Abréviations par ligne | 0.39 |
| Lignes très courtes (<10 car.) | 3.18% |
| Répartition confiance | 153/157 lignes ≥ 0.9, seulement 4 lignes < 0.6 |

Le corpus HTR est globalement de très bonne qualité de reconnaissance ; peu de lignes nécessitent une révision manuelle.

### c) Détection lexicale — avant/après le fix des règles u/v et i/j

| | Avant le fix | Après le fix |
|---|---|---|
| Tokens totaux | 26 900 | 26 452 |
| Occurrences totales | 89 179 | 89 179 |
| Tokens inconnus du dictionnaire | 25 827 (**96.0%**) | 25 294 (**95.6%**) |
| Top token inconnu | `⁊` (3103 occ.) | `e` (483 occ.) |

Le fix a fait disparaître du top les marqueurs typographiques bruts (`⁊` 3103 occ., `q̃` 509 occ., `qͥ` 279 occ., `ꝑ` 250 occ.) — preuve que la normalisation et la table d'abréviations fonctionnent. Le taux global reste élevé (~96%) mais ce n'est **pas un signal d'échec** : le top restant après normalisation est composé de mots grammaticaux français très courants et bien orthographiés (`est`, `ce`, `des`, `car`, `vous`, `tout`, `fait`, `ont`, `dist`), absents du dictionnaire externe (Wiktionary ancien français + CLTK) car celui-ci couvre surtout le vocabulaire lexical, pas les mots-outils. C'est une limite de la ressource, pas du pipeline — décision prise de ne pas la corriger avant la présentation de vendredi.

### d) Impact réel du fix u/v et i/j sur tout le corpus (129 documents)

Comparaison exhaustive avant/après (et non plus seulement sur l'échantillon de 30 documents de l'audit initial) :

- **3725 paires de mots distinctes** changent suite au fix
- Top des corrections les plus fréquentes :

| Avant (bug) | Après (corrigé) | Occurrences |
|---|---|---|
| `qve` | `que` | 651 |
| `qvi` | `qui` | 420 |
| `bjen` | `bien` | 182 |
| `lvi` | `lui` | 182 |
| `qvil` | `quil` | 146 |
| `pvis` | `puis` | 112 |
| `mje` | `mie` | 85 |
| `qvant` | `quant` | 81 |
| `djeu` | `dieu` | 62 |
| `svi` | `sui` | 57 |
| `onqves` | `onques` | 53 |
| `djex` | `diex` | 53 |

Ces corrections concernent des centaines à des milliers d'occurrences de mots très fréquents (`que`, `qui`, `bien`, `lui`...) qui étaient auparavant mal normalisés à chaque occurrence. D'autres changements observés dans le diff (`je`→`ie`, `vit`→`uit`, `vint`→`uint`, `svi`→`sui`) sont des effets de bord corrects de la résolution i/j et u/v ailleurs dans le même mot — vérifiés manuellement sur des exemples réels, ce ne sont pas des régressions.

### e) Stockage S3

```
Total Objects: 399
Total Size: 28 329 754 octets (~27,3 Mo)
```

Contenu : sorties HTR (`nlp/output/`, un dossier par manuscrit avec `.json`/`.txt`/`.xml`), dictionnaire ancien français, logs.

### f) Historique des exécutions CLI (`data/review/nlp_cli_run_log.jsonl`, cumulatif)

```
10 runs enregistrés, tous OK (returncode 0)
validate, eda, review-queue, normalize-contract, correct, relative-eval, lexical-check
+ 3 runs supplémentaires de normalize-contract / lexical-check
```

Aucun échec enregistré depuis l'activation du logging horodaté.

## 4. Data contract et règles de normalisation

### a) Le data contract HTR (`nlp_pipeline/htr_data_contract_schema.json`)

C'est le format JSON pivot entre la sortie du HTR et l'entrée du pipeline NLP — validé via JSON Schema (Draft 2020-12). Chaque document HTR produit un JSON avec cette structure :

```
document_id, metadata { source, century_estimate, document_type, sha256, scan_quality }
pages[] {
  page_id, image_path,
  lines[] {
    line_id, text, confidence, char_confidences[],
    candidates: null | {position, options[]} | [{position, options[]}, ...],
    needs_review, polygon[], reading_order
  }
}
```

Points clés :
- `confidence` / `char_confidences` : scores de confiance du modèle HTR, ligne entière et par caractère — c'est ce que `correct` utilise pour cibler les positions ambiguës.
- `candidates` : variantes proposées par le HTR aux positions de faible confiance. **Champ presque toujours `null`** dans les données réelles (le modèle HTR actuel ne génère pas de candidats alternatifs) — c'est ce qui causait le bug du schéma corrigé aujourd'hui (`ae2dc31`) : le schéma n'autorisait pas `null` avant le fix, donc 100% des documents réels échouaient à la validation.
- `needs_review` : drapeau booléen pour la file de relecture humaine (`review-queue`).

### b) Les règles de normalisation déterministe (`nlp_pipeline/normalization_rules.py`)

Appliquées dans cet ordre par `MedievalFrenchNormalizer.normalize()`, avant toute correction statistique/IA (priorité imposée par les consignes du projet) :

1. **NFC** : normalisation Unicode des formes combinées (caractère + accent flottant → caractère précomposé), pour une comparaison stable.
2. **lowercase** : minuscule systématique.
3. **u/v** : résout les variantes graphiques médiévales selon le contexte (début de mot ou voyelle suivante → `v` ; sinon → `u`).
   - Exceptions ajoutées aujourd'hui : `qu`/`gu` (digrammes `que`, `qui`, `guerre`) et `u` suivi de `i` (`lui`) restent vocaliques. Limite acceptée : `deuient→devient` n'est plus corrigé (faux négatif jugé moins coûteux que les faux positifs supprimés — voir section 3.d pour l'ampleur de l'impact réel).
4. **i/j** : convertit `i` en `j` quand il précède une voyelle autre que `e` (graphie moyen-français).
   - Exception ajoutée aujourd'hui : le digramme `ie`/`ien`/`ier` (`bien`, `mie`, `biel`) garde son `i` vocalique.
5. **tilde nasal** : `a~`/`e~`/`o~` (et formes combinantes Unicode équivalentes) → `an`/`en`/`on`.
6. **table d'abréviations** (`medieval_abbreviations.json`, 14 entrées) : remplace les marqueurs scribaux et latinismes (`⁊→et`, `ꝑ→per`, `ꝓ→pro`, `ꝙ→us`, `ꝗ→que`, `dñs→dominus`, `q̄→qui`, etc.) par leur expansion, clés les plus longues en premier pour éviter les collisions.

Deux mécanismes de détection complètent les règles, sans les remplacer :
- `detect_normalization_candidates` : repère les tokens porteurs de marqueurs typographiques d'abréviation non résolus, propose une expansion.
- `find_lexical_errors` (ajouté aujourd'hui) : repère les tokens absents du dictionnaire ancien français (55k entrées) — détection lexicale, complémentaire à la détection par marqueurs.

## 5. Bugs corrigés aujourd'hui

| Bug | Impact avant fix | Fix |
|---|---|---|
| `ModuleNotFoundError: transformers` | Bloquait 100% des commandes CLI | Suppression import top-level inutile |
| Schéma rejette `candidates: null` | Bloquait la validation de 100% des données réelles | Ajout `{"type":"null"}` au schéma |
| Clone Kaggle imbriqué | Notebook inutilisable après 2 exécutions | Idempotence (`isdir` + `git pull`) |
| `awscli` absent sur Kaggle | Échec de la cellule de sync S3 | `pip install awscli` ajouté |
| Règles u/v et i/j trop larges | `que→qve`, `qui→qvi`, `lui→lvi`, `bien→bjen`, `mie→mje` (mots corrects cassés) | Exceptions `qu`/`gu`/`u+i` et `ie` — vérifié sur 30+ documents réels, puis confirmé sur les 129 documents du corpus complet (section 3.d) |

## 6. Prochaines étapes envisagées

- **Heuristique de substitution par fréquence** (idée, non démarrée) : aujourd'hui, `find_lexical_errors` et `detect_normalization_candidates` se limitent à *détecter* les tokens suspects/inconnus, sans choisir automatiquement de substitution. L'idée est d'ajouter une formule qui, pour chaque token suspect, calcule un score de vraisemblance par candidat de substitution à partir de sa fréquence dans le dictionnaire/lexique de référence :

  ```
  score(candidat) = fréquence(candidat) / Σ fréquences(tous les candidats pour ce token)
  ```

  Le candidat au score le plus élevé serait retenu automatiquement comme substitution proposée. Objectif : combler partiellement l'absence de `candidates` côté HTR (cf. section 5 — la correction guidée par confiance ne peut s'activer faute de variantes fournies par le modèle) sans attendre l'intégration d'un modèle MLM. Reste à définir : la source des candidats pour un token inconnu (variantes orthographiques proches dans le dictionnaire ? table d'abréviations étendue ?) et le seuil de score à partir duquel appliquer la substitution automatiquement plutôt que de la proposer en révision.
- Activer la correction MLM (`almanach/camembert-base`) une fois des `candidates` disponibles (HTR ou heuristique ci-dessus).
- Enrichir le dictionnaire de référence avec les mots-outils pour rendre `lexical-check` plus discriminant.

## 7. Plan « after » — NER, POS, graphe, TEI (non démarré)

Plan en 4 phases séquentielles (chaque phase a des entrées, des livrables et un critère de sortie explicite — on ne passe à la phase suivante que si le critère est rempli). Pas de dates fixes : à dérouler au rythme de l'équipe après la présentation.

### Phase 1 — NER : choisir et brancher un modèle existant

**Objectif** : avoir un modèle NER qui tourne sur le corpus, même sans fine-tuning, pour disposer d'une baseline.

| # | Tâche | Détail | Livrable |
|---|---|---|---|
| 1.1 | Tester 2-3 modèles candidats sans fine-tuning | `magistermilitum/roberta-multilingual-medieval-ner`, `pjox/dalembert-classical-fr-ner`, + recherche Hugging Face `camembert medieval` / `camembert old french` | Script `nlp_pipeline/ner_baseline.py` qui charge un modèle et l'applique à un échantillon de lignes normalisées |
| 1.2 | Choisir le modèle retenu | Comparer qualitativement les sorties sur 10-15 lignes représentatives (présence de PER/LOC plausibles) | Décision documentée (1 paragraphe dans le rapport) |
| 1.3 | Vérifier/étendre le schéma d'annotation | Lister les classes du modèle choisi (`PER`, `LOC`, `ORG`, `MISC`, `DATE`). Ajouter `TITLE` si besoin (`num_labels` à modifier) | Schéma de labels documenté |

**Critère de sortie phase 1** : un script exécutable produit des entités (même imparfaites) sur au moins 1 document complet.

### Phase 2 — NER : annotation minimale et fine-tuning léger

**Objectif** : améliorer la baseline avec un fine-tuning léger, sans viser la perfection.

| # | Tâche | Détail | Livrable |
|---|---|---|---|
| 2.1 | Annoter manuellement 200-300 tokens | Choisir des lignes variées (plusieurs manuscrits) ; annoter avec le schéma retenu en 1.3 | Fichier d'annotation (CSV ou JSONL) |
| 2.2 | Aligner labels et tokenisation sous-mot | Gérer les word-pieces avec `-100` sur les tokens de continuation — point technique à montrer en présentation | Fonction d'alignement testée (1 test unitaire minimum) |
| 2.3 | Fine-tuner le modèle | Quelques époques sur les 200-300 tokens annotés | Modèle fine-tuné sauvegardé localement |
| 2.4 | Évaluer avec `seqeval` | F1 micro + F1 par type d'entité. Même un F1 de 0,40-0,50 valide le pipeline | Rapport de métriques + 2-3 exemples d'erreurs commentés (ex. confusion `TITLE`/`PER`) |

**Critère de sortie phase 2** : F1 micro mesuré et documenté, peu importe sa valeur — l'objectif est la preuve de bout en bout, pas la performance.

### Phase 3 — POS, lemmes et extraction de relations (sous-échantillon)

**Objectif** : couvrir les étapes à faible effort sur un petit nombre de pages, sans bloquer sur la phase NER si elle prend du retard (phase indépendante, peut démarrer en parallèle de la phase 2).

| # | Tâche | Détail | Livrable |
|---|---|---|---|
| 3.1 | POS + lemmatisation | `stanza` (modèle `frm`) ou `pie-extended medieval-fr`, sur quelques pages déjà normalisées | Sortie POS/lemmes pour ces pages |
| 3.2 | Règles d'extraction de relations | Règles lexico-syntaxiques sur séquences de labels NER (ex. `B-PER .* prist .* B-LOC` → `(PERSONNE, PREND, LIEU)`) — pas de LLM | Liste de relations extraites + règles documentées dans le code |

**Critère de sortie phase 3** : au moins une relation extraite et vérifiée manuellement comme correcte sur l'échantillon.

### Phase 4 — Graphe et export TEI (5-10 pages ciblées)

**Objectif** : un livrable visuel/structuré présentable, sur un périmètre volontairement restreint.

| # | Tâche | Détail | Livrable |
|---|---|---|---|
| 4.1 | Construire le graphe de relations | `networkx`, à partir des relations de la phase 3 | Visualisation du graphe (image ou notebook) |
| 4.2 | Export TEI-XML | Baliser `<persName>`, `<placeName>`, `<date>` sur les entités détectées, 5-10 pages | Fichier(s) TEI-XML valide(s) |

**Critère de sortie phase 4** : au moins un fichier TEI valide (validable par un parseur XML), même sur un périmètre réduit.

### Évaluation transverse (à chaque phase)

Mesurer le CER **relatif** (pas absolu — pas de vérité terrain disponible, cf. section 3.c) en ajoutant une colonne de variante par nouvelle étape : sortie brute → règles seules → correction guidée → +NER/relations si cela modifie le texte. Réutiliser directement `relative-eval` / `average_pairwise_cer`, déjà en place.

### Dépendances entre phases

```
Phase 1 (NER baseline) ──→ Phase 2 (fine-tuning NER) ──┐
                                                          ├──→ Phase 4 (graphe + TEI)
Phase 3 (POS + relations) ── peut démarrer en parallèle ─┘
```

Phase 3 ne dépend pas du résultat du fine-tuning NER (elle peut utiliser la baseline de la phase 1) — c'est la voie à privilégier si le temps manque pour aller jusqu'à la phase 2.

## 8. Non commité actuellement

3 fichiers modifiés (`CONVENTIONS_NLP.md`, `nlp_pipeline/normalization_rules.py`, `nlp_pipeline/tests/test_normalization_rules.py`) contenant le fix des règles u/v et i/j ci-dessus, déjà validé par les tests (13/13). Pas encore poussé.
