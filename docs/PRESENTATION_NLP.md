# Pipeline NLP — Normalisation de manuscrits médiévaux français

*Résultats sur le corpus complet (129 documents, 16 336 lignes) — 18 juin 2026*

## 1. Le pipeline en un coup d'œil

```
Manuscrits Gallica  →  Extraction/transcription (modèle HTR)  →  Évaluation CER/WER
        ↓                                                              ↓
Alphabet + lexique (diagnostic)              Dictionnaire ancien français (référence)
                                                              ↓
Sortie HTR (JSON)  →  Validation (data contract)  →  Normalisation par règles
                                                              ↓
                                          Détection lexicale + Évaluation relative
```

- **Constitution du dataset** (14-17 juin) : extraction de 16 manuscrits du XIIIe siècle depuis Gallica/BnF, transcription par le modèle HTR, évaluation CER/WER contre une vérité terrain, construction d'un alphabet et d'un lexique diagnostiques, construction d'un dictionnaire de référence ancien français (55k entrées).
- **Entrée du pipeline NLP** : sortie brute du modèle HTR, un JSON par page (texte, confiance par caractère, polygones).
- **Normalisation** : 6 règles déterministes (priorité fixée par les consignes du projet : règles avant IA).
- **Sortie** : texte normalisé + rapports de qualité (lexical, CER relatif).

## 1bis. Constitution du dataset — étapes préalables (14-17 juin)

| Étape | Ce qui a été fait |
|---|---|
| **Extraction depuis le modèle** | 16 manuscrits médiévaux téléchargés depuis Gallica/BnF, segmentés et transcrits par le modèle HTR (Kraken, baseline binarisée CER 26.3%) |
| **Évaluation / correction du dataset** | Confrontation des transcriptions à une vérité terrain ALTO/PageXML pour calculer CER et WER ligne par ligne |
| **Alphabet** | Comptage des caractères bruts par data contract (avec point de code Unicode) pour repérer les caractères parasites |
| **Lexique** | Liste des mots produits par le modèle avec fréquences, pour inspecter le vocabulaire et repérer les anomalies |
| **Dictionnaire de référence** | Fusion Wiktionary (ancien français) + CLTK → 54 921 entrées, utilisées ensuite pour la détection lexicale (`lexical-check`) |

### Sources Gallica/BnF des 16 manuscrits transcrits

*Luce de Gast*, Le Roman de Tristan · Benoit de Sainte-Maure, Le Roman de Troie (2 exemplaires) · Latin 17177 · Livre de jostice et de plet · Bréviaire des Dominicains (BM Toulouse) · Tristan en prose · Recueil de fabliaux, dits, contes en vers · Roman d'Eneas — Wace, Brut (2 exemplaires) · Chanson d'Aspremont · Lancelot-Graal (Lancelot en prose, Queste del Graal, Mort Artu) · Girart d'Amiens, Meliacin ou le Cheval de fust · Recueil de textes, Vie de saints · Notices généalogiques, familles de Périgord — Dossiers de Lespine XLVII · Estoire du Graal — Merlin en prose — Suite Vulgate · Fragments de romans de Chrétien de Troyes (fragments d'Annonay).

Liste complète avec les liens Gallica et les identifiants ARK : voir section 0.a de [RAPPORT_NLP_2026-06-18.md](RAPPORT_NLP_2026-06-18.md).

## 2. Le data contract

Chaque document HTR est un JSON validé par un schéma strict (`htr_data_contract_schema.json`) :

| Champ | Rôle |
|---|---|
| `text` | transcription brute du HTR |
| `confidence`, `char_confidences` | scores de confiance globaux et par caractère |
| `candidates` | variantes alternatives proposées par le HTR (souvent `null` sur ce corpus) |
| `needs_review` | drapeau pour la file de relecture humaine |

**129/129 documents valides** contre le schéma.

## 3. Les règles de normalisation

Appliquées dans l'ordre, avant toute correction statistique :

1. **NFC** — formes Unicode combinées → précomposées
2. **lowercase**
3. **u/v** — résolution contextuelle des variantes graphiques médiévales (`auant`→`avant`, `cheualier`→`chevalier`), avec exceptions pour les digrammes `qu`/`gu` et `u+i`
4. **i/j** — `i` consonantique devant voyelle (sauf digramme `ie`)
5. **tilde nasal** — `a~/e~/o~` → `an/en/on`
6. **table d'abréviations** — 14 marqueurs scribaux et latinismes (`⁊→et`, `ꝑ→per`, `dñs→dominus`...)

### Impact mesuré sur tout le corpus

| Avant | Après | Occurrences |
|---|---|---|
| `qve` | `que` | 651 |
| `qvi` | `qui` | 420 |
| `bjen` | `bien` | 182 |
| `lvi` | `lui` | 182 |
| `qvil` | `quil` | 146 |
| `pvis` | `puis` | 112 |

→ **3725 paires de mots distinctes** corrigées par les règles sur le corpus complet.

## 4. Résultats chiffrés (corpus complet, run du 18/06)

| Mesure | Valeur |
|---|---|
| Documents validés | 129 / 129 (100%) |
| Lignes analysées (EDA) | 16 336 |
| Confiance HTR moyenne | 0.793 |
| Lignes signalées pour révision | 36.8% |
| Tests unitaires | 13 / 13 |
| CER pairwise moyen (raw / normalisé / corrigé) | 0.0667 |
| Tokens couverts par le dictionnaire ancien français (55k entrées) | 4.4% |

## 5. Limites connues, à mentionner à l'oral

- **Détection lexicale (4.4% de couverture)** : le dictionnaire externe (Wiktionary ancien français + CLTK) couvre le vocabulaire lexical mais pas les mots-outils très fréquents (`est`, `ce`, `vous`, `tout`...). Le taux bas reflète une limite de la ressource, pas un échec de normalisation — la preuve : les marqueurs typographiques bruts (`⁊`, `q̃`, `ꝑ`) ont disparu du classement après normalisation.
- **Correction guidée par confiance/MLM** : le scorer CamemBERT (mode MLM, `almanach/camembert-base`) est désormais **activé par défaut** dans `nlp_cli.py correct` (auparavant codé mais non branché par défaut). La limite réelle qui subsiste : `candidates` est `null` sur la quasi-totalité des données réelles de ce corpus, donc même avec le MLM actif, le correcteur n'a aucune variante à arbitrer et **0 correction est appliquée** sur ce run. Le CER pairwise de l'étape `correct` est donc proche de 0 (texte inchangé), ce qui est attendu et documenté, pas un bug.
- **Réinjection** : `needs_review` est désormais recalculé après correction (passe à `false` si toutes les positions ambiguës connues ont été résolues et que la confiance reste bonne) — mais comme aucune correction réelle n'est appliquée sur ce run (point ci-dessus), cette réinjection n'a, pour l'instant, aucun effet observable sur le corpus actuel non plus.
- **Règle u/v** : exclut volontairement `u+i` (ex. `lui`) pour éviter les faux positifs, ce qui empêche `deuient→devient` d'être corrigé — compromis documenté et accepté.

## 6. Prochaines étapes possibles

- Générer de vraies entrées `candidates` côté HTR ou via l'heuristique de substitution par fréquence ci-dessous, pour que le scorer CamemBERT (déjà actif par défaut) ait effectivement des variantes à arbitrer.
- Enrichir le dictionnaire de référence avec les mots-outils pour rendre `lexical-check` plus discriminant.
- **Heuristique de substitution par fréquence** : pour chaque token jugé inconnu/suspect (via `find_lexical_errors` ou `detect_normalization_candidates`), calculer un score de vraisemblance pour chaque substitution candidate à partir de sa fréquence d'apparition dans le dictionnaire/lexique de référence (ex. `score(candidat) = fréquence(candidat) / Σ fréquences(tous les candidats)`), et retenir automatiquement la substitution la plus probable plutôt que de s'arrêter à une simple détection. Vise à pallier l'absence de `candidates` côté HTR (section 5) maintenant que le MLM est prêt à les exploiter.

## 7. Plan « after » (post-présentation) — NER, POS, graphe, TEI

Rien n'est démarré sur ces étapes. Plan en 4 phases séquentielles, chacune avec un critère de sortie clair — pas de dates fixes, à dérouler au rythme de l'équipe.

| Phase | Objectif | Tâches clés | Critère de sortie |
|---|---|---|---|
| **1. NER baseline** | Avoir un modèle NER qui tourne, sans fine-tuning | Tester `magistermilitum/roberta-multilingual-medieval-ner` et `pjox/dalembert-classical-fr-ner` sur un échantillon ; choisir le meilleur ; vérifier/étendre le schéma de labels (+ `TITLE` si besoin) | Entités produites sur ≥1 document complet |
| **2. NER fine-tuning** | Améliorer la baseline | Annoter 200-300 tokens à la main ; aligner labels/word-pieces (`-100`) ; fine-tuner ; évaluer avec `seqeval` (F1 micro + par type) | F1 mesuré et documenté, même si bas (0,40-0,50 acceptable) |
| **3. POS + relations** (parallélisable avec la phase 2) | Couvrir POS/lemmes et relations à faible effort | `stanza` (modèle `frm`) ou `pie-extended medieval-fr` sur quelques pages ; règles lexico-syntaxiques sur les labels NER (ex. `B-PER .* prist .* B-LOC` → PERSONNE-PREND-LIEU), pas de LLM | ≥1 relation extraite et vérifiée correcte |
| **4. Graphe + TEI** | Livrable visuel/structuré sur périmètre réduit | Graphe `networkx` à partir des relations ; export TEI-XML (`<persName>`, `<placeName>`, `<date>`) sur 5-10 pages | ≥1 fichier TEI valide |

**Dépendances** : Phase 1 → Phase 2 → Phase 4. Phase 3 ne dépend que de la Phase 1 (peut utiliser la baseline sans attendre le fine-tuning) — c'est la voie à privilégier si le temps manque pour aller jusqu'à la phase 2.

**Évaluation à chaque phase** : CER **relatif** (comme aujourd'hui avec `relative-eval`), pas de CER absolu — toujours pas de vérité terrain disponible.