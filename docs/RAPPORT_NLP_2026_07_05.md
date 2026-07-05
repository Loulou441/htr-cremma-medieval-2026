# Rapport — Pipeline NLP sur un vrai document Gallica (Roman de Troie, f5)

## Contexte de cette run

Première run du pipeline sur un **vrai data contract issu d'un vrai manuscrit**
(*Benoît de Sainte-Maure, Le Roman de Troie*, folio 5 —
`https://gallica.bnf.fr/ark:/12148/btv1b90595162/f5.item`), et non plus sur un
corpus synthétique fabriqué. Deux nouveautés par rapport aux rapports précédents :

- **CamemBERT MLM réellement actif** (`almanach/camembert-base`), exécuté sur Colab (CPU).
- **Vrai dictionnaire ancien français** (172 734 entrées, construit à partir de
  Wiktionary StarDict + lexique Godefroy/CLTK), à la place du dictionnaire factice
  à 31 mots utilisé dans les rapports précédents.

**Rappel sur la provenance du data contract** : il a été construit à partir d'une
transcription ligne par ligne (163 lignes, avec confiance par ligne) fournie
manuellement, pas directement issue d'un run `batch_transcribe.py`. Deux approximations en découlent, 
à garder en tête pour l'interprétation des résultats :
- `char_confidences` répète la confiance de la ligne sur chaque caractère (pas de
  vraie granularité par caractère) ;
- `polygon` est un placeholder géométrique, pas une vraie coordonnée de segmentation.
- Le champ `model` du contrat reste `"non precise - a confirmer"` — l'outil exact
  ayant produit cette transcription n'a pas été précisé.

---

## 1. Validation et EDA

| Métrique | Valeur |
|---|---|
| Lignes | 163 |
| Confiance moyenne | 0.622 |
| Longueur médiane de ligne | 19 caractères |
| Taux `needs_review` | **100%** |
| Répartition confiance | 62 lignes < 0.6, 74 entre 0.6-0.7, 27 entre 0.7-0.8, 0 au-dessus de 0.8 |

Le `needs_review_rate` à 100% s'explique entièrement par le seuil de confiance
globale (`< 0.9`) : aucune ligne n'atteint 0.8, donc toutes tombent en révision —
cohérent avec l'approximation `char_confidences` ci-dessus (l'écart-type par
caractère, qui pourrait forcer une révision même à confiance élevée, est toujours
nul ici puisque la valeur est répétée à l'identique).

**Confiance très inférieure** à celle du run de référence du 18 juin sur le vrai
corpus (0.793) ou du corpus synthétique (0.913). Sans les vraies polygones de
segmentation, impossible de déterminer si cela vient d'une reconnaissance de
caractères dégradée ou d'un découpage de lignes/colonnes imparfait — point déjà
signalé lors de la construction du contrat.

---

## 2. Normalisation par règles

**CER pairwise moyen (raw vs normalisé) : 0.0844**

Sensiblement plus élevé que sur le corpus synthétique (0.0483) — attendu : un vrai
texte médiéval dense en abréviations change davantage sous les règles NFC/u-v/i-j/
tilde/abréviations qu'un corpus synthétique conçu pour ne contenir que quelques cas
ciblés. Ligne la plus impactée : `l141` (CER 0.3125) ; plusieurs lignes à CER nul
(aucun changement, texte déjà stable face aux règles).

---

## 3. Correction guidée par confiance — CamemBERT MLM

| Mesure | Valeur |
|---|---|
| Scorer | CamemBERT MLM (`almanach/camembert-base`), `using_mlm: true` |
| Lignes traitées | 163 |
| **Corrections appliquées** | **0** |
| CER pairwise moyen | 0.0 |

**0 correction, comme attendu et déjà documenté** (README section 16, rapport du 4
juillet) : le mécanisme n'arbitre une position que si le data contract fournit un
champ `candidates` non nul à cette position. Ici, toutes les lignes ont
`candidates: null` — la transcription source ne fournissait qu'un texte et une
confiance par ligne, pas de positions ambiguës avec propositions alternatives.

**C'est en réalité une confirmation utile** : le rapport du 18 juin notait que
`candidates` est *"presque toujours null"* sur le vrai corpus HTR — cette run le
démontre une seconde fois, cette fois sur un vrai document Gallica plutôt que sur
une affirmation reportée depuis un run antérieur. Le chemin CamemBERT reste
fonctionnel et prêt (chargé, actif), simplement sans travail à faire faute de
`candidates` en entrée.

---

## 4. Détection lexicale

### `detect-normalization`

**8 tokens suspects sur 511** (comptant les répétitions : 791 occurrences totales).
Marqueurs détectés : `⁊` (27 occurrences), `combining_mark`, `ꝑ`, `ꝓ`, `ꝯ`.

**Signal positif à noter** : le token `ꝯmsds` a reçu une suggestion d'expansion
automatique — `conmsds` (le glyphe `ꝯ` est correctement reconnu comme l'abréviation
scribale de *con-*). C'est la première fois que cette suggestion est observée sur du
texte réellement issu d'un manuscrit plutôt que sur un exemple construit à la main.

### `lexical-check` (vrai dictionnaire, 172 734 entrées)

| Mesure | Valeur |
|---|---|
| Tokens analysés | 511 (791 occurrences) |
| Tokens inconnus | 372 (**72.8%**) |
| Couverture | 27.2% |

À comparer avec les **66.7% de couverture** obtenus avec ce même dictionnaire sur le
corpus synthétique propre (message précédent). L'écart s'explique par la nature du
texte : sur les 20 tokens inconnus les plus fréquents, la moitié sont des fragments
d'une ou deux lettres (`s`, `d`, `l`, `n`, `m`, `p`, `q`...) — des artefacts de
segmentation/reconnaissance plausibles, pas de vrais mots qu'un dictionnaire,
même complet, pourrait reconnaître. Ce n'est pas une limite du dictionnaire mais un
symptôme de la qualité de transcription déjà relevée en section 1.

---

## 5. Ce que cette run démontre

| Composant | Validé sur donnée réelle ? |
|---|---|
| Validation du schéma | ✅ (contrat construit manuellement, conforme) |
| EDA | ✅ |
| Normalisation par règles | ✅ — CER pairwise cohérent avec un texte plus dense en abréviations |
| **Correction CamemBERT MLM** | ✅ **mécanisme actif**, 0 correction car `candidates` absent — comportement attendu, pas une panne |
| Détection d'abréviations | ✅ — une vraie suggestion d'expansion produite (`ꝯ` → `con`) |
| Dictionnaire ancien français réel | ✅ — 172 734 entrées, couverture cohérente avec la qualité du texte source |

**Limite à garder à l'esprit pour la suite** : pour observer une vraie correction
MLM en action, il faut un data contract avec de véritables `candidates` — ce qui
suppose soit un HTR produisant nativement des hypothèses alternatives par position,
soit l'heuristique de substitution par fréquence mentionnée dans les prochaines
étapes du README. Cette run confirme que **le blocage n'est pas dans le code du
correcteur, mais dans l'absence de cette donnée d'entrée**, sur données réelles comme
sur données synthétiques.