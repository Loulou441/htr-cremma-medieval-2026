"""Garde-fou sur les chemins par defaut du CLI.

Bug reel corrige : DEFAULT_SCHEMA / DEFAULT_ABBR dans nlp_cli.py pointaient
vers nlp_pipeline/*.json alors que ces fichiers avaient ete deplaces dans
nlp_pipeline/json_files/. Consequence : `validate`, `normalize`,
`normalize-contract`, `ablation` et `correct` plantaient (FileNotFoundError)
des qu'ils etaient appeles sans --schema/--abbreviations explicite, c'est-a-
dire dans leur usage le plus courant.

Ce test ne verifie pas le contenu des fichiers (deja couvert ailleurs), 
seulement que les constantes du CLI restent alignees avec l'emplacement reel
des fichiers sur le disque, quel que soit cet emplacement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import nlp_cli


def test_default_schema_path_exists():
    assert Path(nlp_cli.DEFAULT_SCHEMA).is_file(), (
        f"DEFAULT_SCHEMA pointe vers un fichier inexistant : {nlp_cli.DEFAULT_SCHEMA}"
    )


def test_default_abbreviations_path_exists():
    assert Path(nlp_cli.DEFAULT_ABBR).is_file(), (
        f"DEFAULT_ABBR pointe vers un fichier inexistant : {nlp_cli.DEFAULT_ABBR}"
    )