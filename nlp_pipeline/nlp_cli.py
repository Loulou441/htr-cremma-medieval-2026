"""CLI to execute NLP workflow"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from cer_utils import average_pairwise_cer, cer
from confidence_correction import ConfidenceGuidedCorrector
from htr_data_contract import (
    compute_eda,
    export_review_csv,
    load_json,
    save_json,
    seal_test_set,
    split_review_buckets,
    stratified_split_records,
    validate_contract,
)
from normalization_rules import (
    MedievalFrenchNormalizer,
    detect_normalization_candidates,
    find_lexical_errors,
)


DEFAULT_SCHEMA = str(Path(__file__).parent / "json_files" / "htr_data_contract_schema.json")
DEFAULT_ABBR = str(Path(__file__).parent / "json_files" / "medieval_abbreviations.json")
DEFAULT_DICTIONARY = str(Path(__file__).parent / "resources" / "dictionnaire_ancien_francais.json")
RUN_LOG_PATH = str(Path(__file__).parent / "data" / "review" / "nlp_cli_run_log.jsonl")


def log_run(command: str, args: argparse.Namespace, returncode: int) -> None:
    path = Path(RUN_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "returncode": returncode,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_input_paths(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    path_str = str(input_path)

    if any(ch in path_str for ch in "*?[" ):
        matches = sorted(glob.glob(path_str, recursive=True))
        return [Path(match) for match in matches if Path(match).is_file()]

    if input_path.is_dir():
        return sorted([path for path in input_path.rglob("*.json") if path.is_file()])

    if input_path.is_file():
        return [input_path]

    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_contracts(input_path: str | Path) -> list[dict]:
    paths = resolve_input_paths(input_path)
    if not paths:
        raise FileNotFoundError(f"No JSON contracts found at: {input_path}")

    contracts: list[dict] = []
    for path in paths:
        contracts.append(load_json(path))
    return contracts


def merge_contracts(contracts: list[dict]) -> dict:
    pages: list[dict] = []
    for contract in contracts:
        pages.extend(contract.get("pages", []))
    return {"pages": pages}


def compute_eda_from_contracts(contracts: list[dict]) -> dict:
    merged_contract = merge_contracts(contracts)
    return compute_eda(merged_contract)


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        paths = resolve_input_paths(args.input)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not paths:
        print(f"[ERROR] no JSON files found for input: {args.input}")
        return 1

    schema = load_json(args.schema)
    any_errors = False
    for path in paths:
        contract = load_json(path)
        errors = validate_contract(contract, schema)
        if errors:
            any_errors = True
            print(f"[ERROR] {path}")
            for e in errors:
                print(f"  {e}")
        else:
            print(f"OK: {path}")

    if any_errors:
        return 1

    print("Validation OK: all contracts are valid.")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    try:
        contracts = load_contracts(args.input)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    metrics = compute_eda_from_contracts(contracts)
    output = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def cmd_review_queue(args: argparse.Namespace) -> int:
    try:
        contracts = load_contracts(args.input)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    contract = merge_contracts(contracts)
    buckets = split_review_buckets(
        contract,
        direct_threshold=args.direct_threshold,
        exclude_threshold=args.exclude_threshold,
        char_std_threshold=args.char_std_threshold,
    )

    export_review_csv(buckets["review"], args.csv_output)
    Path(args.json_output).write_text(
        json.dumps(buckets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Direct : {len(buckets['direct'])}")
    print(f"Review : {len(buckets['review'])}")
    print(f"Exclude: {len(buckets['exclude'])}")
    print(f"CSV review queue: {args.csv_output}")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    normalizer = MedievalFrenchNormalizer.from_json(args.abbreviations)

    if args.text:
        print(normalizer.normalize(args.text))
        return 0

    rows = []
    with open(args.csv_input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row.get(args.source_col, "")
            row[args.output_col] = normalizer.normalize(source)
            rows.append(row)

    with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [args.source_col, args.output_col])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Normalized CSV written to: {args.csv_output}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    rows = []
    with open(args.csv_input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No rows in CSV.")
        return 1

    all_on = MedievalFrenchNormalizer.from_json(args.abbreviations)

    before = 0.0
    after = 0.0
    for row in rows[: args.limit]:
        ref = row[args.reference_col]
        src = row[args.hypothesis_col]
        before += cer(ref, src)
        after += cer(ref, all_on.normalize(src))

    n = min(len(rows), args.limit)
    before /= n
    after /= n

    print(f"CER before: {before:.4f}")
    print(f"CER after : {after:.4f}")
    print(f"Gain      : {before - after:.4f}")
    return 0


def cmd_relative_eval(args: argparse.Namespace) -> int:
    rows = []
    with open(args.csv_input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No rows in CSV.")
        return 1

    variant_cols = [col.strip() for col in args.variant_cols.split(",") if col.strip()]
    if not variant_cols:
        print("No variant columns specified.")
        return 1

    per_row_scores = []
    for row in rows[: args.limit]:
        variants = [row.get(col, "") for col in variant_cols]
        if any(v == "" for v in variants):
            continue
        per_row_scores.append(average_pairwise_cer(variants))

    if not per_row_scores:
        print("No valid rows with all variant columns present.")
        return 1

    avg_score = sum(per_row_scores) / len(per_row_scores)
    print(f"Average pairwise CER across variants: {avg_score:.4f}")
    print(f"Rows evaluated: {len(per_row_scores)}")
    return 0


def cmd_correct(args: argparse.Namespace) -> int:
    try:
        paths = resolve_input_paths(args.input)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not paths:
        print(f"[ERROR] no JSON files found for input: {args.input}")
        return 1

    if len(paths) > 1 and not args.output_dir:
        print("[ERROR] When correcting multiple contracts, specify --output-dir.")
        return 1

    contracts = [load_json(path) for path in paths]
    try:
        corrector = ConfidenceGuidedCorrector(
            threshold=args.threshold,
            mlm_model=args.mlm_model,
            device=args.mlm_device,
            use_mlm=not args.no_mlm,
        )
    except ImportError as exc:
        print(f"[ERROR] {exc}")
        print("[ERROR] Le scorer CamemBERT (MLM) est actif par defaut pour la commande 'correct'.")
        print("        Installez transformers + sentencepiece + torch, ou relancez avec --no-mlm")
        print("        pour utiliser le scorer heuristique de repli (cf. CONVENTIONS_NLP.md).")
        return 1

    if corrector.using_mlm:
        print(f"Scorer : CamemBERT MLM ({corrector.mlm_model_name})")
    else:
        print("Scorer : heuristique (MLM désactivé via --no-mlm ou indisponible)")

    events: list[dict] = []
    output_paths: list[Path] = []
    all_line_results = []

    for path, contract in zip(paths, contracts):
        result = corrector.apply_to_contract(
            contract,
            log_path=args.log_output,
            update_needs_review=not args.no_review_update,
        )
        events.extend(result["events"])
        all_line_results.extend(result["line_results"])

        if len(paths) == 1 and not args.output_dir:
            out_path = Path(args.output)
        else:
            out_dir = Path(args.output_dir or args.output)
            if out_dir.suffix.lower() == ".json" and len(paths) > 1:
                print("[ERROR] --output-dir must be a directory when correcting multiple contracts.")
                return 1
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / path.name

        save_json(contract, out_path)
        output_paths.append(out_path)

    n_lines_total = len(all_line_results)
    n_lines_corrected = sum(1 for r in all_line_results if r.events)
    mean_pairwise_cer = (
        sum(r.pairwise_cer for r in all_line_results) / n_lines_total if n_lines_total else 0.0
    )
    n_needs_review_after = sum(
        1
        for path, contract in zip(paths, contracts)
        for page in contract.get("pages", [])
        for line in page.get("lines", [])
        if bool(line.get("needs_review", False))
    )

    print(f"Lignes traitees      : {n_lines_total}")
    print(f"Corrections appliquees: {len(events)} (sur {n_lines_corrected} ligne(s))")
    print(f"CER pairwise moyen (raw vs corrige): {mean_pairwise_cer:.4f}")
    if not args.no_review_update:
        print(f"Lignes needs_review apres correction: {n_needs_review_after}")
    else:
        print("Reinjection needs_review desactivee (--no-review-update)")

    if len(output_paths) == 1:
        print(f"Updated contract   : {output_paths[0]}")
    else:
        print(f"Updated contracts  : {len(output_paths)} files in {output_paths[0].parent}")
    print(f"Correction log     : {args.log_output}")

    if args.cer_output:
        cer_report = {
            "n_lines_total": n_lines_total,
            "n_lines_corrected": n_lines_corrected,
            "mean_pairwise_cer": mean_pairwise_cer,
            "using_mlm": corrector.using_mlm,
            "mlm_model": corrector.mlm_model_name,
            "per_line": [
                {
                    "line_id": r.line_id,
                    "pairwise_cer": r.pairwise_cer,
                    "n_corrections": len(r.events),
                }
                for r in all_line_results
            ],
        }
        Path(args.cer_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.cer_output).write_text(
            json.dumps(cer_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"CER report         : {args.cer_output}")

    return 0


def cmd_normalize_contract(args: argparse.Namespace) -> int:
    try:
        paths = resolve_input_paths(args.input)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not paths:
        print(f"[ERROR] no JSON files found for input: {args.input}")
        return 1

    if len(paths) > 1 and not args.output_dir:
        print("[ERROR] When normalizing multiple contracts, specify --output-dir.")
        return 1

    normalizer = MedievalFrenchNormalizer.from_json(args.abbreviations)
    output_paths: list[Path] = []
    per_line_cer: list[dict] = []

    for path in paths:
        contract = load_json(path)
        for page in contract.get("pages", []):
            for line in page.get("lines", []):
                text = str(line.get("text", ""))
                normalized_text = normalizer.normalize(text)
                line["normalized_text"] = normalized_text
                # CER pairwise (brut vs normalisé) : mesure l'ampleur du changement
                # introduit par les règles à cette étape (pas de vérité terrain
                # disponible, cf. CONVENTIONS_NLP.md section 3).
                per_line_cer.append(
                    {
                        "line_id": line.get("line_id", ""),
                        "pairwise_cer": cer(text, normalized_text),
                    }
                )

        if len(paths) == 1 and not args.output_dir:
            out_path = Path(args.output)
        else:
            out_dir = Path(args.output_dir or args.output)
            if out_dir.suffix.lower() == ".json" and len(paths) > 1:
                print("[ERROR] --output-dir must be a directory when normalizing multiple contracts.")
                return 1
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / path.name

        save_json(contract, out_path)
        output_paths.append(out_path)

    mean_pairwise_cer = (
        sum(row["pairwise_cer"] for row in per_line_cer) / len(per_line_cer) if per_line_cer else 0.0
    )
    print(f"Lignes normalisees : {len(per_line_cer)}")
    print(f"CER pairwise moyen (raw vs normalise): {mean_pairwise_cer:.4f}")

    if len(output_paths) == 1:
        print(f"Normalized contract saved to: {output_paths[0]}")
    else:
        print(f"Normalized contracts saved: {len(output_paths)} files in {output_paths[0].parent}")

    if args.cer_output:
        cer_report = {
            "n_lines": len(per_line_cer),
            "mean_pairwise_cer": mean_pairwise_cer,
            "per_line": per_line_cer,
        }
        Path(args.cer_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.cer_output).write_text(
            json.dumps(cer_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"CER report         : {args.cer_output}")

    return 0


def cmd_detect_normalization(args: argparse.Namespace) -> int:
    abbreviations = {}
    if args.abbreviations and Path(args.abbreviations).exists():
        abbreviations = json.loads(Path(args.abbreviations).read_text(encoding="utf-8"))

    results = detect_normalization_candidates(
        lexicon_path=args.lexicon,
        output_dir=args.output_dir,
        top_n=args.top_n,
        abbreviations=abbreviations,
    )
    output = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def cmd_lexical_check(args: argparse.Namespace) -> int:
    if not Path(args.dictionary).exists():
        print(f"[ERROR] Dictionary not found: {args.dictionary}")
        return 1

    results = find_lexical_errors(
        dictionary_path=args.dictionary,
        lexicon_path=args.lexicon,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    output = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    split = stratified_split_records(records, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        Path(out_dir / f"{name}.json").write_text(
            json.dumps(split[name], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    digest = seal_test_set(
        split["test"],
        out_dir / "test_sealed.json",
        out_dir / "test_set.sha256",
    )

    print(f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"Test set SHA-256: {digest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cours 1 - Day-1 NLP workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Validate HTR data contract against schema")
    p.add_argument("--input", required=True)
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("eda", help="Compute EDA metrics from contract")
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.set_defaults(func=cmd_eda)

    p = sub.add_parser("review-queue", help="Build triage buckets and review CSV")
    p.add_argument("--input", required=True)
    p.add_argument("--csv-output", default="data/review/review_queue.csv")
    p.add_argument("--json-output", default="data/review/review_buckets.json")
    p.add_argument("--direct-threshold", type=float, default=0.9)
    p.add_argument("--exclude-threshold", type=float, default=0.6)
    p.add_argument("--char-std-threshold", type=float, default=0.2)
    p.set_defaults(func=cmd_review_queue)

    p = sub.add_parser("normalize", help="Normalize text via rule-based normalizer")
    p.add_argument("--abbreviations", default=DEFAULT_ABBR)
    p.add_argument("--text")
    p.add_argument("--csv-input")
    p.add_argument("--csv-output", default="data/normalized/normalized.csv")
    p.add_argument("--source-col", default="text")
    p.add_argument("--output-col", default="text_normalized")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("ablation", help="Compute CER before/after normalization")
    p.add_argument("--csv-input", required=True)
    p.add_argument("--reference-col", default="reference")
    p.add_argument("--hypothesis-col", default="text")
    p.add_argument("--abbreviations", default=DEFAULT_ABBR)
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser(
        "relative-eval",
        help="Compute average pairwise CER across several transcription variants",
    )
    p.add_argument("--csv-input", required=True)
    p.add_argument(
        "--variant-cols",
        default="raw,text_normalized,corrected",
        help="Comma-separated CSV columns containing variant transcriptions",
    )
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(func=cmd_relative_eval)

    p = sub.add_parser("normalize-contract", help="Normalize an HTR contract by adding normalized_text fields")
    p.add_argument("--input", required=True, help="Path to a JSON contract file or a directory containing JSON contracts")
    p.add_argument("--output", default="data/contracts/contract_normalized.json")
    p.add_argument("--output-dir", help="Directory to write normalized contracts when input contains multiple JSON files")
    p.add_argument("--abbreviations", default=DEFAULT_ABBR)
    p.add_argument(
        "--cer-output",
        default="data/review/normalize_cer_report.json",
        help="Optional JSON file to write the pairwise CER report (raw vs normalized, per line and average)",
    )
    p.set_defaults(func=cmd_normalize_contract)

    p = sub.add_parser(
        "detect-normalization",
        help="Discover dataset-specific normalization candidates from lexicon or output directory",
    )
    p.add_argument("--lexicon", help="Path to a lexicon CSV (e.g. nlp/lexique/lexicon.csv)")
    p.add_argument("--output-dir", help="Path to a prediction output directory (e.g. nlp/output)")
    p.add_argument("--top-n", type=int, default=50, help="Number of suspicious tokens to report")
    p.add_argument("--abbreviations", default=DEFAULT_ABBR)
    p.add_argument("--output", help="Optional JSON file to write the candidate report")
    p.set_defaults(func=cmd_detect_normalization)

    p = sub.add_parser(
        "lexical-check",
        help="Flag tokens absent from the Old French dictionary (real lexical errors, not just abbreviation markers)",
    )
    p.add_argument("--dictionary", default=DEFAULT_DICTIONARY, help="Path to dictionnaire_ancien_francais.json")
    p.add_argument("--lexicon", help="Path to a lexicon CSV (e.g. data/lexique/lexicon.csv)")
    p.add_argument("--output-dir", help="Path to a prediction output directory (e.g. data/nlp_output)")
    p.add_argument("--top-n", type=int, default=50, help="Number of unknown tokens to report")
    p.add_argument("--output", help="Optional JSON file to write the report")
    p.set_defaults(func=cmd_lexical_check)

    p = sub.add_parser("correct", help="Apply confidence-guided corrections (CamemBERT MLM by default)")
    p.add_argument("--input", required=True, help="Path to a JSON contract file or a directory containing JSON contracts")
    p.add_argument("--output", default="data/contracts/contract_corrected.json")
    p.add_argument("--output-dir", help="Directory to write corrected contracts when input contains multiple JSON files")
    p.add_argument("--log-output", default="data/review/correction_log.jsonl")
    p.add_argument(
        "--cer-output",
        default="data/review/correction_cer_report.json",
        help="Optional JSON file to write the pairwise CER report (raw vs corrected, per line and average)",
    )
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument(
        "--mlm-model",
        default="almanach/camembert-base",
        help="Masked language model used for contextual arbitration (default: almanach/camembert-base)",
    )
    p.add_argument(
        "--mlm-device",
        default="auto",
        help="Device for MLM scoring: cuda, cpu, or auto",
    )
    p.add_argument(
        "--no-mlm",
        action="store_true",
        help="Disable the MLM scorer and fall back to the lightweight heuristic scorer",
    )
    p.add_argument(
        "--no-review-update",
        action="store_true",
        help="Do not recompute needs_review after correction (keep the original HTR flag)",
    )
    p.set_defaults(func=cmd_correct)

    p = sub.add_parser("split", help="Create stratified split and seal test set")
    p.add_argument("--records", required=True, help="JSON array with century_estimate and document_type")
    p.add_argument("--output-dir", default="data/splits_nlp")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=67)
    p.set_defaults(func=cmd_split)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    returncode = int(args.func(args))
    log_run(args.command, args, returncode)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())