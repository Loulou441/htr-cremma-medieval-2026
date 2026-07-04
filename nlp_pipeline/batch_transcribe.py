import hashlib
import json
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

try:
    from kraken import blla, rpred
    from PIL import Image
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    print("Assurez-vous d'avoir installé kraken et pillow (voir requirements.txt).")
    sys.exit(1)

# Configuration
TXT_FILE = "resources/manuscrits_xiii_siecle.txt"
OUTPUT_DIR = Path("data/predictions")
DOWNLOAD_DIR = Path("data/downloads")
MODEL_PATH = Path("models/exp3opt_finetune_20260615_1849.safetensors")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_data_contract(
    image_bytes: bytes,
    image_filename: str,
    preds: list,
    model_name: str,
    confidence_threshold: float = 0.9,
) -> dict:
    """Build an HTR data contract JSON from Kraken predictions.
    La structure produite doit rester strictement conforme à nlp_pipeline/json_files/htr_data_contract_schema.json.
    """
    doc_id = Path(image_filename).stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = []
    for i, p in enumerate(preds, start=1):
        text = p.prediction or ""
        char_confs = list(p.confidences) if p.confidences else []
        confidence = sum(char_confs) / len(char_confs) if char_confs else 0.0

        char_std = statistics.pstdev(char_confs) if len(char_confs) > 1 else 0.0
        needs_review = confidence < confidence_threshold or char_std > 0.2

        polygon = []
        if hasattr(p, "cuts") and p.cuts:
            try:
                polygon = [[int(x), int(y)] for x, y in p.cuts]
            except Exception:
                polygon = []

        lines.append({
            "line_id": f"{doc_id}_l{i:03d}",
            "text": text,
            "confidence": round(confidence, 4),
            "char_confidences": [round(c, 4) for c in char_confs],
            "candidates": None,
            "needs_review": needs_review,
            "polygon": polygon,
            "reading_order": i,
        })

    return {
        "document_id": doc_id,
        "metadata": {
            "source": image_filename,
            "century_estimate": "XIII",
            "document_type": "unknown",
            "scan_quality": "unknown",
            "sha256": _sha256_bytes(image_bytes),
            "model": model_name,
            "produced_at": datetime.now().isoformat(timespec="seconds"),
        },
        "pages": [
            {
                "page_id": "p001",
                "image_path": image_filename,
                "lines": lines,
            }
        ],
    }


def save_data_contract(contract: dict) -> Path:
    """Save data contract JSON to OUTPUT_DIR and return the path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{contract['document_id']}.json"
    out_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path

def extract_gallica_ark(url):
    """Extrait l'identifiant ARK de Gallica à partir d'une URL."""
    match = re.search(r'(ark:/[0-9]+/btv[a-b0-9]+)', url)
    if match:
        return match.group(1)
    return None

def get_local_pages(ark_id):
    """Liste les pages déjà téléchargées pour un manuscrit, triées par numéro de page."""
    ark_slug = ark_id.split("/")[-1]
    ark_dir = DOWNLOAD_DIR / ark_slug
    if not ark_dir.exists():
        return []

    def page_num(p):
        m = re.search(r'_f(\d+)\.jpg$', p.name)
        return int(m.group(1)) if m else 0

    return sorted(ark_dir.glob(f"{ark_slug}_f*.jpg"), key=page_num)

def load_local_model():
    """Charge le modèle Kraken fine-tuné directement depuis le disque (hors HuggingFace)."""
    from kraken.models import load_safetensors
    from kraken.lib.models import TorchSeqRecognizer

    vgsl_models = load_safetensors(str(MODEL_PATH), tasks=["recognition"])
    return TorchSeqRecognizer(vgsl_models[0], device="cpu")

def process_batch():
    # 1. Extraction des manuscrits du fichier texte
    if not Path(TXT_FILE).exists():
        print(f"❌ Le fichier {TXT_FILE} est introuvable.")
        return

    content = Path(TXT_FILE).read_text(encoding="utf-8")
    # Recherche de toutes les lignes contenant un lien Gallica btv
    lines = content.splitlines()
    manuscripts = []
    
    for line in lines:
        if "https://gallica.bnf.fr" in line:
            # Nettoyage rapide du nom
            name = line.split(":")[0].replace("", "").replace("", "").replace("", "").strip()
            url = line.split("https://")[1]
            url = "https://" + url
            ark_id = extract_gallica_ark(url)
            if ark_id:
                manuscripts.append({"name": name if name else "Manuscrit_Inconnu", "ark_id": ark_id})

    # Sélection des manuscrits uniques (dédoublonnage par ARK)
    seen = set()
    unique_manuscripts = []
    for m in manuscripts:
        if m["ark_id"] not in seen:
            seen.add(m["ark_id"])
            unique_manuscripts.append(m)

    print(f"📚 {len(unique_manuscripts)} manuscrits uniques trouvés. Lancement du traitement par lot...")

    # 2. Chargement du modèle fine-tuné local (exp3opt_finetune_20260615_1849)
    if not MODEL_PATH.exists():
        print(f"❌ Modèle introuvable : {MODEL_PATH}")
        return
    print(f"⏳ Chargement du modèle Kraken `{MODEL_PATH.name}`...")
    net = load_local_model()
    print("✅ Modèle chargé avec succès sur le CPU.")

    # 3. Boucle sur les manuscrits et leurs pages déjà téléchargées
    total = len(unique_manuscripts)
    for idx, ms in enumerate(unique_manuscripts, start=1):
        pages = get_local_pages(ms["ark_id"])
        print(f"\n📖 [{idx}/{total}] Traitement : {ms['name']} ({ms['ark_id']}) — {len(pages)} page(s) locale(s)")

        if not pages:
            print("  ⚠️ Aucune page téléchargée trouvée, manuscrit ignoré.")
            continue

        for img_path in pages:
            print(f"  📄 {img_path.name}...", end="", flush=True)

            try:
                # Ouverture et conversion en niveaux de gris (Mode L), requis par le modèle Kraken
                image = Image.open(img_path).convert("L")

                # Étape 1 : Segmentation native de Kraken (BLLA)
                segmentation = blla.segment(image, device="cpu")

                if not segmentation.lines:
                    print(" Ignorée (Aucune ligne détectée).")
                    continue

                # Étape 2 : Transcription par reconnaissance
                preds = list(rpred.rpred(net, image, segmentation))

                # Étape 3 : Construction du Data Contract
                image_bytes = img_path.read_bytes()
                contract = build_data_contract(
                    image_bytes=image_bytes,
                    image_filename=img_path.name,
                    preds=preds,
                    model_name=MODEL_PATH.name
                )

                # Customisation mineure du titre de métadonnées pour refléter le vrai manuscrit
                contract["metadata"]["document_type"] = ms["name"]

                # Étape 4 : Sauvegarde automatique du JSON final
                out_path = save_data_contract(contract)
                print(f" Sauvegardée -> `{out_path}`")

            except Exception as e:
                print(f" Erreur critique lors de la transcription : {e}")

if __name__ == "__main__":
    process_batch()
    print("\n🏁 Traitement par lot terminé. Les fichiers JSON sont dans `data/predictions/`.")