"""Download pinned official Hugging Face weights, ready for Azure registration.

Run on the administrator's computer; no GPU or Azure connection is needed.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
# Paths are relative to models/<parser>/ and match the existing workers.
# Change revisions deliberately; download upgrades into a fresh --models-dir.
MODELS = {
    "mineru": {
        "pipeline": {
            "repo_id": "opendatalab/PDF-Extract-Kit-1.0",
            "revision": "ed6b654c018d742e65a17671e379c5e6ecc87ec9",
            # Same model groups as MinerU 3.3.1's pipeline downloader.
            "allow_patterns": [
                "models/Layout/PP-DocLayoutV2/*",
                "models/MFR/unimernet_hf_small_2503/*",
                "models/MFR/pp_formulanet_plus_m/*",
                "models/OCR/paddleocr_torch/*",
                "models/TabRec/SlanetPlus/slanet-plus.onnx",
                "models/TabRec/UnetStructure/unet.onnx",
                "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx",
            ],
        },
        "vlm": {
            "repo_id": "opendatalab/MinerU2.5-Pro-2605-1.2B",
            "revision": "bff20d4ae2bf202df9f45284b4d43681555a97ed",
        },
    },
    "paddle": {
        "official_models/PP-DocLayoutV3": {
            "repo_id": "PaddlePaddle/PP-DocLayoutV3",
            "revision": "241f8bdfc77a7c7bee915a5057aaee58c235a8d3",
        },
        "official_models/PaddleOCR-VL-1.6": {
            "repo_id": "PaddlePaddle/PaddleOCR-VL-1.6",
            "revision": "c5630abae1d940eafe0697512a0325494b02ab42",
        },
    },
}


def download(parser, models_dir):
    destination = models_dir / parser
    for name, source in MODELS[parser].items():
        print(f"Downloading {source['repo_id']}@{source['revision']} -> {destination / name}",
              flush=True)
        # local_dir writes real files in the layout expected by our workers.
        # HF keeps download metadata here, so rerunning can reuse completed files.
        snapshot_download(**source, local_dir=destination / name)
    print(f"Download complete: {destination}. Ready for register_models.py.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parser", required=True, choices=["mineru", "paddle"])
    cli.add_argument("--models-dir", type=Path, default=ROOT / "models",
                     help="Destination parent directory (default: repository's models/)")
    args = cli.parse_args()
    download(args.parser, args.models_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
