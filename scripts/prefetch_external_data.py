#!/usr/bin/env python3
"""
Download and cache external speech corpora used for WAXAL training.

Sources (per language):
  lin: Common Voice (ln) + FLEURS (ln_cd)
  lug: Common Voice (lg) + FLEURS (lg_ug)
  sna: FLEURS (sn_zw) only

Common Voice is gated — set HF_TOKEN (and accept the dataset license on HF).
FLEURS is open and should always download.

Usage:
  python scripts/prefetch_external_data.py --langs lin,sna,lug
  python scripts/prefetch_external_data.py --langs lin --sources fleurs
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prefetch_external")


def configure_hf_cache():
    if os.environ.get("HF_HOME"):
        return
    if os.path.exists("/kaggle/temp"):
        hf_home = "/kaggle/temp/hf_home"
    elif os.path.exists("/kaggle/working"):
        hf_home = "/tmp/hf_home"
    else:
        hf_home = str(ROOT / "hf_home")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    logger.info(f"HF_HOME={hf_home}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, default="lin,sna,lug")
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma list e.g. common_voice,fleurs. Empty = all configured sources.",
    )
    args = parser.parse_args()
    configure_hf_cache()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    sources = [x.strip() for x in args.sources.split(",") if x.strip()] or None

    from src.data.external_corpora import EXTERNAL_CORPUS_CONFIGS, load_external_corpus

    for lang in langs:
        if lang not in EXTERNAL_CORPUS_CONFIGS:
            logger.warning(f"No external config for {lang}; skip.")
            continue
        logger.info(f"=== Prefetch external data for {lang} ===")
        ds = load_external_corpus(lang, sources=sources)
        if ds is None:
            logger.warning(f"No external data loaded for {lang}")
        else:
            logger.info(f"{lang}: cached {len(ds)} external examples")

    logger.info("Prefetch complete.")


if __name__ == "__main__":
    main()
