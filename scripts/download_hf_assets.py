#!/usr/bin/env python3
"""
Utility script to pre-download all required Hugging Face models and datasets 
for the WAXAL ASR challenge. This avoids HF throttling or network failures 
during active GPU/TPU training runs.
"""
import os
import argparse
import shutil
import logging
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCTC, AutoModelForSpeechSeq2Seq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Models to download
MODELS = [
    "facebook/mms-300m",
    "openai/whisper-small",
]

# Datasets and configs to download
DATASETS = [
    # Common Voice 17.0 (gated, requires HF_TOKEN)
    {"path": "mozilla-foundation/common_voice_17_0", "name": "ln", "splits": ["train", "validation", "test"]},
    {"path": "mozilla-foundation/common_voice_17_0", "name": "lg", "splits": ["train", "validation", "test"]},
    # FLEURS
    {"path": "google/fleurs", "name": "ln_cd", "splits": ["train", "validation"]},
    {"path": "google/fleurs", "name": "lg_ug", "splits": ["train", "validation"]},
    {"path": "google/fleurs", "name": "sn_zw", "splits": ["train", "validation"]},
]

def main():
    parser = argparse.ArgumentParser(description="Download HF models and datasets to cache.")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", help="Local directory to store the HF cache.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (required for gated datasets).")
    parser.add_argument("--zip_output", type=str, default=None, help="If specified, compress the cache directory into this zip/tar file.")
    args = parser.parse_args()

    # Configure HuggingFace cache directories
    os.environ["HF_HOME"] = os.path.abspath(args.cache_dir)
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
    
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN is not set. Downloading gated datasets (e.g. Common Voice 17.0) "
            "will fail if you have not logged in or provided a token."
        )

    # 1. Download Models
    logger.info("=== Downloading Models ===")
    for model_id in MODELS:
        logger.info(f"Downloading model & processor for: {model_id}")
        try:
            AutoProcessor.from_pretrained(model_id, token=token)
            if "mms" in model_id:
                AutoModelForCTC.from_pretrained(model_id, token=token)
            else:
                AutoModelForSpeechSeq2Seq.from_pretrained(model_id, token=token)
            logger.info(f"Successfully downloaded {model_id}")
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")

    # 2. Download Datasets
    logger.info("=== Downloading Datasets ===")
    for ds_info in DATASETS:
        path = ds_info["path"]
        name = ds_info["name"]
        logger.info(f"Downloading dataset: {path} [{name}]")
        for split in ds_info["splits"]:
            try:
                load_dataset(
                    path, 
                    name, 
                    split=split, 
                    trust_remote_code=True, 
                    token=token
                )
                logger.info(f"Successfully downloaded {path} [{name}] - {split}")
            except Exception as e:
                logger.error(f"Failed to download {path} [{name}] split '{split}': {e}")

    logger.info("=== All downloads completed ===")
    
    # 3. Zip Output if requested
    if args.zip_output:
        logger.info(f"Compressing cache directory '{args.cache_dir}' into '{args.zip_output}'...")
        # Get base name and format
        base_name, ext = os.path.splitext(args.zip_output)
        if ext == ".zip":
            fmt = "zip"
        elif ext in [".tar", ".gz", ".tgz"]:
            fmt = "gztar"
            base_name = base_name.replace(".tar", "")
        else:
            fmt = "zip"
            base_name = args.zip_output
            
        shutil.make_archive(base_name, fmt, args.cache_dir)
        logger.info(f"Compression completed: {base_name}.{fmt if fmt != 'gztar' else 'tar.gz'}")

if __name__ == "__main__":
    main()
