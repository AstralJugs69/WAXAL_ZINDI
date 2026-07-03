#!/usr/bin/env python3
"""
Utility script to pre-download all required Hugging Face models and datasets 
for the WAXAL ASR challenge. This avoids HF throttling or network failures 
during active GPU/TPU training runs.
"""
import os
import argparse
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Corrected models configuration matching WAXAL codebase
MODELS = [
    {"id": "facebook/mms-300m", "type": "mms_model"},
    {"id": "facebook/mms-1b-all", "type": "mms_processor", "langs": ["lin", "lug", "sna"]},
    {"id": "openai/whisper-small", "type": "whisper"},
]

DATASETS = [
    {"path": "mozilla-foundation/common_voice_17_0", "name": "ln", "splits": ["train", "validation", "test"]},
    {"path": "mozilla-foundation/common_voice_17_0", "name": "lg", "splits": ["train", "validation", "test"]},
    {"path": "google/fleurs", "name": "ln_cd", "splits": ["train", "validation"]},
    {"path": "google/fleurs", "name": "lg_ug", "splits": ["train", "validation"]},
    {"path": "google/fleurs", "name": "sn_zw", "splits": ["train", "validation"]},
]

def main():
    parser = argparse.ArgumentParser(description="Download HF models and datasets to cache.")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", help="Local directory to store the HF cache.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token.")
    args = parser.parse_args()

    # Configure HuggingFace cache directories
    os.environ["HF_HOME"] = os.path.abspath(args.cache_dir)
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
    
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN is not set. Downloading gated datasets may fail.")

    # 1. Download Models & Processors
    logger.info("=== Downloading Models ===")
    for model_info in MODELS:
        model_id = model_info["id"]
        m_type = model_info["type"]
        logger.info(f"Processing: {model_id} ({m_type})")
        try:
            if m_type == "mms_model":
                from transformers import Wav2Vec2ForCTC
                Wav2Vec2ForCTC.from_pretrained(model_id, token=token)
            elif m_type == "mms_processor":
                from transformers import Wav2Vec2Processor
                for lang in model_info["langs"]:
                    logger.info(f"  Downloading Wav2Vec2Processor for language: {lang}")
                    Wav2Vec2Processor.from_pretrained(model_id, target_lang=lang, token=token)
            elif m_type == "whisper":
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                WhisperProcessor.from_pretrained(model_id, token=token)
                WhisperForConditionalGeneration.from_pretrained(model_id, token=token)
            logger.info(f"Successfully downloaded {model_id}")
        except Exception as e:
            logger.error(f"Failed {model_id}: {e}")

    # 2. Download Datasets
    logger.info("=== Downloading Datasets ===")
    for ds_info in DATASETS:
        path = ds_info["path"]
        name = ds_info["name"]
        for split in ds_info["splits"]:
            try:
                load_dataset(path, name, split=split, trust_remote_code=True, token=token)
                logger.info(f"Successfully loaded {path} [{name}] ({split})")
            except Exception as e:
                logger.error(f"Failed {path} [{name}] split {split}: {e}")

    logger.info("=== All downloads completed ===")

if __name__ == "__main__":
    main()
