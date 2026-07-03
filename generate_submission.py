#!/usr/bin/env python3
"""
Submission Generation Script for WAXAL ASR Challenge.
Runs end-to-end inference on the test set parquet files using the trained adapters and KenLMs.
"""
import os
import argparse
import glob
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset, Audio
from huggingface_hub import list_repo_files

from src.inference.pipeline import ProductionASRPipeline
from src.data.dataset import normalize_text

def get_audio_data(audio_info):
    """
    Robustly extracts the raw audio numpy array and sampling rate from HuggingFace Audio column.
    """
    if audio_info is None:
        return None, None
    if isinstance(audio_info, dict):
        return audio_info.get("array"), audio_info.get("sampling_rate")
    if hasattr(audio_info, "array") and hasattr(audio_info, "sampling_rate"):
        return audio_info.array, audio_info.sampling_rate
    try:
        return audio_info["array"], audio_info["sampling_rate"]
    except Exception:
        pass
    return None, None

def load_waxal_test_dataset(lang):
    """
    Loads test parquet files directly from Hugging Face Hub for the specified language.
    """
    repo_id = "google/WaxalNLP"
    print(f"\nRetrieving test files list for {repo_id} ({lang})...")
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Failed to list files from Hugging Face: {e}")
        return None
        
    lang_dir = f"data/ASR/{lang}"
    test_patterns = [f for f in all_files if f.startswith(lang_dir) and f"{lang}-test-" in f and f.endswith(".parquet")]
    
    if not test_patterns:
        print(f"Warning: No test parquet files found for language: {lang}")
        return None
        
    test_urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}" for f in test_patterns]
    print(f"Loading {len(test_urls)} test parquet files...")
    
    ds = load_dataset("parquet", data_files={"test": test_urls})["test"]
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Zindi Submission CSV")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory where models and adapters are saved")
    parser.add_argument("--test_csv", type=str, default="Test.csv", help="Path to Zindi Test.csv file")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Path to output submission.csv")
    parser.add_argument("--beam_width", type=int, default=128, help="Beam search width for pyctcdecode")
    parser.add_argument("--alpha", type=float, default=0.5, help="LM weight (alpha) for decoder")
    parser.add_argument("--beta", type=float, default=1.5, help="Word insertion penalty (beta) for decoder")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Test CSV
    if not os.path.exists(args.test_csv):
        # Check parent folder or alternative locations
        alt_test = os.path.join("..", args.test_csv)
        if os.path.exists(alt_test):
            args.test_csv = alt_test
        else:
            raise FileNotFoundError(f"Test CSV not found at: {args.test_csv}")
            
    print(f"Loading test IDs from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)
    test_ids = list(test_df["ID"].values)
    print(f"Total test IDs: {len(test_ids)}")

    # 2. Determine target languages present in Test.csv
    # Audio IDs usually start with "lug_", "lin_", "sna_"
    target_langs = set()
    for tid in test_ids:
        prefix = tid.split("_")[0]
        if prefix in ["lin", "lug", "sna"]:
            target_langs.add(prefix)
            
    if not target_langs:
        # Fallback to all three if prefixes cannot be parsed
        target_langs = {"lin", "lug", "sna"}
    target_langs = list(target_langs)
    print(f"Target languages found in Test.csv: {target_langs}")

    # 3. Locate trained adapters and KenLMs
    adapter_paths = {}
    kenlm_paths = {}
    
    for lang in target_langs:
        # Search for outputs folder for the language fold 0
        lang_dir_pattern = os.path.join(args.outputs_dir, f"{lang}_*_fold*")
        matching_dirs = glob.glob(lang_dir_pattern)
        
        if matching_dirs:
            # Pick first matching folder, look for best_model
            best_model_path = os.path.join(matching_dirs[0], "best_model")
            if os.path.exists(best_model_path):
                adapter_paths[lang] = best_model_path
                print(f"Found trained adapter for {lang} at: {best_model_path}")
                
                # Check for KenLM
                lm_path = os.path.join(best_model_path, "lm.bin")
                if os.path.exists(lm_path):
                    kenlm_paths[lang] = lm_path
                    print(f"  Found KenLM model for {lang} at: {lm_path}")
                else:
                    # Check for lm_ref file
                    ref_path = os.path.join(best_model_path, "lm_bin_path.txt")
                    if os.path.exists(ref_path):
                        with open(ref_path, "r") as f:
                            resolved_lm = f.read().strip()
                        if os.path.exists(resolved_lm):
                            kenlm_paths[lang] = resolved_lm
                            print(f"  Found KenLM model for {lang} via ref: {resolved_lm}")
                            
    # 4. Initialize Production Routing Pipeline
    print("\nInitializing Production ASR Pipeline...")
    pipeline = ProductionASRPipeline(
        base_model_id="facebook/mms-300m",
        target_languages=target_langs,
        adapter_paths=adapter_paths if adapter_paths else None,
        kenlm_paths=kenlm_paths,
        beam_width=args.beam_width
    )
    
    # Push model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.model = pipeline.model.to(device)
    print(f"Model placed on device: {device}")
    
    # Overwrite default decoder alpha and beta if requested
    for lang, decoder in pipeline.decoders.items():
        if decoder is not None:
            decoder._alpha = args.alpha
            decoder._beta = args.beta

    # 5. Load and index the test datasets lazily (prevent RAM OOMs)
    # Map from ID -> (dataset, index_in_dataset)
    id_to_audio_locator = {}
    
    for lang in target_langs:
        ds = load_waxal_test_dataset(lang)
        if ds is not None:
            # Only select the ID column to construct the index map fast without reading raw audio
            ids_ds = ds.select_columns(["id"])
            for idx, ex in enumerate(tqdm(ids_ds, desc=f"Indexing {lang} test IDs")):
                ex_id = ex["id"]
                id_to_audio_locator[ex_id] = (ds, idx)
                
    # 6. Perform batch-free lazy inference
    predictions = []
    
    print("\nRunning test set inference...")
    for tid in tqdm(test_ids, desc="Transcribing test audios"):
        if tid in id_to_audio_locator:
            ds, idx = id_to_audio_locator[tid]
            try:
                # Retrieve the full row (triggers lazy audio decoding for this specific sample)
                example = ds[idx]
                audio_array, sr = get_audio_data(example["audio"])
                
                if audio_array is not None and sr is not None:
                    transcription, lang_route = pipeline.transcribe(audio_array, sr)
                    # Normalize predictions for Zindi submission criteria
                    final_text = normalize_text(transcription)
                    predictions.append({"ID": tid, "Target": final_text})
                else:
                    print(f"\nWarning: Empty audio array for ID: {tid}")
                    predictions.append({"ID": tid, "Target": ""})
            except Exception as e:
                print(f"\nError transcribing {tid}: {e}")
                predictions.append({"ID": tid, "Target": ""})
        else:
            # Fallback if ID is missing from dataset
            print(f"\nWarning: Test ID {tid} not found in Hugging Face test splits.")
            predictions.append({"ID": tid, "Target": ""})
            
    # 7. Write to CSV
    output_df = pd.DataFrame(predictions)
    # Ensure ID ordering matches original Test.csv exactly
    output_df = output_df.set_index("ID").reindex(test_ids).reset_index()
    
    # Save output
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSuccessfully generated submission file: {args.output_csv}")
    print(output_df.head(10))

if __name__ == "__main__":
    main()
