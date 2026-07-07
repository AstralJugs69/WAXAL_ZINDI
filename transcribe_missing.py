#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import datasets
import librosa

# Add current directory to path to allow importing from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_submission import (
    _load_gemma_model_and_processor, 
    _prepare_gemma_inputs, 
    _as_mono_float32_audio, 
    _move_inputs_to_device, 
    get_outputs_dir,
    GEMMA_USER_MESSAGE,
    GEMMA_SYSTEM_MESSAGE
)
from src.data.dataset import normalize_text
from src.inference.pipeline import VADSegmenter

def main():
    parser = argparse.ArgumentParser(description="Transcribe specific missing audio IDs and update submission.csv")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN is not set. Gated models may fail to load.")

    missing_ids = ["lug_76835", "sna_37509", "sna_77897"]
    print(f"Targeting missing audio IDs: {missing_ids}")

    # 1. Fetch missing audio data from Hugging Face
    audio_data_dict = {}
    languages = {"lug_76835": "lug", "sna_37509": "sna", "sna_77897": "sna"}
    
    for lang in ["lug", "sna"]:
        lang_ids = [k for k, v in languages.items() if v == lang]
        if not lang_ids:
            continue
        print(f"Loading test split for {lang} from Hugging Face...")
        try:
            lang_test = datasets.load_dataset("google/WaxalNLP", name=f"{lang}_asr", split="test", streaming=True)
            lang_test = lang_test.cast_column("audio", datasets.Audio(sampling_rate=16000))
            for ex in lang_test:
                ex_id = ex.get("id") or ex.get("client_id") or ex.get("speaker_id")
                if ex_id in lang_ids:
                    print(f"  Found audio for {ex_id}")
                    audio_data_dict[ex_id] = {
                        "array": np.asarray(ex["audio"]["array"]).copy(),
                        "sampling_rate": ex["audio"]["sampling_rate"]
                    }
        except Exception as e:
            print(f"Error loading {lang} split: {e}")

    # 2. Transcribe each missing ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vad = VADSegmenter()
    predictions = {}

    # Group target IDs by language so we only load each model once
    for lang in ["lug", "sna"]:
        lang_ids = [k for k in missing_ids if languages.get(k) == lang]
        if not lang_ids:
            continue
            
        custom_gemma_dir = f"{get_outputs_dir()}/{lang}_gemma-3n-E2B-it_fold0/best_model"
        if not os.path.exists(custom_gemma_dir):
            print(f"Error: Fine-tuned Gemma directory not found for {lang} at {custom_gemma_dir}")
            continue

        print(f"Loading Gemma model and processor for {lang}...")
        import logging
        logger = logging.getLogger("worker")
        model, processor = _load_gemma_model_and_processor(custom_gemma_dir, device_map, hf_token, logger)
        model.eval()

        for audio_id in lang_ids:
            audio_data = audio_data_dict.get(audio_id)
            if audio_data is None:
                print(f"Warning: Audio data for {audio_id} was not found in the Hugging Face dataset!")
                predictions[audio_id] = ""
                continue

            try:
                y = _as_mono_float32_audio(audio_data["array"])
                sr = audio_data["sampling_rate"]
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    y = _as_mono_float32_audio(y)
                    sr = 16000

                duration = len(y) / sr
                if duration <= 20.0:
                    chunks = [y]
                else:
                    chunks = vad.segment(y, sr=16000)

                # Limit chunk size to max 20 seconds
                MAX_CHUNK_SAMPLES = 20 * 16000
                safe_chunks = []
                for chunk in chunks:
                    if len(chunk) > MAX_CHUNK_SAMPLES:
                        for start in range(0, len(chunk), MAX_CHUNK_SAMPLES):
                            sub_chunk = chunk[start:start+MAX_CHUNK_SAMPLES]
                            if len(sub_chunk) >= 16000 * 0.5:
                                safe_chunks.append(sub_chunk)
                    else:
                        safe_chunks.append(chunk)
                chunks = safe_chunks

                transcriptions = []
                for chunk in chunks:
                    chunk = _as_mono_float32_audio(chunk)
                    if chunk is None:
                        continue
                    
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": GEMMA_SYSTEM_MESSAGE}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio": chunk},
                                {"type": "text", "text": GEMMA_USER_MESSAGE},
                            ],
                        }
                    ]
                    inputs = _prepare_gemma_inputs(processor, messages, chunk, logger)
                    model_dtype = getattr(model, "dtype", None)
                    if model_dtype is None:
                        try:
                            model_dtype = next(model.parameters()).dtype
                        except StopIteration:
                            model_dtype = None
                    inputs = _move_inputs_to_device(inputs, device, dtype=model_dtype)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=96,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            use_cache=True,
                            do_sample=False,
                            repetition_penalty=1.15,
                            no_repeat_ngram_size=4,
                        )
                    input_len = inputs["input_ids"].shape[1]
                    chunk_text = processor.tokenizer.decode(
                        outputs[0][input_len:], skip_special_tokens=True
                    )
                    
                    normalized_chunk = normalize_text(chunk_text)
                    if normalized_chunk:
                        transcriptions.append(normalized_chunk)

                final_text = " ".join(transcriptions)
                predictions[audio_id] = final_text
                print(f"Transcribed {audio_id}: '{final_text}'")
            except Exception as e:
                print(f"Failed to transcribe {audio_id}: {e}")
                predictions[audio_id] = ""

        # Free GPU memory
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Update the existing submission.csv
    submission_path = "submission.csv"
    if not os.path.exists(submission_path):
        submission_path = "outputs/submission.csv"
    
    if os.path.exists(submission_path):
        print(f"Reading existing submission at: {submission_path}")
        sub_df = pd.read_csv(submission_path)
        id_col = "ID" if "ID" in sub_df.columns else "id" if "id" in sub_df.columns else sub_df.columns[0]
        target_col = "Target" if "Target" in sub_df.columns else "target" if "target" in sub_df.columns else sub_df.columns[1]
        
        # Ensure string type
        sub_df[id_col] = sub_df[id_col].astype(str)
        
        updated_count = 0
        for audio_id, text in predictions.items():
            mask = sub_df[id_col] == audio_id
            if mask.any():
                sub_df.loc[mask, target_col] = text
                updated_count += 1
                print(f"Updated {audio_id} in {submission_path} with: '{text}'")
            else:
                # If not found, append it
                new_row = {id_col: audio_id, target_col: text}
                sub_df = pd.concat([sub_df, pd.DataFrame([new_row])], ignore_index=True)
                updated_count += 1
                print(f"Appended {audio_id} to {submission_path} with: '{text}'")
                
        sub_df.to_csv(submission_path, index=False)
        print(f"Successfully wrote {updated_count} updates to {submission_path}")
    else:
        print(f"Error: submission.csv not found at {submission_path}")

if __name__ == "__main__":
    main()
