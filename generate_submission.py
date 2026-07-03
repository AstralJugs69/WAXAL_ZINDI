#!/usr/bin/env python3
import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("inference")

# Import helpers from our codebase
from src.inference.pipeline import VADSegmenter, LanguageIdentifier
from src.decoding.ctc_decoder import create_ctc_decoder, decode_logits
from src.data.dataset import parse_robust_csv, normalize_text

def find_test_audio_dir(search_roots, test_ids):
    """
    Finds the directory containing the test audio files by matching IDs from Test.csv.
    """
    logger.info("Scanning for test audio directory...")
    candidate_dirs = {}
    extensions = (".mp3", ".wav", ".m4a")
    
    # We will search in the search_roots
    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            # Skip hidden or system dirs
            if ".git" in root or "__pycache__" in root or "outputs" in root:
                continue
            # If the directory contains any audio files, count matches
            audio_files = [f for f in files if f.lower().endswith(extensions)]
            if audio_files:
                # Check how many test IDs match files in this directory
                matched_count = 0
                # To speed up, we check a sample of 200 IDs
                sample_ids = test_ids[:200]
                for audio_id in sample_ids:
                    found = False
                    for ext in extensions:
                        if f"{audio_id}{ext}" in files:
                            found = True
                            break
                    if found:
                        matched_count += 1
                if matched_count > 0:
                    candidate_dirs[root] = matched_count
                    logger.info(f"Directory {root} matched {matched_count}/{len(sample_ids)} sample test IDs.")
                    
    if candidate_dirs:
        # Return the directory with the highest matches
        best_dir = max(candidate_dirs, key=candidate_dirs.get)
        logger.info(f"Selected best audio directory: {best_dir} with {candidate_dirs[best_dir]}/{len(sample_ids)} sample matches.")
        return best_dir
        
    logger.warning("Could not find any directory matching the test IDs. Defaulting to current directory.")
    return "."

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for inference: {device}")
    
    # 1. Resolve paths
    test_csv_path = "Test.csv"
    if not os.path.exists(test_csv_path):
        # Try finding Test.csv recursively
        found = glob.glob("**/Test.csv", recursive=True)
        if found:
            test_csv_path = found[0]
            
    logger.info(f"Loading test CSV from: {test_csv_path}")
    test_df = parse_robust_csv(test_csv_path)
    
    # Extract test IDs as a list
    test_ids = list(test_df["id"].dropna())
    
    # Scan current directory and /kaggle/input
    audio_dir = find_test_audio_dir([".", "/kaggle/input"], test_ids)
    logger.info(f"Test audio directory: {audio_dir}")
    
    # 2. Define target languages
    target_languages = ["lin", "sna", "lug"]
    
    # Check what custom models we have trained
    models = {}
    processors = {}
    decoders = {}
    
    for lang in target_languages:
        # Check custom fold checkpoint
        custom_model_dir = f"outputs/{lang}_mms-300m_fold0/best_model"
        if os.path.exists(custom_model_dir):
            logger.info(f"Found custom fine-tuned model for {lang} at {custom_model_dir}")
            model = Wav2Vec2ForCTC.from_pretrained(custom_model_dir)
            processor = Wav2Vec2Processor.from_pretrained(custom_model_dir)
            # Re-set target lang to be absolutely sure
            processor.tokenizer.set_target_lang(lang)
            
            # Check KenLM binary
            lm_path = os.path.join(custom_model_dir, "lm.bin")
            if os.path.exists(lm_path):
                logger.info(f"Found compiled KenLM binary for {lang} at {lm_path}")
                try:
                    vocab = processor.tokenizer.get_vocab()
                    decoders[lang] = create_ctc_decoder(vocab_dict=vocab, kenlm_model_path=lm_path)
                except Exception as e:
                    logger.warning(f"Could not build KenLM decoder for {lang}: {e}. Falling back to greedy decoding.")
                    decoders[lang] = None
            else:
                logger.warning(f"KenLM binary not found for {lang} at {lm_path}. Using greedy decoding.")
                decoders[lang] = None
        else:
            # Fallback to pre-trained mms-1b-all for this language
            logger.info(f"No custom model found for {lang}. Falling back to pre-trained facebook/mms-1b-all")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang=lang, ignore_mismatched_sizes=True)
            processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all", target_lang=lang)
            processor.tokenizer.set_target_lang(lang)
            decoders[lang] = None
            
        model = model.to(device)
        model.eval()
        models[lang] = model
        processors[lang] = processor
        
    # Load Language Identifier
    lid = LanguageIdentifier(target_languages=target_languages)
    vad = VADSegmenter()
    
    # 3. Perform Inference
    predictions = []
    logger.info(f"Starting inference on {len(test_df)} test files...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        audio_id = row["id"]
        
        # Resolve audio file path
        audio_path = None
        for ext in [".mp3", ".wav", ".m4a", ""]:
            temp_path = os.path.join(audio_dir, f"{audio_id}{ext}")
            if os.path.exists(temp_path):
                audio_path = temp_path
                break
                
        if not audio_path:
            logger.warning(f"Could not find audio file for ID: {audio_id}. Skipping.")
            predictions.append({"ID": audio_id, "Target": ""})
            continue
            
        try:
            # Load audio at 16kHz mono
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Segment audio via VAD to prevent OOM
            chunks = vad.segment(y, sr=16000)
            
            # Run Language Identification on the longest segment
            longest_chunk = max(chunks, key=len)
            detected_lang = lid.identify(longest_chunk, sr=16000)
            
            # Route to model, processor, and decoder
            model = models[detected_lang]
            processor = processors[detected_lang]
            decoder = decoders[detected_lang]
            
            transcriptions = []
            for chunk in chunks:
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = model(**inputs).logits[0].cpu().numpy()
                    
                # Decode logits
                if decoder is not None:
                    chunk_text = decode_logits(decoder, logits, beam_width=128)
                else:
                    # Greedy decoding fallback
                    pred_ids = np.argmax(logits, axis=-1)
                    chunk_text = processor.decode(pred_ids)
                    
                normalized_chunk = normalize_text(chunk_text)
                if normalized_chunk:
                    transcriptions.append(normalized_chunk)
                    
            final_text = " ".join(transcriptions)
            predictions.append({"ID": audio_id, "Target": final_text})
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_id}: {e}")
            predictions.append({"ID": audio_id, "Target": ""})
            
    # 4. Save to submission CSV
    submission_df = pd.DataFrame(predictions)
    # Ensure correct column headers as per SampleSubmission.csv
    submission_df.to_csv("submission.csv", index=False)
    logger.info("Successfully generated submission.csv!")

if __name__ == "__main__":
    main()
