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

def get_outputs_dir():
    if os.path.exists("/kaggle/temp/outputs"):
        return "/kaggle/temp/outputs"
    elif os.path.exists("/tmp/outputs"):
        return "/tmp/outputs"
    return "outputs"

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
    

    
    # 2. Define target languages
    target_languages = ["lin", "sna", "lug"]
    
    # Check what custom models we have trained
    models = {}
    processors = {}
    decoders = {}
    model_families = {}
    
    for lang in target_languages:
        # Check custom fold checkpoint for Gemma first
        custom_gemma_dir = f"{get_outputs_dir()}/{lang}_gemma-3n-E2B-it_fold0/best_model"
        custom_mms_dir = f"{get_outputs_dir()}/{lang}_mms-300m_fold0/best_model"
        
        if os.path.exists(custom_gemma_dir):
            logger.info(f"Found custom fine-tuned Gemma model for {lang} at {custom_gemma_dir}")
            import timm
            from transformers import Gemma3nForConditionalGeneration, AutoProcessor
            model = Gemma3nForConditionalGeneration.from_pretrained(
                custom_gemma_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(custom_gemma_dir)
            model_families[lang] = "gemma"
            decoders[lang] = None
        elif os.path.exists(custom_mms_dir):
            logger.info(f"Found custom fine-tuned MMS model for {lang} at {custom_mms_dir}")
            model = Wav2Vec2ForCTC.from_pretrained(custom_mms_dir)
            processor = Wav2Vec2Processor.from_pretrained(custom_mms_dir)
            processor.tokenizer.set_target_lang(lang)
            model_families[lang] = "mms"
            
            # Check KenLM binary
            lm_path = os.path.join(custom_mms_dir, "lm.bin")
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
            model_families[lang] = "mms"
            decoders[lang] = None
            
        if model_families[lang] != "gemma":
            model = model.to(device)
        model.eval()
        models[lang] = model
        processors[lang] = processor
        
    # Load Language Identifier
    lid = LanguageIdentifier(target_languages=target_languages)
    vad = VADSegmenter()
    
    # 3. Load all test audio from HuggingFace dataset (streamed to avoid downloading train parquets)
    import datasets
    audio_dict = {}
    for lang in ["lin", "sna", "lug"]:
        logger.info(f"Streaming test split from HF Hub for {lang}...")
        try:
            lang_test = datasets.load_dataset("google/WaxalNLP", name=f"{lang}_asr", split="test", streaming=True)
            lang_test = lang_test.cast_column("audio", datasets.Audio(sampling_rate=16000))
            count = 0
            for ex in lang_test:
                ex_id = ex.get("id") or ex.get("client_id") or ex.get("speaker_id")
                if ex_id:
                    audio_dict[ex_id] = ex["audio"]
                    count += 1
            logger.info(f"Successfully mapped {count} streamed test examples for {lang}")
        except Exception as e:
            logger.warning(f"Failed to stream test split for {lang}: {e}")
            
    # 4. Perform Inference
    predictions = []
    logger.info(f"Starting inference on {len(test_df)} test examples...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        audio_id = row["id"]
        audio_data = audio_dict.get(audio_id)
        
        if audio_data is None:
            logger.warning(f"Could not find audio data in HF test set for ID: {audio_id}. Skipping.")
            predictions.append({"ID": audio_id, "Target": ""})
            continue
            
        try:
            # Get raw audio array and sampling rate
            y = np.asarray(audio_data["array"]).flatten()
            sr = audio_data["sampling_rate"]
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            
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
                if model_families[detected_lang] == "gemma":
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are an assistant that transcribes speech accurately."}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio": chunk.flatten()},
                                {"type": "text", "text": "Please transcribe this audio."},
                            ],
                        }
                    ]
                    chat_prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(
                        text=chat_prompt,
                        audio=chunk.flatten(),
                        return_tensors="pt",
                    ).to(device)
                    
                    if hasattr(model, "dtype"):
                        inputs = {k: v.to(model.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
                        
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            pad_token_id=processor.tokenizer.pad_token_id,
                        )
                    input_len = inputs.input_ids.shape[1]
                    chunk_text = processor.tokenizer.decode(
                        outputs[0][input_len:], skip_special_tokens=True
                    )
                else:
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
