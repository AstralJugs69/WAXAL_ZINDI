#!/usr/bin/env python3
import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
import multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("inference")

# Import helpers from our codebase
from src.inference.pipeline import VADSegmenter
from src.decoding.ctc_decoder import create_ctc_decoder, decode_logits
from src.data.dataset import parse_robust_csv, normalize_text

def get_outputs_dir():
    if os.path.exists("/kaggle/temp/outputs"):
        return "/kaggle/temp/outputs"
    elif os.path.exists("/tmp/outputs"):
        return "/tmp/outputs"
    return "outputs"

def inference_worker(worker_id, gpu_idx, shard_df, target_languages, outputs_dir, output_temp_csv):
    """
    Worker process running inference on a specific GPU for a shard of the dataset.
    """
    import os
    import torch
    import librosa
    import numpy as np
    import datasets
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Gemma3nForConditionalGeneration, AutoProcessor
    from src.inference.pipeline import VADSegmenter
    from src.decoding.ctc_decoder import create_ctc_decoder, decode_logits

    device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Worker {worker_id} starting on device {device} with {len(shard_df)} items...")

    # 1. Load models for target languages onto this GPU
    models = {}
    processors = {}
    decoders = {}
    model_families = {}

    for lang in target_languages:
        custom_gemma_dir = f"{outputs_dir}/{lang}_gemma-3n-E2B-it_fold0/best_model"
        custom_mms_dir = f"{outputs_dir}/{lang}_mms-300m_fold0/best_model"

        if os.path.exists(custom_gemma_dir):
            logger.info(f"Worker {worker_id} loading fine-tuned Gemma model for {lang} from {custom_gemma_dir}...")
            # Map specific GPU for this worker
            model = Gemma3nForConditionalGeneration.from_pretrained(
                custom_gemma_dir,
                torch_dtype=torch.bfloat16,
                device_map={"": device} if device.startswith("cuda") else "cpu"
            )
            processor = AutoProcessor.from_pretrained(custom_gemma_dir)
            model_families[lang] = "gemma"
            decoders[lang] = None
        elif os.path.exists(custom_mms_dir):
            logger.info(f"Worker {worker_id} loading fine-tuned MMS model for {lang} from {custom_mms_dir}...")
            model = Wav2Vec2ForCTC.from_pretrained(custom_mms_dir)
            processor = Wav2Vec2Processor.from_pretrained(custom_mms_dir)
            processor.tokenizer.set_target_lang(lang)
            model_families[lang] = "mms"

            # Check KenLM binary
            lm_path = os.path.join(custom_mms_dir, "lm.bin")
            if os.path.exists(lm_path):
                try:
                    vocab = processor.tokenizer.get_vocab()
                    decoders[lang] = create_ctc_decoder(vocab_dict=vocab, kenlm_model_path=lm_path)
                except Exception as e:
                    logger.warning(f"Worker {worker_id} could not build KenLM decoder for {lang}: {e}")
                    decoders[lang] = None
            else:
                decoders[lang] = None
        else:
            logger.info(f"Worker {worker_id} falling back to pre-trained facebook/mms-1b-all for {lang}...")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang=lang, ignore_mismatched_sizes=True)
            processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all", target_lang=lang)
            processor.tokenizer.set_target_lang(lang)
            model_families[lang] = "mms"
            decoders[lang] = None

        if model_families[lang] != "gemma" and device.startswith("cuda"):
            model = model.to(device)
        model.eval()
        models[lang] = model
        processors[lang] = processor

    # Initialize VAD for this worker
    vad = VADSegmenter()

    # 2. Stream and cache test split for the target languages
    audio_dict = {}
    for lang in target_languages:
        try:
            lang_test = datasets.load_dataset("google/WaxalNLP", name=f"{lang}_asr", split="test", streaming=True)
            lang_test = lang_test.cast_column("audio", datasets.Audio(sampling_rate=16000))
            for ex in lang_test:
                ex_id = ex.get("id") or ex.get("client_id") or ex.get("speaker_id")
                if ex_id:
                    # Contiguous array copy to prevent lazy downloads
                    audio_dict[ex_id] = {
                        "array": np.asarray(ex["audio"]["array"]).copy(),
                        "sampling_rate": ex["audio"]["sampling_rate"]
                    }
        except Exception as e:
            logger.warning(f"Worker {worker_id} failed to stream test split for {lang}: {e}")

    # 3. Perform Inference
    predictions = []
    # Only show progress bar on Worker 0 to keep console clean
    disable_tqdm = (worker_id != 0)
    
    for idx, row in tqdm(shard_df.iterrows(), total=len(shard_df), disable=disable_tqdm, desc=f"Worker {worker_id}"):
        audio_id = row["id"]
        audio_data = audio_dict.get(audio_id)

        if audio_data is None:
            predictions.append({"ID": audio_id, "Target": ""})
            continue

        try:
            y = audio_data["array"]
            sr = audio_data["sampling_rate"]
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)

            # Optimization: Skip VAD for short audio files (duration <= 15s) to avoid 5x generation calls
            duration = len(y) / 16000
            if duration > 15.0:
                chunks = vad.segment(y, sr=16000)
                if not chunks:
                    chunks = [y]
            else:
                chunks = [y]

            # Parse language from ID prefix (e.g. lug_96114 -> lug)
            detected_lang = "lin"
            for possible_lang in ["lin", "sna", "lug"]:
                if audio_id.startswith(possible_lang + "_"):
                    detected_lang = possible_lang
                    break

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
                        max_length=2048,
                    )
                    if device.startswith("cuda"):
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    if hasattr(model, "dtype"):
                        inputs = {k: v.to(model.dtype) if v.is_floating_point() else v for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            pad_token_id=processor.tokenizer.pad_token_id,
                        )
                    input_len = inputs["input_ids"].shape[1]
                    chunk_text = processor.tokenizer.decode(
                        outputs[0][input_len:], skip_special_tokens=True
                    )
                    transcriptions.append(chunk_text)
                else:
                    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                    if device.startswith("cuda"):
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = model(**inputs).logits[0].cpu().numpy()

                    if decoder is not None:
                        chunk_text = decode_logits(decoder, logits, beam_width=128)
                    else:
                        predicted_ids = np.argmax(logits, axis=-1)
                        chunk_text = processor.decode(predicted_ids)
                    transcriptions.append(chunk_text)

            final_text = " ".join(transcriptions)
            predictions.append({"ID": audio_id, "Target": final_text})
        except Exception as e:
            logger.error(f"Worker {worker_id} failed to transcribe {audio_id}: {e}")
            predictions.append({"ID": audio_id, "Target": ""})

    # Save shard results to temporary CSV
    pd.DataFrame(predictions).to_csv(output_temp_csv, index=False)
    logger.info(f"Worker {worker_id} finished. Shard predictions written to {output_temp_csv}")

def main():
    test_csv_path = "Test.csv"
    if not os.path.exists(test_csv_path):
        found = glob.glob("**/Test.csv", recursive=True)
        if found:
            test_csv_path = found[0]

    logger.info(f"Loading test CSV from: {test_csv_path}")
    test_df = parse_robust_csv(test_csv_path)

    target_languages = ["lin", "sna", "lug"]
    outputs_dir = get_outputs_dir()

    # Detect CUDA GPU resources
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Detected {num_gpus} GPUs.")

    # Calculate optimal parallel workers
    # Gemma 3n uses ~9GB VRAM. 
    # If high VRAM is available (like 48GB L40S or 80GB A100 per GPU), we can run multiple workers per GPU!
    workers_per_gpu = 1
    if num_gpus > 0:
        try:
            device = torch.cuda.current_device()
            total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            # If VRAM > 40GB, we run 3 workers per GPU to fully saturate compute cores!
            # If VRAM > 20GB, we run 2 workers per GPU.
            if total_memory_gb > 40.0:
                workers_per_gpu = 3
            elif total_memory_gb > 20.0:
                workers_per_gpu = 2
        except Exception:
            pass

    num_workers = max(1, num_gpus * workers_per_gpu)
    logger.info(f"Spawning {num_workers} parallel workers ({workers_per_gpu} workers per GPU across {num_gpus} GPUs)...")

    # Shard the Test DataFrame
    shards = np.array_split(test_df, num_workers)

    # Spawn worker processes
    mp.set_start_method("spawn", force=True)
    processes = []
    temp_csvs = []

    for worker_id in range(num_workers):
        # Assign worker to a specific GPU index round-robin
        gpu_idx = worker_id % max(1, num_gpus)
        temp_csv = f"submission_worker_{worker_id}.csv"
        temp_csvs.append(temp_csv)

        p = mp.Process(
            target=inference_worker,
            args=(worker_id, gpu_idx, shards[worker_id], target_languages, outputs_dir, temp_csv)
        )
        p.start()
        processes.append(p)

    logger.info("Waiting for all parallel workers to complete...")
    for p in processes:
        p.join()

    # Consolidate predictions in the exact original order of Test.csv
    logger.info("Consolidating parallel predictions...")
    prediction_maps = {}
    for temp_csv in temp_csvs:
        if os.path.exists(temp_csv):
            try:
                df_temp = pd.read_csv(temp_csv)
                for _, r in df_temp.iterrows():
                    prediction_maps[r["ID"]] = r["Target"]
                # Cleanup temp file
                os.remove(temp_csv)
            except Exception as e:
                logger.error(f"Error reading temporary file {temp_csv}: {e}")

    # Build final DataFrame in original Test.csv ID sequence
    final_predictions = []
    for _, row in test_df.iterrows():
        audio_id = row["id"]
        # Default to empty if missing
        target_text = prediction_maps.get(audio_id, "")
        if pd.isna(target_text):
            target_text = ""
        final_predictions.append({"ID": audio_id, "Target": target_text})

    submission_df = pd.DataFrame(final_predictions)
    submission_df.to_csv("submission.csv", index=False)
    logger.info("Successfully generated submission.csv!")

if __name__ == "__main__":
    main()
