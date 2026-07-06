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

def get_outputs_dir():
    if os.path.exists("/kaggle/temp/outputs"):
        return "/kaggle/temp/outputs"
    elif os.path.exists("/tmp/outputs"):
        return "/tmp/outputs"
    return "outputs"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("inference")

# Import VAD Segmenter from codebase
from src.inference.pipeline import VADSegmenter
from src.decoding.ctc_decoder import create_ctc_decoder, decode_logits
from src.data.dataset import parse_robust_csv, normalize_text

def worker_inference(worker_id, num_gpus, target_languages, test_ids_shard, audio_dict_shard, return_dict):
    """
    Worker function executed in a separate process.
    Loads models on the assigned GPU and transcribes its shard of test audios.
    """
    import logging
    # Set up child logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    
    import torch
    import librosa
    import numpy as np
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from transformers import Gemma3nForConditionalGeneration, AutoProcessor
    from src.decoding.ctc_decoder import create_ctc_decoder, decode_logits
    from src.inference.pipeline import VADSegmenter
    from src.data.dataset import normalize_text
    
    # 1. Assign GPU device
    device_id = worker_id % num_gpus if num_gpus > 0 else 0
    device = torch.device(f"cuda:{device_id}" if num_gpus > 0 else "cpu")
    
    # Set PyTorch/XLA or device maps
    device_map = {"": device_id} if num_gpus > 0 else "cpu"
    
    worker_logger.info(f"[Worker {worker_id}] Initializing models on {device}...")
    
    models = {}
    processors = {}
    model_families = {}
    decoders = {}
    
    # Load models
    for lang in target_languages:
        custom_gemma_dir = f"{get_outputs_dir()}/{lang}_gemma-3n-E2B-it_fold0/best_model"
        custom_mms_dir = f"{get_outputs_dir()}/{lang}_mms-300m_fold0/best_model"
        
        if os.path.exists(custom_gemma_dir):
            model = Gemma3nForConditionalGeneration.from_pretrained(
                custom_gemma_dir,
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
            processor = AutoProcessor.from_pretrained(custom_gemma_dir)
            model_families[lang] = "gemma"
            decoders[lang] = None
        elif os.path.exists(custom_mms_dir):
            model = Wav2Vec2ForCTC.from_pretrained(custom_mms_dir).to(device)
            processor = Wav2Vec2Processor.from_pretrained(custom_mms_dir)
            processor.tokenizer.set_target_lang(lang)
            model_families[lang] = "mms"
            
            # Check KenLM binary
            lm_path = os.path.join(custom_mms_dir, "lm.bin")
            if os.path.exists(lm_path):
                try:
                    vocab = processor.tokenizer.get_vocab()
                    decoders[lang] = create_ctc_decoder(vocab_dict=vocab, kenlm_model_path=lm_path)
                except Exception:
                    decoders[lang] = None
            else:
                decoders[lang] = None
        else:
            # Fallback to pre-trained mms-1b-all for this language
            model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang=lang, ignore_mismatched_sizes=True).to(device)
            processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all", target_lang=lang)
            processor.tokenizer.set_target_lang(lang)
            model_families[lang] = "mms"
            decoders[lang] = None
            
        model.eval()
        models[lang] = model
        processors[lang] = processor
        
    vad = VADSegmenter()
    predictions = {}
    
    worker_logger.info(f"[Worker {worker_id}] Starting inference on {len(test_ids_shard)} examples...")
    
    for audio_id in test_ids_shard:
        audio_data = audio_dict_shard.get(audio_id)
        if audio_data is None:
            predictions[audio_id] = ""
            continue
            
        try:
            y = audio_data["array"]
            sr = audio_data["sampling_rate"]
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                
            # Segment audio via VAD to prevent OOM
            chunks = vad.segment(y, sr=16000)
            
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
                    # Move inputs to assigned device and dtype
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
                else:
                    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        logits = model(**inputs).logits[0].cpu().numpy()
                        
                    if decoder is not None:
                        chunk_text = decode_logits(decoder, logits, beam_width=128)
                    else:
                        pred_ids = np.argmax(logits, axis=-1)
                        chunk_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                        
                normalized_chunk = normalize_text(chunk_text)
                if normalized_chunk:
                    transcriptions.append(normalized_chunk)
                    
            predictions[audio_id] = " ".join(transcriptions)
        except Exception as e:
            worker_logger.error(f"[Worker {worker_id}] Failed to transcribe {audio_id}: {e}")
            predictions[audio_id] = ""
            
    return_dict[worker_id] = predictions
    worker_logger.info(f"[Worker {worker_id}] Finished shard.")

def main():
    logger.info("Initializing high-performance parallel inference pipeline...")
    
    # 1. Resolve paths
    test_csv_path = "Test.csv"
    if not os.path.exists(test_csv_path):
        found = glob.glob("**/Test.csv", recursive=True)
        if found:
            test_csv_path = found[0]
            
    logger.info(f"Loading test CSV from: {test_csv_path}")
    test_df = parse_robust_csv(test_csv_path)
    
    # Extract test IDs as a list
    test_ids = list(test_df["id"].dropna())
    
    # 2. Define target languages
    target_languages = ["lin", "sna", "lug"]
    
    # 3. Load all test audio from HuggingFace dataset (streamed and eagerly cached in RAM to avoid lazy downloads during inference)
    import datasets
    audio_dict = {}
    for lang in ["lin", "sna", "lug"]:
        logger.info(f"Streaming and caching test split from HF Hub for {lang}...")
        try:
            lang_test = datasets.load_dataset("google/WaxalNLP", name=f"{lang}_asr", split="test", streaming=True)
            lang_test = lang_test.cast_column("audio", datasets.Audio(sampling_rate=16000))
            count = 0
            for ex in lang_test:
                ex_id = ex.get("id") or ex.get("client_id") or ex.get("speaker_id")
                if ex_id:
                    # Contiguous array copy to release file descriptors and prevent memory leaks
                    audio_dict[ex_id] = {
                        "array": np.asarray(ex["audio"]["array"]).copy(),
                        "sampling_rate": ex["audio"]["sampling_rate"]
                    }
                    count += 1
            logger.info(f"Successfully cached {count} test examples in RAM for {lang}")
        except Exception as e:
            logger.warning(f"Failed to cache test split for {lang}: {e}")

    # Determine GPU counts and configure parallel processes
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        # Launch 4 parallel processes per GPU to saturate CUDA Tensor Cores without OOMing
        num_workers = num_gpus * 4
        logger.info(f"Detected {num_gpus} GPUs. Launching {num_workers} parallel workers (4 per GPU)...")
    else:
        num_workers = 2
        logger.info(f"No GPUs detected. Launching {num_workers} CPU-bound workers...")

    # Split test IDs and cache dictionaries into equal shards for parallel workers
    shards = [[] for _ in range(num_workers)]
    for idx, audio_id in enumerate(test_ids):
        shards[idx % num_workers].append(audio_id)
        
    sharded_audio_dicts = [{} for _ in range(num_workers)]
    for w_id in range(num_workers):
        for audio_id in shards[w_id]:
            if audio_id in audio_dict:
                sharded_audio_dicts[w_id][audio_id] = audio_dict[audio_id]

    # Initialize multiprocessing with spawn context (critical for CUDA compatibility)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    processes = []

    logger.info("Spawning parallel inference worker processes...")
    for w_id in range(num_workers):
        p = ctx.Process(
            target=worker_inference,
            args=(w_id, num_gpus, target_languages, shards[w_id], sharded_audio_dicts[w_id], return_dict)
        )
        processes.append(p)
        p.start()

    logger.info("All worker processes spawned. Monitoring progress...")
    # Monitor completion of subprocesses
    for p in processes:
        p.join()

    logger.info("All parallel workers completed. Compiling final predictions...")
    
    # 4. Compile predictions in the original order of Test.csv
    predictions = []
    for idx, row in test_df.iterrows():
        audio_id = row["id"]
        transcription = ""
        for w_id in range(num_workers):
            if return_dict.get(w_id) and audio_id in return_dict[w_id]:
                transcription = return_dict[w_id][audio_id]
                break
        predictions.append({"ID": audio_id, "Target": transcription})
        
    # 5. Save to submission CSV
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv("submission.csv", index=False)
    logger.info("Successfully generated submission.csv in the correct format!")

if __name__ == "__main__":
    main()
