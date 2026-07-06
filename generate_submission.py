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

GEMMA_MODEL_ID = "google/gemma-3n-E2B-it"
GEMMA_SYSTEM_MESSAGE = "You are an assistant that transcribes speech accurately."
GEMMA_USER_MESSAGE = "Please transcribe this audio."
GEMMA_MAX_INPUT_TOKENS = 2048
NO_AUDIO_REFUSAL_MARKERS = (
    "no audio provided",
    "please provide the audio",
    "provide the audio",
    "cannot transcribe",
    "can't transcribe",
    "unable to transcribe",
)

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


def _as_mono_float32_audio(audio):
    """
    Converts dataset audio/chunks to the mono float32 arrays expected by Gemma 3n.
    Invalid or empty audio returns None so callers can skip it deterministically.
    """
    if audio is None:
        return None
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        if arr.shape[0] < arr.shape[-1]:
            arr = arr[0]
        else:
            arr = arr[:, 0]
    arr = np.ascontiguousarray(arr.flatten())
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def _is_no_audio_refusal(text):
    normalized = normalize_text(text)
    return any(marker in normalized for marker in NO_AUDIO_REFUSAL_MARKERS)


def _move_inputs_to_device(inputs, device, dtype=None):
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if dtype is not None and value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _has_audio_features(inputs):
    audio_feature_keys = {
        "input_features",
        "input_features_mask",
        "input_values",
        "audio_features",
        "audio_attention_mask",
    }
    return any(key in inputs for key in audio_feature_keys)


def _prepare_gemma_inputs(processor, messages, audio, worker_logger=None):
    """
    Prefer Gemma 3n's processor-level chat template. It inserts the audio soft
    tokens and builds audio features in one step, matching the current HF docs.
    A fallback keeps compatibility with older processor versions used by the
    original notebook.
    """
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            max_length=GEMMA_MAX_INPUT_TOKENS,
        )
        if _has_audio_features(inputs):
            return inputs
        if worker_logger is not None:
            worker_logger.debug("Gemma processor chat template returned no audio features; falling back.")
    except (TypeError, ValueError, AttributeError) as exc:
        if worker_logger is not None:
            worker_logger.debug(f"Falling back to manual Gemma processor call: {exc}")

    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=chat_prompt,
        audio=[audio],
        return_tensors="pt",
        padding=False,
        max_length=GEMMA_MAX_INPUT_TOKENS,
    )


def _load_gemma_model_and_processor(checkpoint_dir, device_map, hf_token, worker_logger):
    """
    Loads either a PEFT adapter checkpoint (the normal SFTTrainer output) or a
    legacy full-model checkpoint. Avoid strict=False adapter guessing, which can
    silently leave inference running on the unfine-tuned base Gemma model.
    """
    from transformers import Gemma3nForConditionalGeneration, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(checkpoint_dir, token=hf_token)
    except Exception as exc:
        worker_logger.warning(
            f"Could not load Gemma processor from {checkpoint_dir} ({exc}); "
            f"falling back to {GEMMA_MODEL_ID}."
        )
        processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, token=hf_token)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        from peft import PeftModel

        base_model = Gemma3nForConditionalGeneration.from_pretrained(
            GEMMA_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            token=hf_token,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=False)
        worker_logger.info(f"Loaded Gemma PEFT adapter checkpoint from {checkpoint_dir}")
        return model, processor

    try:
        model = Gemma3nForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            token=hf_token,
        )
        worker_logger.info(f"Loaded Gemma full-model checkpoint from {checkpoint_dir}")
        return model, processor
    except Exception as exc:
        worker_logger.warning(
            f"Could not load {checkpoint_dir} as a full Gemma checkpoint ({exc}). "
            "Trying legacy state-dict loading."
        )

    from peft import get_peft_model, LoraConfig
    from safetensors.torch import load_file

    base_model = Gemma3nForConditionalGeneration.from_pretrained(
        GEMMA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        token=hf_token,
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)

    weight_paths = [
        os.path.join(checkpoint_dir, "adapter_model.safetensors"),
        os.path.join(checkpoint_dir, "model.safetensors"),
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
    ]
    weight_path = next((path for path in weight_paths if os.path.exists(path)), None)
    if weight_path is None:
        raise FileNotFoundError(f"Could not find Gemma weights in {checkpoint_dir}")

    if weight_path.endswith(".safetensors"):
        state_dict = load_file(weight_path)
    else:
        state_dict = torch.load(weight_path, map_location="cpu")

    load_result = model.load_state_dict(state_dict, strict=False)
    loaded_lora_keys = sum(1 for key in state_dict if "lora_" in key)
    if loaded_lora_keys == 0:
        worker_logger.warning(f"Legacy Gemma checkpoint {weight_path} did not contain LoRA keys.")
    worker_logger.info(
        f"Loaded legacy Gemma state dict from {weight_path}; "
        f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
    )
    return model, processor

def worker_inference(worker_id, num_gpus, target_languages, test_ids_shard, audio_dict_shard, return_dict, hf_token):
    """
    Worker function executed in a separate process.
    Loads models on the assigned GPU and transcribes its shard of test audios.
    """
    import os
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    # Set default HF home to persist across Restarts and load cache
    os.environ["HF_HOME"] = "/teamspace/studios/this_studio/hf_home"
    os.environ["HF_HUB_CACHE"] = "/teamspace/studios/this_studio/hf_home/hub"
    os.environ["HF_DATASETS_CACHE"] = "/teamspace/studios/this_studio/hf_home/datasets"

    import logging
    # Set up child logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    
    import torch
    import librosa
    import numpy as np
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
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
            model, processor = _load_gemma_model_and_processor(
                custom_gemma_dir,
                device_map,
                hf_token,
                worker_logger,
            )
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
    
    processed_count = 0
    for audio_id in test_ids_shard:
        processed_count += 1
            
        audio_data = audio_dict_shard.get(audio_id)
        if audio_data is None:
            predictions[audio_id] = ""
            continue
            
        try:
            y = _as_mono_float32_audio(audio_data["array"])
            sr = audio_data["sampling_rate"]
            if y is None or sr is None:
                worker_logger.warning(f"[Worker {worker_id}] Missing/invalid audio for {audio_id}")
                predictions[audio_id] = ""
                continue
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                y = _as_mono_float32_audio(y)
                sr = 16000
                if y is None:
                    worker_logger.warning(f"[Worker {worker_id}] Resampled audio became invalid for {audio_id}")
                    predictions[audio_id] = ""
                    continue
                
            # Skip VAD segmenting for short audios (<20s) to speed up CPU preprocessing
            duration = len(y) / sr
            if duration <= 20.0:
                chunks = [y]
            else:
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
                chunk = _as_mono_float32_audio(chunk)
                if chunk is None:
                    worker_logger.warning(f"[Worker {worker_id}] Skipping invalid chunk for {audio_id}")
                    continue

                if model_families[detected_lang] == "gemma":
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
                    inputs = _prepare_gemma_inputs(processor, messages, chunk, worker_logger)
                    # Move inputs to assigned device and dtype
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
                            max_new_tokens=128,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            use_cache=True,
                            do_sample=False,
                        )
                    input_len = inputs["input_ids"].shape[1]
                    prompt_str = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
                    full_gen_str = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
                    chunk_text = processor.tokenizer.decode(
                        outputs[0][input_len:], skip_special_tokens=True
                    )
                    if processed_count <= 5:
                        worker_logger.info(f"[Worker {worker_id}] DEBUG Prompt: '{prompt_str}'")
                        worker_logger.info(f"[Worker {worker_id}] DEBUG Full Gen: '{full_gen_str}'")
                        worker_logger.info(f"[Worker {worker_id}] DEBUG Sliced Text: '{chunk_text}'")
                    if _is_no_audio_refusal(chunk_text):
                        worker_logger.warning(
                            f"[Worker {worker_id}] Dropping Gemma no-audio refusal for {audio_id}: '{chunk_text}'"
                        )
                        chunk_text = ""
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
                    
            final_text = " ".join(transcriptions)
            predictions[audio_id] = final_text
            worker_logger.info(f"[Worker {worker_id}] Transcribed {audio_id} ({(processed_count/len(test_ids_shard)):.1%}): '{final_text}'")
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
        # Launch 2 parallel processes per GPU to avoid maxing out VRAM
        num_workers = num_gpus * 2
        logger.info(f"Detected {num_gpus} GPUs. Launching {num_workers} parallel workers (2 per GPU)...")
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

    import sys
    hf_token = os.environ.get("HF_TOKEN")
    if "--hf_token" in sys.argv:
        idx = sys.argv.index("--hf_token")
        if idx + 1 < len(sys.argv):
            hf_token = sys.argv[idx + 1]
    logger.info("Spawning parallel inference worker processes...")
    for w_id in range(num_workers):
        p = ctx.Process(
            target=worker_inference,
            args=(w_id, num_gpus, target_languages, shards[w_id], sharded_audio_dicts[w_id], return_dict, hf_token)
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
