#!/usr/bin/env python3
import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path

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
    outputs_dir = os.environ.get("WAXAL_OUTPUTS_DIR")
    if outputs_dir:
        path = Path(outputs_dir).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def configure_hf_cache():
    """
    Point HF cache at a writable location without hardcoding Lightning paths.
    Order: existing env → Kaggle temp → local project → Lightning (only if present).
    """
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]

    candidates = []
    if os.path.exists("/kaggle/temp"):
        candidates.append("/kaggle/temp/hf_home")
    if os.path.exists("/kaggle/working"):
        candidates.append("/tmp/hf_home")
    lightning_home = "/teamspace/studios/this_studio/hf_home"
    if os.path.exists("/teamspace/studios/this_studio"):
        candidates.append(lightning_home)
    candidates.append(str(Path(__file__).resolve().parent / "hf_home"))

    hf_home = candidates[0]
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    return hf_home


def canonical_example_id(example):
    """
    Map a HF example to a Zindi submission ID.
    Never fall back to speaker_id — that caused silent blank predictions.
    """
    if not isinstance(example, dict):
        return None
    for key in ("id", "client_id"):
        value = example.get(key)
        if value is None:
            continue
        audio_id = str(value).strip()
        if audio_id:
            return audio_id
    return None

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


def _is_english_hallucination(text):
    normalized = normalize_text(text)
    words = set(normalized.split())
    english_stopwords = {
        "the", "of", "and", "is", "it", "to", "in", "that", "was", "for", 
        "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", 
        "have", "from", "or", "one", "had", "by", "word", "but", "not", 
        "what", "all", "were", "we", "when", "your", "can", "said", "there", 
        "use", "an", "each", "which", "she", "do", "how", "their", "if", 
        "will", "up", "other", "about", "out", "many", "then", "them", 
        "these", "so", "some", "her", "would", "make", "like", "him", 
        "into", "has", "look", "two", "more", "write", "go", "see", 
        "number", "no", "way", "could", "people", "my", "than", "first", 
        "water", "been", "call", "who", "oil", "its", "now", "find"
    }
    matched = words.intersection(english_stopwords)
    return len(matched) >= 4


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


def _find_gemma_weight_path(checkpoint_dir):
    weight_paths = [
        os.path.join(checkpoint_dir, "adapter_model.safetensors"),
        os.path.join(checkpoint_dir, "model.safetensors"),
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
    ]
    return next((path for path in weight_paths if os.path.exists(path)), None)


def _load_state_dict_file(weight_path):
    if weight_path is None:
        return None
    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(weight_path)
    try:
        return torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(weight_path, map_location="cpu")


def _looks_like_peft_state_dict(state_dict):
    if not state_dict:
        return False
    return any("lora_" in key or ".base_layer." in key for key in state_dict)


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
            processor_kwargs={
                "text_kwargs": {
                    "max_length": GEMMA_MAX_INPUT_TOKENS,
                },
                "audio_kwargs": {
                    "max_length": 400000,
                    "truncation": False,
                },
            },
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
    try:
        return processor(
            text=chat_prompt,
            audio=[audio],
            return_tensors="pt",
            padding=False,
            text_kwargs={"max_length": GEMMA_MAX_INPUT_TOKENS},
            audio_kwargs={"max_length": 400000, "truncation": False},
        )
    except TypeError:
        return processor(
            text=chat_prompt,
            audio=[audio],
            return_tensors="pt",
            padding=False,
            max_length=GEMMA_MAX_INPUT_TOKENS,
        )


def _build_gemma_peft_model(device_map, hf_token):
    from peft import get_peft_model, LoraConfig
    from transformers import Gemma3nForConditionalGeneration

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
    return get_peft_model(base_model, peft_config)


def _with_base_model_prefix(state_dict):
    prefixed = {}
    for key, value in state_dict.items():
        prefixed[key if key.startswith("base_model.") else f"base_model.{key}"] = value
    return prefixed


def _load_legacy_gemma_lora_checkpoint(weight_path, device_map, hf_token, worker_logger):
    state_dict = _load_state_dict_file(weight_path)
    if not _looks_like_peft_state_dict(state_dict):
        return None

    model = _build_gemma_peft_model(device_map, hf_token)
    lora_key_count = sum(1 for key in state_dict if "lora_" in key)
    candidates = [
        ("as-saved", state_dict),
        ("base_model-prefixed", _with_base_model_prefix(state_dict)),
    ]

    best = None
    for candidate_name, candidate_state in candidates:
        result = model.load_state_dict(candidate_state, strict=False)
        unexpected_lora = [key for key in result.unexpected_keys if "lora_" in key]
        missing_lora = [key for key in result.missing_keys if "lora_" in key]
        matched_lora = sum(1 for key in candidate_state if "lora_" in key) - len(unexpected_lora)
        score = (matched_lora, -len(missing_lora), -len(unexpected_lora))
        if best is None or score > best[0]:
            best = (score, candidate_name, candidate_state, result, matched_lora, missing_lora, unexpected_lora)

    _, candidate_name, candidate_state, result, matched_lora, missing_lora, unexpected_lora = best
    # Reload the winning mapping because trial loads above may partially mutate the model.
    result = model.load_state_dict(candidate_state, strict=False)
    matched_lora = sum(1 for key in candidate_state if "lora_" in key) - len(
        [key for key in result.unexpected_keys if "lora_" in key]
    )
    if lora_key_count and matched_lora <= 0:
        raise RuntimeError(
            f"Found {lora_key_count} LoRA keys in {weight_path}, but none matched the reconstructed PEFT model."
        )

    worker_logger.info(
        f"Loaded legacy Gemma LoRA checkpoint from {weight_path} using {candidate_name}; "
        f"matched_lora={matched_lora}/{lora_key_count}, "
        f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
    )
    return model


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

    weight_path = _find_gemma_weight_path(checkpoint_dir)

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

    if weight_path is not None:
        legacy_lora_model = _load_legacy_gemma_lora_checkpoint(weight_path, device_map, hf_token, worker_logger)
        if legacy_lora_model is not None:
            return legacy_lora_model, processor

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

    if weight_path is None:
        raise FileNotFoundError(f"Could not find Gemma weights in {checkpoint_dir}")

    model = _build_gemma_peft_model(device_map, hf_token)
    state_dict = _load_state_dict_file(weight_path)

    aligned_state_dict = _with_base_model_prefix(state_dict)
    load_result = model.load_state_dict(aligned_state_dict, strict=False)
    loaded_lora_keys = sum(1 for key in aligned_state_dict if "lora_" in key)
    if loaded_lora_keys == 0:
        worker_logger.warning(f"Legacy Gemma checkpoint {weight_path} did not contain LoRA keys.")
    worker_logger.info(
        f"Loaded legacy Gemma state dict from {weight_path}; "
        f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
    )
    return model, processor


def _read_expected_submission_ids(test_csv_path):
    sample_paths = [
        "SampleSubmission.csv",
        os.path.join(os.path.dirname(test_csv_path), "SampleSubmission.csv"),
    ]
    sample_path = next((path for path in sample_paths if os.path.exists(path)), None)
    source_path = sample_path or test_csv_path

    df = pd.read_csv(source_path, dtype=str)
    id_col = "ID" if "ID" in df.columns else "id" if "id" in df.columns else df.columns[0]
    ids = df[id_col].dropna().astype(str).str.strip().tolist()

    seen = set()
    duplicates = []
    for audio_id in ids:
        if audio_id in seen:
            duplicates.append(audio_id)
        seen.add(audio_id)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"{source_path} contains duplicate IDs: {preview}")

    logger.info(f"Loaded {len(ids)} expected submission IDs from {source_path}")
    return ids


def _merge_worker_predictions(worker_results, expected_ids=None):
    # Seed every expected ID so a crashed worker cannot drop rows from the merge.
    combined_predictions = {str(i).strip(): "" for i in (expected_ids or [])}
    duplicate_predictions = []

    for worker_id, predictions in worker_results.items():
        if not predictions:
            logger.warning(f"Worker {worker_id} returned no predictions.")
            continue
        for audio_id, transcription in dict(predictions).items():
            audio_id = str(audio_id).strip()
            text = "" if transcription is None else str(transcription)
            if audio_id in combined_predictions and combined_predictions[audio_id].strip():
                # Keep the first non-empty prediction; overwrite only if current is blank.
                if text.strip():
                    duplicate_predictions.append(audio_id)
                continue
            if audio_id in combined_predictions and not combined_predictions[audio_id].strip() and text.strip():
                combined_predictions[audio_id] = text
                continue
            if audio_id in combined_predictions and not text.strip():
                continue
            combined_predictions[audio_id] = text

    if duplicate_predictions:
        logger.warning(
            f"Observed duplicate worker predictions for {len(set(duplicate_predictions))} IDs. "
            f"First few: {sorted(set(duplicate_predictions))[:10]}"
        )
    return combined_predictions


def _sanitize_target(text) -> str:
    """
    Clean a prediction for CSV / Zindi.

    CRITICAL: empty Target cells become NaN when Zindi does pd.read_csv(...).
    Many Zindi metrics then report those rows as
      "Missing entries for IDs …"
    even though the ID row was present in the file. Never emit a truly empty field.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    # collapse newlines / control chars that can break CSV row alignment
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _write_validated_submission(expected_ids, combined_predictions, output_path="submission.csv", max_blank_frac=0.02):
    import csv

    expected_ids = [str(i).strip() for i in expected_ids]
    expected_set = set(expected_ids)
    if len(expected_ids) != len(expected_set):
        raise RuntimeError("expected_ids contains duplicates — fix SampleSubmission/Test.csv first.")

    extra_ids = sorted(set(str(k).strip() for k in combined_predictions) - expected_set)
    if extra_ids:
        logger.warning(
            f"Ignoring {len(extra_ids)} predictions for IDs not present in expected submission. "
            f"First few: {extra_ids[:10]}"
        )

    # Always emit ONE row per expected ID (SampleSubmission order). Never drop rows.
    raw_targets = []
    blank_ids = []
    for audio_id in expected_ids:
        cleaned = _sanitize_target(combined_predictions.get(audio_id, ""))
        if not cleaned:
            blank_ids.append(audio_id)
            # Non-empty placeholder so Zindi's read_csv does not turn Target into NaN.
            # A single space is ignored by most WER normalizers after strip, but keeps the cell present.
            cleaned = " "
        raw_targets.append(cleaned)

    submission_df = pd.DataFrame(
        {"ID": expected_ids, "Target": raw_targets},
        columns=["ID", "Target"],
    )

    if len(submission_df) != len(expected_ids):
        raise RuntimeError(f"Submission row count mismatch: got {len(submission_df)}, expected {len(expected_ids)}")
    if submission_df["ID"].isna().any():
        raise RuntimeError("Submission contains null IDs.")
    if submission_df["ID"].duplicated().any():
        dupes = submission_df.loc[submission_df["ID"].duplicated(), "ID"].head(10).tolist()
        raise RuntimeError(f"Submission contains duplicate IDs: {dupes}")
    if set(submission_df["ID"].astype(str)) != expected_set:
        missing = sorted(expected_set - set(submission_df["ID"].astype(str)))
        raise RuntimeError(f"Submission is missing expected IDs: {missing[:10]}")

    # Count "real" blanks (whitespace-only placeholders after strip)
    effective_blank_mask = submission_df["Target"].astype(str).str.strip().eq("")
    blank_count = int(effective_blank_mask.sum())
    blank_frac = blank_count / max(len(submission_df), 1)
    blank_ids = submission_df.loc[effective_blank_mask, "ID"].tolist()
    if blank_count:
        logger.warning(
            f"Submission contains {blank_count} blank/placeholder transcriptions "
            f"({blank_frac:.1%}). First few: {blank_ids[:10]}"
        )
    if blank_frac > max_blank_frac:
        raise RuntimeError(
            f"Blank transcription rate {blank_frac:.1%} exceeds limit {max_blank_frac:.1%}. "
            f"Fix model/audio loading before submitting. Blank IDs sample: {blank_ids[:20]}"
        )

    # Defensive: every Target must be non-empty so Zindi never sees NaN after read_csv
    empty_cells = submission_df["Target"].astype(str).eq("")
    if empty_cells.any():
        submission_df.loc[empty_cells, "Target"] = " "

    temp_path = f"{output_path}.tmp"
    # QUOTE_ALL + explicit UTF-8 avoids Excel/parser edge cases on special orthography
    submission_df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
        na_rep=" ",
    )
    os.replace(temp_path, output_path)

    # Round-trip verify the way Zindi typically loads files (default read_csv)
    verify = pd.read_csv(output_path, dtype=str)
    if list(verify.columns)[:2] != ["ID", "Target"]:
        raise RuntimeError(f"Written submission has unexpected columns: {list(verify.columns)}")
    if len(verify) != len(expected_ids):
        raise RuntimeError(
            f"Round-trip row count mismatch: wrote {len(expected_ids)}, re-read {len(verify)}"
        )
    verify_ids = set(verify["ID"].astype(str).str.strip())
    missing_rt = sorted(expected_set - verify_ids)
    if missing_rt:
        raise RuntimeError(f"Round-trip missing IDs after write: {missing_rt[:10]}")
    # NaN Targets after default read_csv == Zindi "Missing entries"
    nan_targets = verify["Target"].isna() | (verify["Target"].astype(str).str.lower() == "nan")
    if nan_targets.any():
        bad = verify.loc[nan_targets, "ID"].head(20).tolist()
        raise RuntimeError(
            f"Round-trip found NaN/empty Targets (Zindi reports these as Missing entries): {bad}"
        )

    logger.info(
        f"Validated submission written to {output_path}: {len(submission_df)} rows, "
        f"{blank_count} placeholders, 0 missing IDs (round-trip OK)."
    )
    return blank_ids


def worker_inference(
    worker_id,
    num_gpus,
    target_languages,
    test_ids_shard,
    audio_dict_shard,
    return_dict,
    hf_token,
    prefer_gemma=False,
):
    """
    Worker function executed in a separate process.
    Loads models on the assigned GPU and transcribes its shard of test audios.
    Prefers per-language fine-tuned MMS unless prefer_gemma=True.
    """
    import os
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    # Keep any parent HF cache settings; only configure if unset.
    configure_hf_cache()

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
    
    # Load models — MMS fine-tunes first (primary strategy); Gemma only if preferred/available.
    for lang in target_languages:
        custom_gemma_dir = f"{get_outputs_dir()}/{lang}_gemma-3n-E2B-it_fold0/best_model"
        custom_mms_dir = f"{get_outputs_dir()}/{lang}_mms-300m_fold0/best_model"
        has_mms = os.path.exists(custom_mms_dir)
        has_gemma = os.path.exists(custom_gemma_dir)

        use_gemma = prefer_gemma and has_gemma
        use_mms = (not use_gemma) and has_mms
        if not use_mms and not use_gemma and has_gemma:
            # Only fall back to Gemma when no MMS checkpoint exists.
            use_gemma = True

        if use_mms:
            model = Wav2Vec2ForCTC.from_pretrained(custom_mms_dir).to(device)
            processor = Wav2Vec2Processor.from_pretrained(custom_mms_dir)
            try:
                processor.tokenizer.set_target_lang(lang)
            except Exception:
                pass
            model_families[lang] = "mms"
            worker_logger.info(f"[Worker {worker_id}] Loaded fine-tuned MMS for {lang} from {custom_mms_dir}")

            lm_path = os.path.join(custom_mms_dir, "lm.bin")
            if os.path.exists(lm_path):
                try:
                    vocab = processor.tokenizer.get_vocab()
                    decoders[lang] = create_ctc_decoder(vocab_dict=vocab, kenlm_model_path=lm_path)
                except Exception as exc:
                    worker_logger.warning(f"[Worker {worker_id}] KenLM load failed for {lang}: {exc}")
                    decoders[lang] = None
            else:
                decoders[lang] = None
        elif use_gemma:
            model, processor = _load_gemma_model_and_processor(
                custom_gemma_dir,
                device_map,
                hf_token,
                worker_logger,
            )
            model_families[lang] = "gemma"
            decoders[lang] = None
            worker_logger.info(f"[Worker {worker_id}] Loaded Gemma for {lang} from {custom_gemma_dir}")
        else:
            # Fallback to pre-trained mms-1b-all for this language
            model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/mms-1b-all", target_lang=lang, ignore_mismatched_sizes=True
            ).to(device)
            processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all", target_lang=lang)
            processor.tokenizer.set_target_lang(lang)
            model_families[lang] = "mms"
            decoders[lang] = None
            worker_logger.warning(
                f"[Worker {worker_id}] No fine-tuned checkpoint for {lang}; using base mms-1b-all"
            )
            
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
                
            # Limit chunk size to max 20 seconds (320000 samples at 16kHz)
            # to prevent integer overflow and match model training limits
            MAX_CHUNK_SAMPLES = 20 * 16000
            safe_chunks = []
            for chunk in chunks:
                if len(chunk) > MAX_CHUNK_SAMPLES:
                    for start in range(0, len(chunk), MAX_CHUNK_SAMPLES):
                        sub_chunk = chunk[start:start+MAX_CHUNK_SAMPLES]
                        if len(sub_chunk) >= 16000 * 0.5: # skip tiny fragments < 0.5s
                            safe_chunks.append(sub_chunk)
                else:
                    safe_chunks.append(chunk)
            chunks = safe_chunks
            
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
                            max_new_tokens=96,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            use_cache=True,
                            do_sample=False,
                            repetition_penalty=1.15,
                            no_repeat_ngram_size=4,
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
                    elif _is_english_hallucination(chunk_text):
                        worker_logger.warning(
                            f"[Worker {worker_id}] Dropping Gemma English hallucination for {audio_id}: '{chunk_text}'"
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

def _load_test_audio_dict(test_ids):
    """
    Load HF test splits and cache audio keyed only by canonical example IDs.

    Prefer non-streaming load (full ID coverage); fall back to streaming if RAM fails.
    """
    import datasets

    expected = set(str(i).strip() for i in test_ids)
    audio_dict = {}

    for lang in ["lin", "sna", "lug"]:
        logger.info(f"Loading and caching test split from HF Hub for {lang}...")
        lang_ds = None
        # Non-streaming is more reliable for full coverage; streaming as fallback.
        for streaming in (False, True):
            try:
                lang_ds = datasets.load_dataset(
                    "google/WaxalNLP",
                    name=f"{lang}_asr",
                    split="test",
                    streaming=streaming,
                )
                lang_ds = lang_ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
                mode = "streaming" if streaming else "map-style"
                logger.info(f"  {lang}: loaded test split ({mode})")
                break
            except Exception as e:
                logger.warning(f"  {lang}: load failed (streaming={streaming}): {e}")
                lang_ds = None
        if lang_ds is None:
            logger.warning(f"Failed to cache test split for {lang}")
            continue

        count = 0
        try:
            iterator = lang_ds if hasattr(lang_ds, "__iter__") else iter(lang_ds)
            for ex in iterator:
                if not isinstance(ex, dict):
                    # datasets row object
                    try:
                        ex = dict(ex)
                    except Exception:
                        continue
                ex_id = canonical_example_id(ex)
                if not ex_id:
                    continue
                ex_id = str(ex_id).strip()
                # Keep any expected ID; also keep lang-prefixed IDs that match this split
                if ex_id not in expected and not ex_id.startswith(f"{lang}_"):
                    continue
                audio = ex.get("audio") or {}
                arr = audio.get("array") if isinstance(audio, dict) else None
                sr = audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000
                if arr is None:
                    continue
                audio_dict[ex_id] = {
                    "array": np.asarray(arr).copy(),
                    "sampling_rate": sr,
                }
                count += 1
            logger.info(f"Successfully cached {count} test examples in RAM for {lang}")
        except Exception as e:
            logger.warning(f"Failed while iterating test split for {lang}: {e}")

    still_missing = sorted(i for i in expected if i not in audio_dict)
    if still_missing:
        logger.warning(
            f"{len(still_missing)} expected Test IDs have no audio after HF load. "
            f"First few: {still_missing[:10]}"
        )
    else:
        logger.info(f"Audio coverage complete: {len(audio_dict)} / {len(expected)} IDs")
    return audio_dict


def _run_parallel_inference(test_ids, audio_dict, target_languages, hf_token, prefer_gemma=False):
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        num_workers = max(1, num_gpus)  # 1 worker/GPU is more stable than 2
        logger.info(f"Detected {num_gpus} GPUs. Launching {num_workers} parallel workers...")
    else:
        num_workers = 1
        logger.info("No GPUs detected. Launching 1 CPU worker...")

    shards = [[] for _ in range(num_workers)]
    for idx, audio_id in enumerate(test_ids):
        shards[idx % num_workers].append(audio_id)

    sharded_audio_dicts = [{} for _ in range(num_workers)]
    for w_id in range(num_workers):
        for audio_id in shards[w_id]:
            if audio_id in audio_dict:
                sharded_audio_dicts[w_id][audio_id] = audio_dict[audio_id]

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    processes = []

    logger.info("Spawning parallel inference worker processes...")
    for w_id in range(num_workers):
        p = ctx.Process(
            target=worker_inference,
            args=(
                w_id,
                num_gpus,
                target_languages,
                shards[w_id],
                sharded_audio_dicts[w_id],
                return_dict,
                hf_token,
                prefer_gemma,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.warning(
                f"Worker process {p.pid} exited with code {p.exitcode}; "
                "its IDs may be blank and will be retried if possible."
            )

    worker_results = {w_id: dict(return_dict.get(w_id, {})) for w_id in range(num_workers)}
    return _merge_worker_predictions(worker_results, expected_ids=test_ids)


def _retry_blank_ids(blank_ids, audio_dict, target_languages, hf_token, prefer_gemma=False):
    """Single-process retry for blank predictions (replaces transcribe_missing.py)."""
    if not blank_ids:
        return {}
    logger.info(f"Retrying {len(blank_ids)} blank IDs in a single-process pass...")
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    shard_audio = {aid: audio_dict[aid] for aid in blank_ids if aid in audio_dict}
    missing_audio = [aid for aid in blank_ids if aid not in audio_dict]
    if missing_audio:
        logger.warning(f"Retry cannot recover {len(missing_audio)} IDs with no audio: {missing_audio[:10]}")

    p = ctx.Process(
        target=worker_inference,
        args=(0, torch.cuda.device_count(), target_languages, blank_ids, shard_audio, return_dict, hf_token, prefer_gemma),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        logger.warning(f"Retry worker exited with code {p.exitcode}")
    return dict(return_dict.get(0, {}))


def _ensure_best_models_from_checkpoints(target_languages):
    """
    If best_model/ is missing but Lightning .ckpt exists, export HF weights so
    generate_submission uses the fine-tune instead of base mms-1b-all.
    """
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    for lang in target_languages:
        out = Path(get_outputs_dir()) / f"{lang}_mms-300m_fold0"
        best = out / "best_model"
        if best.is_dir() and (best / "config.json").exists():
            continue
        ckpt_dir = out / "checkpoints"
        if not ckpt_dir.is_dir():
            logger.warning(f"No best_model and no checkpoints for {lang} under {out}")
            continue
        # Prefer lowest val_loss epoch ckpt, then last.ckpt, then newest
        scored = []
        for p in ckpt_dir.glob("*.ckpt"):
            name = p.name
            m = re.search(r"val_loss=([0-9.]+)", name)
            if m:
                try:
                    scored.append((0, float(m.group(1)), p))
                    continue
                except ValueError:
                    pass
            if name.startswith("last"):
                scored.append((1, 0.0, p))
            else:
                scored.append((2, -p.stat().st_mtime, p))
        if not scored:
            logger.warning(f"No .ckpt files for {lang}")
            continue
        scored.sort(key=lambda t: (t[0], t[1]))
        ckpt_path = scored[0][2]
        logger.info(f"Exporting {lang} best_model from {ckpt_path} …")
        try:
            # weights_only=False required for Lightning checkpoints on torch>=2.6
            try:
                blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(str(ckpt_path), map_location="cpu")
            state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
            # Strip Lightning prefix model.
            cleaned = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                cleaned[nk] = v
            # Need a base architecture; processor from facebook/mms-300m or fine-tune config if present
            base_id = "facebook/mms-300m"
            processor = Wav2Vec2Processor.from_pretrained(base_id, target_lang=lang)
            model = Wav2Vec2ForCTC.from_pretrained(
                base_id, target_lang=lang, ignore_mismatched_sizes=True
            )
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            logger.info(
                f"  load_state_dict {lang}: missing={len(missing)} unexpected={len(unexpected)}"
            )
            best.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best))
            processor.save_pretrained(str(best))
            try:
                processor.tokenizer.set_target_lang(lang)
            except Exception:
                pass
            logger.info(f"  Wrote {best}")
        except Exception as exc:
            logger.warning(f"Failed to export best_model for {lang} from ckpt: {exc}")


def main():
    logger.info("Initializing high-performance parallel inference pipeline...")
    configure_hf_cache()

    # 1. Resolve paths — SampleSubmission is the Zindi contract (preferred over Test.csv)
    test_csv_path = "Test.csv"
    if not os.path.exists(test_csv_path):
        found = glob.glob("**/Test.csv", recursive=True)
        if found:
            test_csv_path = found[0]

    logger.info(f"Loading expected IDs (SampleSubmission preferred) near: {test_csv_path}")
    test_ids = _read_expected_submission_ids(test_csv_path)
    logger.info(f"Expected submission size: {len(test_ids)} IDs")

    # 2. Define target languages
    target_languages = ["lin", "sna", "lug"]

    import sys
    hf_token = os.environ.get("HF_TOKEN")
    prefer_gemma = "--prefer-gemma" in sys.argv
    max_blank_frac = 0.02
    if "--hf_token" in sys.argv:
        idx = sys.argv.index("--hf_token")
        if idx + 1 < len(sys.argv):
            hf_token = sys.argv[idx + 1]
    if "--max-blank-frac" in sys.argv:
        idx = sys.argv.index("--max-blank-frac")
        if idx + 1 < len(sys.argv):
            max_blank_frac = float(sys.argv[idx + 1])

    # 2b. Export best_model from Lightning ckpt when missing (e.g. lin after partial Kaggle restore)
    if "--skip-export-best" not in sys.argv:
        _ensure_best_models_from_checkpoints(target_languages)

    # 3. Load audio once (shared by main pass + blank retry)
    audio_dict = _load_test_audio_dict(test_ids)
    logger.info(
        f"Audio cache ready: {len(audio_dict)} / {len(test_ids)} expected IDs "
        f"({len(audio_dict)/max(len(test_ids),1):.1%} coverage)"
    )
    if len(audio_dict) < len(test_ids):
        missing_audio = [i for i in test_ids if i not in audio_dict]
        logger.warning(
            f"{len(missing_audio)} IDs have no audio — they would become Zindi "
            f"'Missing entries' if Target were left empty. Sample: {missing_audio[:10]}"
        )

    # 4. Main parallel pass
    combined = _run_parallel_inference(
        test_ids, audio_dict, target_languages, hf_token, prefer_gemma=prefer_gemma
    )
    # Guarantee every expected key exists before retry/write
    for aid in test_ids:
        combined.setdefault(aid, "")

    # 5. Retry blanks (eliminates need for transcribe_missing.py)
    blank_ids = [aid for aid in test_ids if not str(combined.get(aid, "")).strip()]
    if blank_ids:
        retry_preds = _retry_blank_ids(
            blank_ids, audio_dict, target_languages, hf_token, prefer_gemma=prefer_gemma
        )
        for aid, text in retry_preds.items():
            if text and str(text).strip():
                combined[aid] = str(text)

    # 6. Validate and write (fails hard if blank rate is still too high)
    #    Empty Targets are replaced with a space so Zindi never sees NaN → "Missing entries".
    _write_validated_submission(
        test_ids,
        combined,
        output_path="submission.csv",
        max_blank_frac=max_blank_frac,
    )

    # 7. Final human-readable integrity report
    check = pd.read_csv("submission.csv", dtype=str)
    n = len(check)
    n_blank = int(check["Target"].fillna("").astype(str).str.strip().eq("").sum())
    logger.info(
        f"INTEGRITY: rows={n} expected={len(test_ids)} "
        f"id_set_match={set(check['ID'].astype(str)) == set(test_ids)} "
        f"whitespace_placeholders={n_blank}"
    )
    logger.info("Successfully generated submission.csv in the correct format!")


if __name__ == "__main__":
    main()
