import os
tpu_active = os.environ.get("TPU_NAME") or os.environ.get("TPU_ACCELERATOR_TYPE") or os.path.exists("/usr/share/tpu-support")

if tpu_active:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
else:
    try:
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
    except Exception:
        pass
os.environ["JAX_PLATFORMS"] = "cpu"  # Prevent JAX from locking TPU device on import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Prevent XLA client memory pre-allocation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # Prevent VRAM fragmentation OOMs on 16GB GPUs (P100/T4)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"  # Use high performance transfer with Xet for HuggingFace datasets
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable fast Rust-based HF downloader backend

# Monkey-patch jax to avoid AttributeError: module 'jax' has no attribute 'named_scope' in torch_xla profiler
try:
    import jax
    if not hasattr(jax, "named_scope"):
        class DummyNamedScope:
            def __init__(self, name, *args, **kwargs):
                self.name = name
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        jax.named_scope = DummyNamedScope
except Exception:
    pass

# Configure HuggingFace cache directories dynamically to persist on the working disk drive
working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
hf_home = os.environ.get("HF_HOME", os.path.join(working_dir, "hf_home"))
os.environ["HF_HOME"] = hf_home
os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))


# Debug log TPU environment variables for diagnosis
tpu_vars = {k: v for k, v in os.environ.items() if "TPU" in k}
if tpu_vars:
    print(f"[TPU Env Debug] {tpu_vars}")

# Clean up environment variables to force PJRT single-host local device discovery on single TPU VM nodes (like Colab TPU v5e-1)
# and prevent metadata service query mismatch (expected 4 workers, got 1)
os.environ["TPU_SKIP_MDS_QUERY"] = "1"
if os.environ.get("TPU_ACCELERATOR_TYPE") == "v5e-1":
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"

for env_var in ["TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID", "TPU_WORKER_ADDRESSES", "TPU_WORKER_PORT"]:
    if env_var in os.environ:
        os.environ.pop(env_var)

import argparse
import yaml
import logging
import torch
import numpy as np
import sys
from pathlib import Path

# Dynamic Numba-NumPy 2.x compatibility resolver
# Eagerly imports librosa to trigger all lazy Numba ufunc decorations, patching missing NumPy 2.x attributes on-the-fly.
while True:
    try:
        import librosa.core.audio
        break
    except AttributeError as e:
        msg = str(e)
        if "module 'numpy' has no attribute" in msg:
            attr = msg.split("'")[-2]
            if attr == "row_stack":
                np.row_stack = np.vstack
            elif attr == "trapz":
                np.trapz = getattr(np, "trapezoid", lambda *args, **kwargs: None)
            elif attr == "in1d":
                try:
                    from numpy.lib.arraysetops import in1d
                    np.in1d = in1d
                except ImportError:
                    np.in1d = np.isin
            else:
                setattr(np, attr, lambda *args, **kwargs: None)
            print(f"[Numba-Compatibility] Dynamically patched missing numpy attribute: {attr}")
            # Clear partially imported numba/librosa modules from cache to force clean retry
            for k in list(sys.modules.keys()):
                if k.startswith("numba") or k.startswith("librosa"):
                    sys.modules.pop(k)
        else:
            raise e

# Monkey-patch numpy.dtypes.StringDType for compatibility with JAX on older numpy versions
try:
    import numpy.dtypes as np_dtypes
except ImportError:
    import sys
    import types
    np_dtypes = types.ModuleType("numpy.dtypes")
    sys.modules["numpy.dtypes"] = np_dtypes
    np.dtypes = np_dtypes

if not hasattr(np_dtypes, "StringDType"):
    class MockStringDType:
        def __init__(self, *args, **kwargs):
            pass
    np_dtypes.StringDType = MockStringDType
    np.dtypes.StringDType = MockStringDType



# Class-level monkey-patching for PyTorch Conv1d, Conv2d, and Linear layers.
# This forces the PyTorch/XLA compiler to compile explicit runtime cast operators directly before any
# convolution or linear operations, resolving XLA's auto-type-promotion compiler quirks on TPU/bf16.
import torch.nn as nn

original_conv1d_forward = nn.Conv1d.forward
def patched_conv1d_forward(self, input):
    if input.dtype != self.weight.dtype:
        input = input.to(dtype=self.weight.dtype)
    return original_conv1d_forward(self, input)
nn.Conv1d.forward = patched_conv1d_forward

original_conv2d_forward = nn.Conv2d.forward
def patched_conv2d_forward(self, input):
    if input.dtype != self.weight.dtype:
        input = input.to(dtype=self.weight.dtype)
    return original_conv2d_forward(self, input)
nn.Conv2d.forward = patched_conv2d_forward

original_linear_forward = nn.Linear.forward
def patched_linear_forward(self, input):
    if input.dtype != self.weight.dtype:
        input = input.to(dtype=self.weight.dtype)
    return original_linear_forward(self, input)
nn.Linear.forward = patched_linear_forward

import jiwer
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    Trainer, 
    TrainingArguments,
    Wav2Vec2Processor,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration
)
from src.data.dataset import prepare_datasets, normalize_text
from src.data.filter import filter_dataset
from src.data.augment import ASRDataCollatorWithPadding, DynamicAugmentator
from src.models.mms_model import get_mms_model_with_adapter, load_processor_for_mms
from src.models.whisper_model import get_whisper_lora_model

def get_outputs_dir():
    outputs_dir = os.environ.get("WAXAL_OUTPUTS_DIR")
    if outputs_dir:
        path = Path(outputs_dir).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parents[2] / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

# Try PyTorch/XLA imports conditionally for TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# Print CUDA Diagnostics on GPU to debug hardware compatibility
if torch.cuda.is_available():
    logger.info(f"=== GPU CUDA Diagnostics ===")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
    try:
        logger.info(f"PyTorch Compiled Architectures: {torch.cuda.get_arch_list()}")
    except Exception as e:
        logger.info(f"Could not retrieve architecture list: {e}")
    logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    logger.info(f"=============================")

def preprocess_logits_for_metrics(logits, labels):
    """
    Computes argmax on the GPU/TPU device during evaluation to prevent
    accumulating huge float32 logit tensors in system RAM.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

from transformers import TrainerCallback
class GarbageCollectionCallback(TrainerCallback):
    """
    Triggers Python garbage collection, PyArrow memory pool release,
    and PyTorch CUDA cache clearing at key steps to prevent RAM/VRAM accumulation.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        import gc
        import torch
        import pyarrow as pa
        gc.collect()
        pa.default_memory_pool().release_unused()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def on_evaluate(self, args, state, control, **kwargs):
        import gc
        import torch
        import pyarrow as pa
        gc.collect()
        pa.default_memory_pool().release_unused()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_compute_metrics_fn(processor, is_seq2seq):
    """
    Returns the metric computation function for evaluation.
    Handles CTC logits (argmax) vs Seq2Seq generated tokens.
    """
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # CTC logits needs argmax, Seq2Seq predictions are token IDs
        if not is_seq2seq:
            if isinstance(pred_ids, tuple):
                pred_ids = pred_ids[0]
            # Only apply argmax if it has not been done already by preprocess_logits_for_metrics
            if pred_ids.ndim == 3:
                pred_ids = np.argmax(pred_ids, axis=-1)
            
        # Replace -100 in labels
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Normalize transcripts to ensure we evaluate on cleaned texts
        pred_str = [normalize_text(p) for p in pred_str]
        label_str = [normalize_text(l) for l in label_str]
        
        # Filter out empty references to avoid division by zero
        valid_preds = []
        valid_labels = []
        for p, l in zip(pred_str, label_str):
            if l.strip():
                valid_preds.append(p)
                valid_labels.append(l)
                
        if not valid_labels:
            return {"wer": 1.0, "cer": 1.0, "final_score": 1.0}
            
        wer = jiwer.wer(reference=valid_labels, hypothesis=valid_preds)
        cer = jiwer.cer(reference=valid_labels, hypothesis=valid_preds)
        final_score = 0.5 * wer + 0.5 * cer
        
        return {"wer": wer, "cer": cer, "final_score": final_score}
        
    return compute_metrics


def run_gemma_evaluation(eval_dataset, model, processor, batch_size=4):
    """
    Decodes audio samples using Gemma 3n's native autoregressive generation
    and evaluates transcripts against references using WER and CER.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    references = []
    predictions = []
    
    for batch in eval_dataset.batch(batch_size=batch_size):
        transcriptions = batch.get("transcription") or batch.get("normalized_transcription") or []
        
        # Collate audio arrays
        audios = []
        for audio_info in batch["audio"]:
            if isinstance(audio_info, dict) and "array" in audio_info:
                arr = np.asarray(audio_info["array"]).flatten()
            else:
                arr = np.asarray(audio_info).flatten()
            audios.append(arr)
            
        # Apply chat templates dynamically
        texts = []
        for i, transcript in enumerate(transcriptions):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an assistant that transcribes speech accurately."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audios[i]},
                        {"type": "text", "text": "Please transcribe this audio."},
                    ],
                }
            ]
            chat_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(chat_prompt)
            
        inputs = processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            max_length=2048,
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            
        input_len = inputs.input_ids.shape[1]
        decoded = processor.tokenizer.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )
        
        references.extend(transcriptions)
        predictions.extend([t.strip() for t in decoded])
        
    refs_normalized = [normalize_text(r) for r in references]
    preds_normalized = [normalize_text(p) for p in predictions]
    
    valid_refs = []
    valid_preds = []
    for r, p in zip(refs_normalized, preds_normalized):
        if r.strip():
            valid_refs.append(r)
            valid_preds.append(p)
            
    if not valid_refs:
        return {"wer": 1.0, "cer": 1.0}
        
    wer = jiwer.wer(valid_refs, valid_preds)
    cer = jiwer.cer(valid_refs, valid_preds)
    return {"wer": wer, "cer": cer}


def run_training(args, config, is_tpu=False, index=0):
    model_id = config["model_id"]
    model_family = config.get("model_family", "mms")
    is_gemma = (model_family == "gemma")
    is_seq2seq = "whisper" in model_id.lower()
    output_dir = f"{get_outputs_dir()}/{args.target_lang}_{model_id.split('/')[-1]}_fold{args.fold}"

    # Determine whether this process is the master (rank-0) process.
    # In DDP mode (torchrun), LOCAL_RANK is set by the launcher.
    # In single-GPU or TPU mode, LOCAL_RANK is absent (defaults to 0).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = (local_rank == 0) and ((not is_tpu) or (index == 0))
    
    # Silence Hugging Face Datasets logging and progress bars on non-main ranks to avoid duplicate output
    import datasets
    if not is_main_process:
        datasets.utils.logging.set_verbosity_error()
        datasets.utils.logging.disable_progress_bar()
    else:
        # Keep master rank clean but visible
        datasets.utils.logging.set_verbosity_warning()
    
    # 1. Prepare datasets
    data_config = config["data"]
    
    if not is_tpu:
        # Prevent CUDA memory fragmentation by enabling expandable segments
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # 1. Set PyTorch thread limits to utilize all CPU threads on GPU
        try:
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            torch.set_num_threads(num_cores)
            os.environ["OMP_NUM_THREADS"] = str(num_cores)
            os.environ["MKL_NUM_THREADS"] = str(num_cores)
            os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
        except Exception:
            pass

        # 2. Limit PyTorch VRAM usage to 95% of available memory
        if torch.cuda.is_available():
            try:
                device_idx = torch.cuda.current_device()
                # Set memory fraction to 95% (0.95)
                torch.cuda.set_per_process_memory_fraction(0.95, device_idx)
                if is_main_process:
                    logger.info("Setting PyTorch CUDA memory fraction limit to 95% (0.95) of total VRAM.")
            except Exception as e:
                logger.warning(f"Could not set CUDA memory fraction: {e}")

        # 3. Utilize 80% of available CPU RAM for caching datasets in RAM
        try:
            import psutil
            cpu_mem = psutil.virtual_memory()
            available_ram_gb = cpu_mem.available / (1024**3)
            total_ram_gb = cpu_mem.total / (1024**3)
            
            # If available RAM is high (e.g. > 25GB free) or total RAM is >= 32GB,
            # we dynamically enable in-memory dataset caching to speed up training.
            if available_ram_gb > 25.0 or total_ram_gb >= 32.0:
                if not data_config.get("cache_in_memory", False):
                    data_config["cache_in_memory"] = True
                    if is_main_process:
                        logger.info(f"High CPU RAM detected (Total: {total_ram_gb:.1f} GB | Available: {available_ram_gb:.1f} GB). Dynamically enabling in-memory dataset caching to maximize training speed.")
        except Exception as e:
            logger.warning(f"Could not configure CPU RAM resource limits: {e}")
    
    # Load pre-filtered CSV splits if available (generated by rank 0 in main() to avoid CPU decoding overlap)
    outputs_dir = get_outputs_dir()
    train_path = f"{outputs_dir}/temp_train_{args.target_lang}_fold{args.fold}.csv"
    val_path = f"{outputs_dir}/temp_val_{args.target_lang}_fold{args.fold}.csv"
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        if is_main_process:
            logger.info("Loading pre-filtered train/val dataset splits from disk...")
        train_split_df = pd.read_csv(train_path)
        val_split_df = pd.read_csv(val_path)
        is_pre_filtered = True
    else:
        if is_main_process:
            logger.info("Pre-filtered splits not found. Fallback to parsing raw CSVs and on-the-fly filtering...")
        train_df, _ = prepare_datasets(
            train_csv_path=data_config["train_csv"],
            test_csv_path=data_config["test_csv"],
            languages=[args.target_lang],
            k_folds=data_config["k_folds"]
        )
        
        # Filter by language
        train_df = train_df[train_df["language"] == args.target_lang]
        
        # Filter using speech rate and duration heuristics
        train_df = filter_dataset(
            train_df,
            duration_min=data_config["duration_min"],
            duration_max=data_config["duration_max"],
            wps_min=data_config["wps_min"],
            wps_max=data_config["wps_max"]
        )
        
        # Get train and validation splits based on target fold
        train_split_df = train_df[train_df["fold"] != args.fold].reset_index(drop=True)
        val_split_df = train_df[train_df["fold"] == args.fold].reset_index(drop=True)
        is_pre_filtered = False

    if is_main_process:
        logger.info(f"Train split size: {len(train_split_df)} || Val split size: {len(val_split_df)}")
    
    # Lazily construct HF datasets using select to avoid copying raw audio bytes to RAM
    from datasets import concatenate_datasets
    from src.data.dataset import load_waxal_dataset_clean
    
    if is_tpu:
        import torch_xla.core.xla_model as xm
        # Force only the master process (Core 0) to load and cache the dataset first
        if xm.is_master_ordinal():
            logger.info(f"Master process (Core 0) caching HF dataset for language '{args.target_lang}'...")
            load_waxal_dataset_clean(args.target_lang)
        
        # Block other TPU cores until master is done caching
        xm.rendezvous("load_dataset_barrier")
        
        # Now all cores load the cached dataset (runs instantly, no duplicate download/disk write)
        full_ds = load_waxal_dataset_clean(args.target_lang)
    else:
        logger.info(f"Loading HF dataset for language '{args.target_lang}' cleanly...")
        full_ds = load_waxal_dataset_clean(args.target_lang)
    
    def build_lazy_dataset(split_df):
        id_to_label = dict(zip(split_df["id"], split_df["normalized_transcription"]))
        id_set = set(id_to_label.keys())
        selected_ds_list = []
        
        for split_name in ["train", "validation"]:
            if split_name not in full_ds:
                continue
            split_ds = full_ds[split_name]
            # Restrict format to ID columns to avoid decoding audio during filtering
            cols_to_format = [c for c in ["id", "client_id", "speaker_id"] if c in split_ds.column_names]
            split_ds_formatted = split_ds.with_format(columns=cols_to_format)
            
            # Vectorized filter — runs in Arrow/C++, much faster than Python enumerate loop
            split_ds_filtered = split_ds_formatted.filter(
                lambda batch: [ex_id in id_set for ex_id in (batch.get("id") or batch.get("client_id") or batch.get("speaker_id"))],
                batched=True,
                batch_size=1000,
                desc=f"Matching IDs in {split_name}"
            )
            if len(split_ds_filtered) > 0:
                selected_ds_list.append(split_ds_filtered)
                
        if not selected_ds_list:
            raise ValueError(f"No matching IDs found in HF dataset for the split.")
            
        concat_ds = concatenate_datasets(selected_ds_list)
        # Keep format restricted to ID columns during mapping to avoid audio decoding
        cols_to_format = [c for c in ["id", "client_id", "speaker_id"] if c in concat_ds.column_names]
        concat_ds_formatted = concat_ds.with_format(columns=cols_to_format)
        
        # Map labels by ID using a fast batched map
        def add_labels(batch):
            ids = batch.get("id") or batch.get("client_id") or batch.get("speaker_id")
            batch["normalized_transcription"] = [id_to_label.get(ex_id, "") for ex_id in ids]
            return batch
        
        concat_ds_mapped = concat_ds_formatted.map(add_labels, batched=True, batch_size=1000, desc="Attaching labels")
        # Restore full format to enable lazy audio decoding for training
        return concat_ds_mapped.with_format(None)

    train_dataset = build_lazy_dataset(train_split_df)
    val_dataset = build_lazy_dataset(val_split_df)
    
    # 2. Setup device configuration
    if is_tpu:
        device = torch_xla.device()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
        
    if is_main_process:
        logger.info(f"Target device resolved: {device}")
        if not is_tpu and torch.cuda.is_available():
            logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(local_rank)}")
            
    # 3. Load processor and model
    if is_gemma:
        import timm
        from src.models.gemma_model import load_processor_for_gemma, get_gemma_model
        
        # Warm start from previously saved best_model weights if available
        checkpoint_dir = f"{output_dir}/best_model"
        if os.path.exists(checkpoint_dir) and os.path.exists(f"{checkpoint_dir}/adapter_config.json"):
            if is_main_process:
                logger.info(f"Warm-starting Gemma adapter weights from existing checkpoint: {checkpoint_dir}")
            processor = load_processor_for_gemma(model_id=checkpoint_dir)
            base_model = get_gemma_model(
                model_id=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if not is_tpu else None
            )
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
        else:
            processor = load_processor_for_gemma(model_id=model_id)
            model = get_gemma_model(
                model_id=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if not is_tpu else None
            )
        if is_tpu:
            model = model.to(device)
    elif is_seq2seq:
        processor = WhisperProcessor.from_pretrained(model_id, language=args.target_lang, task="transcribe")
        model = get_whisper_lora_model(
            model_id=model_id,
            r=config["peft"]["r"],
            lora_alpha=config["peft"]["lora_alpha"],
            target_modules=config["peft"]["target_modules"],
            lora_dropout=config["peft"]["lora_dropout"],
            load_in_8bit=not is_tpu, # Disable 8-bit on TPU (bitsandbytes is CUDA only)
            device=device
        )
    else:
        processor = load_processor_for_mms(model_id=model_id, target_lang=args.target_lang)
        # GPU: always load in float32 — AMP (fp16=True) will cast ops to FP16 on the fly.
        # Weights must remain FP32 for the GradScaler to unscale correctly.
        # TPU: load in bfloat16 — TPU has no GradScaler and bf16 is the native dtype.
        model_dtype = torch.bfloat16 if is_tpu else torch.float32
        model = get_mms_model_with_adapter(
            model_id=model_id,
            target_lang=args.target_lang,
            freeze_feature_extractor=True,
            processor=processor,
            torch_dtype=model_dtype
        )
        model = model.to(device)
        
    # Ensure all targets are mapped and audio is decoded at 16kHz
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Filter dataset by duration and speaking rate (WPS) directly on the HF Dataset
    if not is_pre_filtered:
        if (not is_tpu) or (index == 0):
            logger.info(f"Applying duration [{data_config['duration_min']}s, {data_config['duration_max']}s] and WPS [{data_config['wps_min']}, {data_config['wps_max']}] filters to HF datasets...")
            
        def hf_filter_fn(example):
            from src.data.dataset import get_audio_data
            audio_info = example["audio"]
            array, sr = get_audio_data(audio_info)
            if array is None or sr is None:
                return False
            duration = len(array) / sr
            if duration < data_config["duration_min"] or duration > data_config["duration_max"]:
                return False
            transcript = example.get("normalized_transcription") or example.get("transcription") or ""
            word_count = len(transcript.split())
            if duration > 0:
                wps = word_count / duration
                if wps < data_config["wps_min"] or wps > data_config["wps_max"]:
                    return False
            return True

        train_dataset = train_dataset.filter(hf_filter_fn, desc="Filtering train dataset by duration/WPS")
        val_dataset = val_dataset.filter(hf_filter_fn, desc="Filtering val dataset by duration/WPS")
        
        if is_main_process:
            logger.info(f"After duration/WPS filter — Train dataset: {len(train_dataset)} || Val dataset: {len(val_dataset)}")
    else:
        if is_main_process:
            logger.info("Dataset is already pre-filtered. Skipping duration/WPS filter step.")

    # -----------------------------------------------------------------------
    # Optionally load external open-source corpora (Common Voice, FLEURS)
    # and concatenate with WAXAL training data to prevent acoustic overfitting.
    # All DDP ranks load independently (HF datasets cache handles concurrency).
    # -----------------------------------------------------------------------
    if data_config.get("use_external_corpora", False):
        try:
            from src.data.external_corpora import load_external_corpus
            from datasets import concatenate_datasets as _ext_cat
            ext_sources = data_config.get("external_corpora_sources", ["common_voice", "fleurs"])
            if is_main_process:
                logger.info(f"Loading external corpora for '{args.target_lang}': {ext_sources}")
            external_ds = load_external_corpus(args.target_lang, sources=ext_sources)
            if external_ds is not None and len(external_ds) > 0:
                # We skip duration/WPS filtering on clean external corpora (Common Voice/FLEURS)
                # to prevent pre-decoding thousands of audio files, which causes massive I/O lag.
                train_dataset = _ext_cat([train_dataset, external_ds])
                if is_main_process:
                    logger.info(f"Train dataset after external corpora merge: {len(train_dataset)} examples")
        except Exception as exc:
            logger.warning(f"External corpora loading failed ({exc}). Continuing with WAXAL data only.")
            
    # -----------------------------------------------------------------------
    # In-memory dataset caching configuration (Optional)
    # -----------------------------------------------------------------------
    audio_cache = {}
    if data_config.get("cache_in_memory", False):
        if is_main_process:
            logger.info("Pre-decoding and caching all audio files in RAM in parallel...")
            
        from src.data.dataset import get_audio_data
        from multiprocessing.pool import ThreadPool
        import multiprocessing
        
        # Gather all unique examples to cache
        examples_to_cache = []
        seen_paths = set()
        
        for dataset in [train_dataset, val_dataset]:
            for example in dataset:
                audio_info = example.get("audio")
                if not audio_info:
                    continue
                path = audio_info.get("path") if isinstance(audio_info, dict) else getattr(audio_info, "path", "")
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    examples_to_cache.append((path, audio_info))
                    
        def load_single_audio(item):
            path, audio_info = item
            try:
                y, sr = get_audio_data(audio_info)
                # Resample to 16kHz at caching time (in parallel) to prevent massive CPU overhead
                # during the training loop.
                if y is not None and sr is not None and sr != 16000:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000
                return path, y, sr
            except Exception:
                return path, None, None
                
        num_cores = multiprocessing.cpu_count()
        # Parallel load using ThreadPool to saturate CPU and speed up startup
        with ThreadPool(num_cores) as pool:
            results = pool.map(load_single_audio, examples_to_cache)
            
        for path, y, sr in results:
            if y is not None:
                audio_cache[path] = (y, sr)
                
        if is_main_process:
            logger.info(f"Successfully cached {len(audio_cache)} decoded audio arrays in RAM.")

    # -----------------------------------------------------------------------
    # Precompute length column for LengthGroupedSampler (speeds up training 10x-16x)
    # -----------------------------------------------------------------------
    group_by_length_enabled = config["training"].get("group_by_length", False) or not is_gemma
    
    if group_by_length_enabled:
        if is_main_process:
            logger.info("Computing audio lengths for length-grouped batching...")
            
        def add_length_fn(example):
            path = example["audio"].get("path") if isinstance(example["audio"], dict) else getattr(example["audio"], "path", "")
            if path in audio_cache:
                example["length"] = len(audio_cache[path][0])
            else:
                # Fallback if not cached
                from src.data.dataset import get_audio_data
                y, _ = get_audio_data(example["audio"])
                example["length"] = len(y) if y is not None else 0
            return example
            
        train_dataset = train_dataset.map(add_length_fn, keep_in_memory=True, desc="Adding length column to train dataset")
        val_dataset = val_dataset.map(add_length_fn, keep_in_memory=True, desc="Adding length column to val dataset")

    # Clean up intermediate variables and force garbage collection to free CPU RAM
    if is_main_process:
        logger.info("Cleaning up intermediate dataframes and variables to free CPU RAM...")
    
    # Delete variables
    if 'train_split_df' in locals():
        del train_split_df
    if 'val_split_df' in locals():
        del val_split_df
    if 'train_df' in locals():
        del train_df
    if 'full_ds' in locals():
        del full_ds
    if 'external_ds' in locals():
        del external_ds
        
    import gc
    import pyarrow as pa
    gc.collect()
    pa.default_memory_pool().release_unused()
    
    # JIT warm-up dummy step for TPU to pre-populate compilation cache
    if is_tpu and not is_gemma:
        if index == 0:
            logger.info("Executing JIT warm-up step to populate compilation cache...")
            try:
                model.to(device)
                with torch.no_grad():
                    dummy_input = torch.randn(1, 80, 3000).to(device) if is_seq2seq else torch.randn(1, 80000).to(device)
                    if is_seq2seq:
                        model.model.encoder(dummy_input)
                    else:
                        model(dummy_input)
                import torch_xla.core.xla_model as xm
                xm.mark_step()
                logger.info("JIT warm-up completed successfully.")
            except Exception as e:
                logger.warning(f"JIT warm-up skipped: {e}")
        # Synchronize all TPU cores so they wait for Core 0's graph compilation to complete
        import torch_xla.core.xla_model as xm
        xm.rendezvous("tpu_jit_warmup_barrier")
            
    # 4. Setup augmentator and data collator (apply static bucketing on TPU)
    if is_gemma:
        from src.data.gemma_augment import GemmaDataCollator
        data_collator = GemmaDataCollator(
            processor=processor,
            audio_cache=audio_cache,
            system_message=config["data"].get("system_message"),
            user_message=config["data"].get("user_message")
        )
    else:
        augmentator = DynamicAugmentator()
        data_collator = ASRDataCollatorWithPadding(
            processor=processor,
            augmentator=augmentator,
            is_seq2seq=is_seq2seq,
            sampling_rate=16000,
            static_buckets=is_tpu,
            audio_cache=audio_cache
        )
    
    # 5. Training arguments
    train_args = config["training"]
    output_dir = f"{get_outputs_dir()}/{args.target_lang}_{model_id.split('/')[-1]}_fold{args.fold}"
    
    if is_gemma:
        from trl import SFTConfig
        training_class = SFTConfig
    else:
        training_class = Seq2SeqTrainingArguments if is_seq2seq else TrainingArguments
    
    # Determine default dataloader workers (GPU/CPU mode only)
    if is_tpu:
        default_workers = 0
        prefetch_factor = None
    else:
        import psutil
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # When dataset is cached in RAM, we MUST set num_workers = 0.
        # This is because python's reference counting on the cached dict/objects triggers
        # Copy-on-Write (COW) page faults in child worker processes, causing massive CPU memory inflation
        # and blocking training speed down to 18s/iteration.
        if data_config.get("cache_in_memory", False):
            default_workers = 0
            prefetch_factor = None
        else:
            # Aggressive CPU utilization if high core count (>16) is detected
            # Cap at min(8, num_cores) to prevent PyTorch DataLoader over-subscription queue overhead
            if num_cores > 16:
                default_workers = min(8, num_cores)
                prefetch_factor = 4          # Aggressively prefetch 4 batches in advance
            else:
                default_workers = max(1, int(num_cores * 0.8))
                prefetch_factor = 2
            
    # Dynamic Mixed Precision and Activation Checkpointing Resolver
    bf16_active = is_tpu
    fp16_active = train_args["fp16"] and not is_tpu and torch.cuda.is_available()
    grad_checkpointing = train_args["gradient_checkpointing"]
    
    if not is_tpu and torch.cuda.is_available():
        # Enable hardware-native BF16 mixed-precision on Ampere+ GPUs (A100, H100, RTX 30/40) for massive speedups
        if torch.cuda.is_bf16_supported():
            bf16_active = True
            fp16_active = False
            if is_main_process:
                logger.info("BF16 compatibility detected. Enabling native BF16 training.")
                


    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": train_args["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_args.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": train_args["gradient_accumulation_steps"],
        "learning_rate": float(train_args["learning_rate"]),
        "warmup_steps": train_args["warmup_steps"],
        "num_train_epochs": train_args.get("num_train_epochs", 3.0),
        "max_steps": train_args.get("max_steps", -1),
        "fp16": fp16_active,
        "bf16": bf16_active,
        "gradient_checkpointing": grad_checkpointing,
        "eval_strategy": train_args["evaluation_strategy"],
        "eval_steps": train_args["eval_steps"],
        "save_steps": train_args["save_steps"],
        "logging_steps": train_args["logging_steps"],
        "save_total_limit": train_args["save_total_limit"],
        "load_best_model_at_end": train_args["load_best_model_at_end"],
        "metric_for_best_model": "final_score" if is_seq2seq else train_args["metric_for_best_model"],
        "greater_is_better": False,
        "weight_decay": train_args["weight_decay"],
        "group_by_length": True,  # Enabled to reduce padding overhead and speed up training by up to 16x
        "dataloader_num_workers": default_workers,
        "dataloader_prefetch_factor": prefetch_factor,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True if default_workers > 0 else False,
        "eval_accumulation_steps": 10,  # Periodically clear/accumulate evaluation predictions to CPU
        "remove_unused_columns": False,
        "report_to": ["none"],
        "ddp_find_unused_parameters": True,
        "disable_tqdm": True
    }
    
    # Automatically adjust gradient_accumulation_steps to maintain a constant target effective batch size
    if is_tpu:
        import torch_xla.core.xla_model as xm
        world_size = xm.xrt_world_size()
        
        orig_batch_size = training_kwargs["per_device_train_batch_size"]
        if is_gemma:
            target_batch_size = 2 # Safe batch size for Gemma 3n on 16GB TPU core
        else:
            target_batch_size = 8 # Safe batch size for MMS on 16GB TPU core
            
        training_kwargs["per_device_train_batch_size"] = target_batch_size
        training_kwargs["per_device_eval_batch_size"] = max(1, target_batch_size)
        
        if is_main_process and target_batch_size != orig_batch_size:
            logger.info(f"TPU detected (16GB HBM per core). Dynamically scaled batch size from {orig_batch_size} to {target_batch_size} for optimal resource utilization.")
            
        per_device_batch = training_kwargs["per_device_train_batch_size"]
        target_effective_batch = train_args.get("target_effective_batch", 16)
        accum_steps = max(1, target_effective_batch // (per_device_batch * world_size))
        training_kwargs["gradient_accumulation_steps"] = accum_steps
        if is_main_process:
            logger.info(
                f"Dynamic Hyperparameter Alignment: Active TPU Cores={world_size} | "
                f"Per-device Batch={per_device_batch} | "
                f"Gradient Accumulation Steps={accum_steps} | "
                f"Effective Batch Size={per_device_batch * world_size * accum_steps}"
            )
            
    elif torch.cuda.is_available():
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
            
        # Detect VRAM capacity and dynamically scale batch size to target ~95% VRAM utilization
        device = torch.cuda.current_device()
        total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        
        orig_batch_size = training_kwargs["per_device_train_batch_size"]
        if is_gemma:
            if total_memory_gb < 18.0:
                target_batch_size = 2
            elif total_memory_gb < 30.0:
                target_batch_size = 4
            else:
                # 48GB VRAM (L40S) -> 16 is optimal and safe for Gemma 3n
                target_batch_size = 16
        else:
            if total_memory_gb < 18.0:
                target_batch_size = 8
            elif total_memory_gb < 30.0:
                target_batch_size = 16
            else:
                target_batch_size = 64
            
        training_kwargs["per_device_train_batch_size"] = target_batch_size
        # Also scale eval batch size
        training_kwargs["per_device_eval_batch_size"] = max(1, target_batch_size)
        
        if is_main_process and target_batch_size != orig_batch_size:
            if target_batch_size < orig_batch_size:
                logger.info(f"Low VRAM detected ({total_memory_gb:.2f} GB). Dynamically scaled batch size down from {orig_batch_size} to {target_batch_size} to prevent CUDA OOM.")
            else:
                logger.info(f"High VRAM detected ({total_memory_gb:.2f} GB). Dynamically scaled batch size up from {orig_batch_size} to {target_batch_size} to maximize GPU utilization.")
            
        per_device_batch = training_kwargs["per_device_train_batch_size"]
        target_effective_batch = train_args.get("target_effective_batch", 64)
        accum_steps = max(1, target_effective_batch // (per_device_batch * world_size))
        training_kwargs["gradient_accumulation_steps"] = accum_steps
        if is_main_process:
            logger.info(
                f"Dynamic Hyperparameter Alignment: Active GPUs={world_size} | "
                f"Per-device Batch={per_device_batch} | "
                f"Gradient Accumulation Steps={accum_steps} | "
                f"Effective Batch Size={per_device_batch * world_size * accum_steps}"
            )
            
    if is_gemma:
        training_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
        training_kwargs["max_seq_length"] = train_args.get("max_seq_length", 64)
        training_kwargs["packing"] = False
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    elif is_seq2seq:
        training_kwargs["predict_with_generate"] = True
        training_kwargs["generation_max_length"] = 225
        
    # Filter training_kwargs against the signature of training_class to prevent version mismatch errors (e.g., deprecated group_by_length in transformers 5.x)
    import inspect
    sig = inspect.signature(training_class)
    valid_args = {p.name for p in sig.parameters.values() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    training_kwargs = {k: v for k, v in training_kwargs.items() if k in valid_args}

    trainer_args = training_class(**training_kwargs)
    
    # 6. Initialize trainer
    from src.utils.observability import ObservabilityCallback
    obs_callback = ObservabilityCallback(output_dir=output_dir)
    gc_callback = GarbageCollectionCallback()
    
    if is_gemma:
        import peft
        import trl
        
        class _SFTTrainer(trl.SFTTrainer):
            def create_model_card(self, model_name=None, dataset_name=None, tags=None):
                pass
                
        peft_config = peft.LoraConfig(
            task_type="CAUSAL_LM",
            r=config["peft"]["r"],
            lora_alpha=config["peft"]["lora_alpha"],
            lora_dropout=config["peft"]["lora_dropout"],
            target_modules=config["peft"]["target_modules"],
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        
        shuffled_train = train_dataset.shuffle(seed=train_args.get("seed", 42))
        num_val = min(len(val_dataset), train_args.get("num_validation_examples", 200))
        val_ds_fixed = val_dataset.select(range(num_val))
        
        trainer_kwargs = {
            "model": model,
            "args": trainer_args,
            "data_collator": data_collator,
            "train_dataset": shuffled_train,
            "eval_dataset": val_ds_fixed,
            "peft_config": peft_config,
            "callbacks": [obs_callback, gc_callback]
        }
        trainer = _SFTTrainer(**trainer_kwargs)
    else:
        trainer_class = Seq2SeqTrainer if is_seq2seq else Trainer
        
        trainer_kwargs = {
            "model": model,
            "args": trainer_args,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "data_collator": data_collator,
            "compute_metrics": get_compute_metrics_fn(processor, is_seq2seq),
            "processing_class": processor.feature_extractor,  # Required for CTC padding
            "callbacks": [obs_callback, gc_callback]
        }
        if not is_seq2seq:
            trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics
            
        trainer = trainer_class(**trainer_kwargs)
    
    # 7. Start training
    if is_main_process:
        logger.info("Starting model training...")
    # Flush GPU memory before starting the training loop
    if not is_tpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Check for existing training checkpoints to resume from
    resume_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [
            os.path.join(output_dir, d) 
            for d in os.listdir(output_dir) 
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            try:
                # Sort checkpoints by step number
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                resume_checkpoint = checkpoints[-1]
                if is_main_process:
                    logger.info(f"Existing checkpoint detected. Resuming training from: {resume_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not parse checkpoint paths: {e}")
                
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save the best model — rank 0 only to avoid concurrent file writes
    if is_tpu:
        import torch_xla.core.xla_model as xm
        # Wait for all 8 cores to complete training and catch up
        xm.rendezvous("tpu_save_model_barrier")
        if index == 0:
            logger.info(f"Saving best model to {output_dir}/best_model")
            processor.save_pretrained(f"{output_dir}/best_model")
            if is_gemma:
                trainer.model.save_pretrained(f"{output_dir}/best_model")
            else:
                # Save the consolidated/compiled weights on Core 0
                xm.save(model.state_dict(), f"{output_dir}/best_model/pytorch_model.bin")
                model.config.save_pretrained(f"{output_dir}/best_model")
            
            # Copy best model to Kaggle working output directory for download
            if get_outputs_dir() != "outputs" and os.path.exists("/kaggle/working"):
                dest_dir = f"/kaggle/working/outputs/{args.target_lang}_{model_id.split('/')[-1]}_fold{args.fold}/best_model"
                logger.info(f"Copying best model to Kaggle working output directory: {dest_dir}")
                import shutil
                if os.path.exists(dest_dir):
                    try:
                        shutil.rmtree(dest_dir)
                    except Exception:
                        pass
                try:
                    shutil.copytree(f"{output_dir}/best_model", dest_dir)
                except Exception as e:
                    logger.warning(f"Failed to copy best model to working directory: {e}")
    else:
        if is_main_process:
            logger.info(f"Saving best model to {output_dir}/best_model")
            processor.save_pretrained(f"{output_dir}/best_model")
            if is_gemma:
                trainer.model.save_pretrained(f"{output_dir}/best_model")
            else:
                model.save_pretrained(f"{output_dir}/best_model")
            
            # Copy best model to Kaggle working output directory for download
            if get_outputs_dir() != "outputs" and os.path.exists("/kaggle/working"):
                dest_dir = f"/kaggle/working/outputs/{args.target_lang}_{model_id.split('/')[-1]}_fold{args.fold}/best_model"
                logger.info(f"Copying best model to Kaggle working output directory: {dest_dir}")
                import shutil
                if os.path.exists(dest_dir):
                    try:
                        shutil.rmtree(dest_dir)
                    except Exception:
                        pass
                try:
                    shutil.copytree(f"{output_dir}/best_model", dest_dir)
                except Exception as e:
                    logger.warning(f"Failed to copy best model to working directory: {e}")

        # Run post-training evaluation on validation set for Gemma
        if is_gemma and is_main_process:
            logger.info("Running post-training evaluation on validation set...")
            try:
                val_metrics = run_gemma_evaluation(val_ds_fixed, model, processor)
                logger.info(f"📊 Gemma ASR Validation Metrics:")
                logger.info(f"   WER : {val_metrics['wer']:.2%}")
                logger.info(f"   CER : {val_metrics['cer']:.2%}")
            except Exception as e:
                logger.warning(f"Post-training evaluation failed: {e}")

        # -------------------------------------------------------------------
        # Build KenLM language model binary from training transcripts.
        # This runs once on the master process after training and is a no-op
        # if lm.bin already exists (safe to restart).
        # -------------------------------------------------------------------
        if not is_gemma:
            try:
                from src.decoding.kenlm_utils import build_language_model
                train_path = f"{get_outputs_dir()}/temp_train_{args.target_lang}_fold{args.fold}.csv"
                if os.path.exists(train_path):
                    logger.info(f"Loading transcripts from temporary CSV for KenLM: {train_path}")
                    train_split_df_temp = pd.read_csv(train_path)
                    all_transcripts = list(train_split_df_temp["normalized_transcription"].dropna())
                else:
                    all_transcripts = list(train_split_df["normalized_transcription"].dropna())
                    
                lm_output_dir = f"{output_dir}/best_model"
                logger.info(f"Building KenLM language model from {len(all_transcripts)} training transcripts...")
                lm_bin_path = build_language_model(
                    transcripts=all_transcripts,
                    output_dir=lm_output_dir,
                    kenlm_dir="kenlm",
                    order=5,
                )
                if lm_bin_path:
                    logger.info(f"KenLM binary saved at: {lm_bin_path}")
                    # Write path to a sidecar file so inference pipeline can discover it
                    lm_ref_path = f"{output_dir}/best_model/lm_bin_path.txt"
                    with open(lm_ref_path, "w") as _f:
                        _f.write(lm_bin_path)
                    logger.info(f"LM path reference written to {lm_ref_path}")
            except Exception as exc:
                logger.warning(f"KenLM LM build failed ({exc}). Inference will fall back to greedy decoding.")
            
    # Ensure all spawned TPU processes synchronize at the exit to prevent termination race conditions
    if is_tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("tpu_exit_barrier")

def pre_filter_and_save_splits(args, config):
    """
    Decodes audio files and pre-filters datasets by duration/WPS in a single-process 
    pre-warmup step, saving the resulting splits to temp CSV files. 
    Spawned TPU cores / DDP child ranks can then load these CSVs instantly, 
    preventing concurrent multi-process CPU audio decoding contention.
    """
    import pandas as pd
    from src.data.dataset import prepare_datasets
    
    data_config = config["data"]
    
    # Check if pre-filtered splits already exist to skip this slow operation if possible
    outputs_dir = get_outputs_dir()
    os.makedirs(outputs_dir, exist_ok=True)
    train_path = f"{outputs_dir}/temp_train_{args.target_lang}_fold{args.fold}.csv"
    val_path = f"{outputs_dir}/temp_val_{args.target_lang}_fold{args.fold}.csv"
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        logger.info("Pre-filtered splits already exist on disk. Skipping pre-filtering.")
        return train_path, val_path

    logger.info("Running robust dataset loading...")
    train_df, _ = prepare_datasets(
        train_csv_path=data_config["train_csv"],
        test_csv_path=data_config["test_csv"],
        languages=[args.target_lang],
        k_folds=data_config["k_folds"]
    )
    
    # Filter by target language
    train_df = train_df[train_df["language"] == args.target_lang]
    
    # Get train and validation splits based on target fold
    train_split_df = train_df[train_df["fold"] != args.fold].reset_index(drop=True)
    val_split_df = train_df[train_df["fold"] == args.fold].reset_index(drop=True)
    
    from src.data.dataset import load_waxal_dataset_clean
    full_ds = load_waxal_dataset_clean(args.target_lang)
    
    valid_ids = set()
    
    def hf_filter_fn(example):
        from src.data.dataset import get_audio_data
        audio_info = example["audio"]
        array, sr = get_audio_data(audio_info)
        if array is None or sr is None:
            return False
        duration = len(array) / sr
        if duration < data_config["duration_min"] or duration > data_config["duration_max"]:
            return False
        transcript = example.get("normalized_transcription") or example.get("transcription") or ""
        word_count = len(transcript.split())
        if duration > 0:
            wps = word_count / duration
            if wps < data_config["wps_min"] or wps > data_config["wps_max"]:
                return False
        return True

    # Filter train & validation splits using HF dataset filtering
    for split_name in ["train", "validation"]:
        if split_name in full_ds:
            filtered_ds = full_ds[split_name].filter(hf_filter_fn, desc=f"Pre-filtering {split_name} split")
            for ex in filtered_ds:
                ex_id = ex.get("id") or ex.get("client_id") or ex.get("speaker_id")
                if ex_id:
                    valid_ids.add(ex_id)
                    
    # Filter DataFrames to keep only valid IDs
    train_split_filtered = train_split_df[train_split_df["id"].isin(valid_ids)].reset_index(drop=True)
    val_split_filtered = val_split_df[val_split_df["id"].isin(valid_ids)].reset_index(drop=True)
    
    # Save to outputs dir
    train_split_filtered.to_csv(train_path, index=False)
    val_split_filtered.to_csv(val_path, index=False)
    logger.info(f"Pre-filtered splits saved successfully. Train: {len(train_split_filtered)} | Val: {len(val_split_filtered)}")
    return train_path, val_path

def tpu_worker(index, args, config):
    """
    Worker function spawned on each TPU core.
    """
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    
    # Silence output logs for non-master cores to keep screen clean
    if index != 0:
        logging.getLogger().setLevel(logging.WARNING)
        
    logger.info(f"XLA Process index {index} initialized.")
    run_training(args, config, is_tpu=True, index=index)

def main():
    parser = argparse.ArgumentParser(description="ASR Model Fine-tuning Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to train (0 to k_folds-1)")
    parser.add_argument("--target_lang", type=str, default="lin", help="Target language (lin, sna, lug)")
    parser.add_argument("--tpu", action="store_true", help="Launch training on Google TPU v3-8 VM cores")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override training max steps")
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if args.max_steps and args.max_steps > 0:
        config["training"]["max_steps"] = args.max_steps
        
    if args.tpu:
        if not XLA_AVAILABLE:
            raise ImportError(
                "torch_xla is not installed. To run on TPU, please install PyTorch/XLA: "
                "pip install torch_xla"
            )
        # Pre-warm the HuggingFace dataset and model caches in the main process.
        # With start_method="spawn", each of the 8 child processes starts fresh.
        # Pre-warming ensures they load from disk cache (fast) instead of re-downloading.
        logger.info("Pre-warming dataset and model caches before TPU spawn...")
        data_config = config["data"]
        from src.data.dataset import prepare_datasets
        prepare_datasets(
            train_csv_path=data_config["train_csv"],
            test_csv_path=data_config["test_csv"],
            languages=[args.target_lang],
            k_folds=data_config["k_folds"]
        )
        model_family = config.get("model_family", "mms")
        if model_family == "gemma":
            import timm
            from src.models.gemma_model import load_processor_for_gemma, get_gemma_model
            load_processor_for_gemma(model_id=config["model_id"])
            logger.info(f"Pre-downloading model weights for {config['model_id']} to disk cache...")
            get_gemma_model(model_id=config["model_id"])
        else:
            from src.models.mms_model import load_processor_for_mms
            load_processor_for_mms(model_id=config["model_id"], target_lang=args.target_lang)
            from transformers import Wav2Vec2ForCTC
            logger.info(f"Pre-downloading model weights for {config['model_id']} to disk cache...")
            Wav2Vec2ForCTC.from_pretrained(config["model_id"])
        
        logger.info("Executing dataset pre-filtering before spawning TPU workers...")
        pre_filter_and_save_splits(args, config)
        
        logger.info("Cache pre-warming complete. Spawning 8 TPU worker processes...")

        # start_method="spawn" is required for PJRT — "fork" causes SIGTERM crashes
        # because forked processes inherit the parent's TPU device file descriptors.
        xmp.spawn(tpu_worker, args=(args, config), nprocs=None, start_method="spawn")
    else:
        # GPU/CPU path — verify local rank for DDP
        import time
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        done_flag_path = f"{get_outputs_dir()}/temp_done_{args.target_lang}_fold{args.fold}.txt"
        
        if local_rank == 0:
            if os.path.exists(done_flag_path):
                try:
                    os.remove(done_flag_path)
                except Exception:
                    pass
            logger.info("Executing master pre-filtering of dataset splits...")
            pre_filter_and_save_splits(args, config)
            with open(done_flag_path, "w") as f:
                f.write("done")
        else:
            logger.info(f"Child rank {local_rank} waiting for master process to complete dataset pre-filtering...")
            while not os.path.exists(done_flag_path):
                time.sleep(1)
                
        run_training(args, config, is_tpu=False, index=0)

if __name__ == "__main__":
    main()
