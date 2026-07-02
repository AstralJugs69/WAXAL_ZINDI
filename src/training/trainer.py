import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Prevent JAX from locking TPU device on import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Prevent XLA client memory pre-allocation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # Prevent VRAM fragmentation OOMs on 16GB GPUs (P100/T4)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Use Rust-based parallel downloader for HuggingFace datasets


# Remove Kaggle environment variables that interfere with PJRT single-host auto-detection
for env_var in ["TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"]:
    if env_var in os.environ:
        os.environ.pop(env_var)

import argparse
import yaml
import logging
import torch
import numpy as np
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

def run_training(args, config, is_tpu=False, index=0):
    model_id = config["model_id"]
    is_seq2seq = "whisper" in model_id.lower()

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
    
    # Load pre-filtered CSV splits if available (generated by rank 0 in main() to avoid CPU decoding overlap)
    train_path = f"outputs/temp_train_fold{args.fold}.csv"
    val_path = f"outputs/temp_val_fold{args.fold}.csv"
    
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
            # Vectorized filter — runs in Arrow/C++, much faster than Python enumerate loop
            split_ds_filtered = split_ds.filter(
                lambda batch: [ex_id in id_set for ex_id in (batch.get("id") or batch.get("client_id"))],
                batched=True,
                batch_size=1000,
                desc=f"Matching IDs in {split_name}"
            )
            if len(split_ds_filtered) > 0:
                selected_ds_list.append(split_ds_filtered)
                
        if not selected_ds_list:
            raise ValueError(f"No matching IDs found in HF dataset for the split.")
            
        concat_ds = concatenate_datasets(selected_ds_list)
        
        # Map labels by ID using a fast batched map
        def add_labels(batch):
            ids = batch.get("id") or batch.get("client_id")
            batch["normalized_transcription"] = [id_to_label.get(ex_id, "") for ex_id in ids]
            return batch
        
        concat_ds = concat_ds.map(add_labels, batched=True, batch_size=1000, desc="Attaching labels")
        return concat_ds

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
    if is_seq2seq:
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
                train_dataset = _ext_cat([train_dataset, external_ds])
                if is_main_process:
                    logger.info(f"Train dataset after external corpora merge: {len(train_dataset)} examples")
        except Exception as exc:
            logger.warning(f"External corpora loading failed ({exc}). Continuing with WAXAL data only.")
    
    # JIT warm-up dummy step for TPU to pre-populate compilation cache
    if is_tpu:
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
    augmentator = DynamicAugmentator()
    data_collator = ASRDataCollatorWithPadding(
        processor=processor,
        augmentator=augmentator,
        is_seq2seq=is_seq2seq,
        sampling_rate=16000,
        static_buckets=is_tpu
    )
    
    # 5. Training arguments
    train_args = config["training"]
    output_dir = f"outputs/{args.target_lang}_{model_id.split('/')[-1]}_fold{args.fold}"
    
    training_class = Seq2SeqTrainingArguments if is_seq2seq else TrainingArguments
    
    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": train_args["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_args.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": train_args["gradient_accumulation_steps"],
        "learning_rate": float(train_args["learning_rate"]),
        "warmup_steps": train_args["warmup_steps"],
        "num_train_epochs": train_args["num_train_epochs"],
        "fp16": train_args["fp16"] and not is_tpu and torch.cuda.is_available(),
        "bf16": is_tpu,  # Hardware-accelerated bfloat16 on TPU
        "gradient_checkpointing": train_args["gradient_checkpointing"] and not is_tpu,
        "eval_strategy": train_args["evaluation_strategy"],
        "eval_steps": train_args["eval_steps"],
        "save_steps": train_args["save_steps"],
        "logging_steps": train_args["logging_steps"],
        "save_total_limit": train_args["save_total_limit"],
        "load_best_model_at_end": train_args["load_best_model_at_end"],
        "metric_for_best_model": "final_score" if is_seq2seq else train_args["metric_for_best_model"],
        "greater_is_better": False,
        "weight_decay": train_args["weight_decay"],
        "group_by_length": train_args.get("group_by_length", False),  # Disabled to prevent LengthGroupedSampler error when dynamic padding is used
        # On GPU, use 2 workers to prefetch and decode audio in background while GPU trains.
        # On TPU, keep 0 to avoid Arrow file lock contention across processes.
        "dataloader_num_workers": train_args.get("dataloader_num_workers", 0) if is_tpu else min(train_args.get("dataloader_num_workers", 2), 2),
        "remove_unused_columns": False,
        "report_to": ["none"],
        "ddp_find_unused_parameters": True
    }
    
    # Automatically adjust gradient_accumulation_steps on GPU to maintain a constant effective batch size of 32
    if not is_tpu and torch.cuda.is_available():
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        per_device_batch = training_kwargs["per_device_train_batch_size"]
        target_effective_batch = 32
        accum_steps = max(1, target_effective_batch // (per_device_batch * world_size))
        training_kwargs["gradient_accumulation_steps"] = accum_steps
        if is_main_process:
            logger.info(
                f"Dynamic Hyperparameter Alignment: Active GPUs={world_size} | "
                f"Per-device Batch={per_device_batch} | "
                f"Gradient Accumulation Steps={accum_steps} | "
                f"Effective Batch Size={per_device_batch * world_size * accum_steps}"
            )
            
    if is_seq2seq:
        training_kwargs["predict_with_generate"] = True
        training_kwargs["generation_max_length"] = 225
        
    trainer_args = training_class(**training_kwargs)
    
    # 6. Initialize trainer
    from src.utils.observability import ObservabilityCallback
    obs_callback = ObservabilityCallback(output_dir=output_dir)
    
    trainer_class = Seq2SeqTrainer if is_seq2seq else Trainer
    
    trainer = trainer_class(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics_fn(processor, is_seq2seq),
        processing_class=processor.feature_extractor,  # Required for CTC padding
        callbacks=[obs_callback]
    )
    
    # 7. Start training
    if is_main_process:
        logger.info("Starting model training...")
    # Flush GPU memory before starting the training loop
    if not is_tpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainer.train()
    
    # Save the best model — rank 0 only to avoid concurrent file writes
    if is_tpu:
        import torch_xla.core.xla_model as xm
        # Wait for all 8 cores to complete training and catch up
        xm.rendezvous("tpu_save_model_barrier")
        if index == 0:
            logger.info(f"Saving best model to {output_dir}/best_model")
            processor.save_pretrained(f"{output_dir}/best_model")
            # Save the consolidated/compiled weights on Core 0
            xm.save(model.state_dict(), f"{output_dir}/best_model/pytorch_model.bin")
            model.config.save_pretrained(f"{output_dir}/best_model")
    else:
        if is_main_process:
            logger.info(f"Saving best model to {output_dir}/best_model")
            processor.save_pretrained(f"{output_dir}/best_model")
            model.save_pretrained(f"{output_dir}/best_model")

        # -------------------------------------------------------------------
        # Build KenLM language model binary from training transcripts.
        # This runs once on the master process after training and is a no-op
        # if lm.bin already exists (safe to restart).
        # -------------------------------------------------------------------
        try:
            from src.decoding.kenlm_utils import build_language_model
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
    os.makedirs("outputs", exist_ok=True)
    train_path = f"outputs/temp_train_fold{args.fold}.csv"
    val_path = f"outputs/temp_val_fold{args.fold}.csv"
    
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
                ex_id = ex.get("id") or ex.get("client_id")
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
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
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
        from src.models.mms_model import load_processor_for_mms
        load_processor_for_mms(model_id=config["model_id"], target_lang=args.target_lang)
        
        # Pre-download the model checkpoint to disk cache so workers don't fight for locks
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
        done_flag_path = f"outputs/temp_done_fold{args.fold}.txt"
        
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
