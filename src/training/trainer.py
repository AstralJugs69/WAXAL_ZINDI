import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Prevent JAX from locking TPU device on import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Prevent XLA client memory pre-allocation


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
    
    # 1. Prepare datasets
    data_config = config["data"]
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
    
    if (not is_tpu) or (index == 0):
        logger.info(f"Train split size: {len(train_split_df)} || Val split size: {len(val_split_df)}")
    
    # Lazily construct HF datasets using select to avoid copying raw audio bytes to RAM
    from datasets import concatenate_datasets
    from src.data.dataset import load_waxal_dataset_clean
    if (not is_tpu) or (index == 0):
        logger.info(f"Loading HF dataset for language '{args.target_lang}' cleanly...")
    full_ds = load_waxal_dataset_clean(args.target_lang)
    
    def build_lazy_dataset(split_df):
        id_to_label = dict(zip(split_df["id"], split_df["normalized_transcription"]))
        selected_ds_list = []
        ordered_labels = []
        
        for split_name in ["train", "validation"]:
            if split_name not in full_ds:
                continue
            split_ds = full_ds[split_name]
            # remove_columns prevents Arrow from processing the audio column during metadata scans
            split_ds_meta = split_ds.remove_columns(["audio"])
            
            split_indices = []
            for idx, example in enumerate(split_ds_meta):
                ex_id = example.get("id") or example.get("client_id")
                if ex_id in id_to_label:
                    split_indices.append(idx)
                    ordered_labels.append(id_to_label[ex_id])
                    
            if split_indices:
                selected_ds = split_ds.select(split_indices)
                selected_ds_list.append(selected_ds)
                
        if not selected_ds_list:
            raise ValueError(f"No matching IDs found in HF dataset for the split.")
            
        concat_ds = concatenate_datasets(selected_ds_list)
        concat_ds = concat_ds.add_column("normalized_transcription", ordered_labels)
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
        
    if (not is_tpu) or (index == 0):
        logger.info(f"Target device resolved: {device}")
        if not is_tpu and torch.cuda.is_available():
            logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            
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
        model_dtype = torch.bfloat16 if is_tpu else (torch.float16 if torch.cuda.is_available() else torch.float32)
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
    
    if (not is_tpu) or (index == 0):
        logger.info(f"After duration/WPS filter — Train dataset: {len(train_dataset)} || Val dataset: {len(val_dataset)}")
    
    # JIT warm-up dummy step for TPU to pre-populate compilation cache
    if is_tpu and index == 0:
        logger.info("Executing JIT warm-up step to populate compilation cache...")
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 80, 3000).to(device) if is_seq2seq else torch.randn(1, 80000).to(device)
                if is_seq2seq:
                    model.model.encoder(dummy_input)
                else:
                    model(dummy_input)
            torch_xla.sync()
            logger.info("JIT warm-up completed successfully.")
        except Exception as e:
            logger.warning(f"JIT warm-up skipped: {e}")
            
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
        "dataloader_num_workers": train_args.get("dataloader_num_workers", 0),
        "remove_unused_columns": False,
        "report_to": ["none"]
    }
    
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
    if (not is_tpu) or (index == 0):
        logger.info("Starting model training...")
    # Flush GPU memory before starting the training loop
    if not is_tpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainer.train()
    
    # Save the best model
    if (not is_tpu) or (index == 0):
        logger.info(f"Saving best model to {output_dir}/best_model")
        processor.save_pretrained(f"{output_dir}/best_model")
        model.save_pretrained(f"{output_dir}/best_model")

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
        
        logger.info("Cache pre-warming complete. Spawning 8 TPU worker processes...")

        # start_method="spawn" is required for PJRT — "fork" causes SIGTERM crashes
        # because forked processes inherit the parent's TPU device file descriptors.
        xmp.spawn(tpu_worker, args=(args, config), nprocs=None, start_method="spawn")
    else:
        run_training(args, config, is_tpu=False, index=0)

if __name__ == "__main__":
    main()
