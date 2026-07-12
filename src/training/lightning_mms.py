"""
Minimal PyTorch Lightning MMS-300M CTC trainer for Kaggle TPU.

Design goals (stability over cleverness):
  - Let Lightning own TPU process management (NO manual xmp.spawn)
  - Static audio shapes (pad/truncate to fixed max_samples)
  - num_workers=0 on TPU
  - No monkey-patches, no JIT warm-up hacks, no git hot-reload mid-train
  - Speaker-independent GroupKFold validation
  - Optional external corpora (Common Voice / FLEURS)
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger("lightning_mms")


def _configure_env_for_tpu():
    """Light env hygiene only — avoid aggressive TPU env scrubbing from the old path."""
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Prefer HF caches under Kaggle temp when present
    if not os.environ.get("HF_HOME"):
        if os.path.exists("/kaggle/temp"):
            hf_home = "/kaggle/temp/hf_home"
        elif os.path.exists("/kaggle/working"):
            hf_home = "/tmp/hf_home"
        else:
            hf_home = str(Path.cwd() / "hf_home")
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))


def get_outputs_dir() -> str:
    outputs_dir = os.environ.get("WAXAL_OUTPUTS_DIR")
    if outputs_dir:
        path = Path(outputs_dir).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parents[2] / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class FixedLengthCTCCollator:
    """
    Pads / truncates every waveform to a fixed number of samples so XLA sees
    a stable graph. Variable-length batching is a primary TPU failure mode.
    """

    def __init__(self, processor, max_samples: int, sampling_rate: int = 16000, augment: bool = False):
        self.processor = processor
        self.max_samples = int(max_samples)
        self.sampling_rate = sampling_rate
        self.augment = augment

    def _maybe_augment(self, y: np.ndarray) -> np.ndarray:
        if not self.augment or random.random() > 0.5:
            return y
        # Cheap speed perturbation only (pitch is too slow for TPU host CPU)
        factor = random.uniform(0.9, 1.1)
        if abs(factor - 1.0) < 1e-3 or len(y) == 0:
            return y
        indices = np.arange(0, len(y), factor)
        indices = indices[indices < len(y)].astype(np.float64)
        return np.interp(indices, np.arange(len(y)), y).astype(np.float32)

    def _fix_audio(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(y) > self.max_samples:
            y = y[: self.max_samples]
        elif len(y) < self.max_samples:
            y = np.pad(y, (0, self.max_samples - len(y)))
        return y

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        from src.data.dataset import get_audio_data, normalize_text

        waves = []
        texts = []
        for feat in features:
            audio_info = feat.get("audio")
            y, sr = get_audio_data(audio_info)
            if y is None:
                continue
            y = np.asarray(y, dtype=np.float32).reshape(-1)
            if sr is not None and sr != self.sampling_rate:
                import librosa

                y = librosa.resample(y, orig_sr=sr, target_sr=self.sampling_rate).astype(np.float32)
            y = self._maybe_augment(y)
            y = self._fix_audio(y)
            text = feat.get("normalized_transcription") or feat.get("transcription") or ""
            text = normalize_text(text)
            if not text:
                continue
            waves.append(y)
            texts.append(text)

        if not waves:
            # Degenerate batch — return a tiny dummy so training can skip cleanly
            dummy = np.zeros(self.max_samples, dtype=np.float32)
            waves = [dummy]
            texts = ["a"]

        inputs = self.processor(
            waves,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
        )
        # Prefer tokenizer path (as_target_processor is deprecated in recent transformers)
        label_ids = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).input_ids
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id or 0
        labels = label_ids.masked_fill(label_ids == pad_id, -100)
        batch = {
            "input_values": inputs.input_values,
            "labels": labels,
        }
        if "attention_mask" in inputs:
            batch["attention_mask"] = inputs.attention_mask
        return batch


class WaxalMMSDataModule:
    """
    Lightweight data module (not a full LightningDataModule subclass dependency)
    that builds torch DataLoaders with fixed-length collation.
    """

    def __init__(self, config: dict, target_lang: str, fold: int, processor):
        self.config = config
        self.target_lang = target_lang
        self.fold = fold
        self.processor = processor
        self.data_cfg = config["data"]
        self.train_cfg = config["training"]
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        from datasets import Audio, concatenate_datasets
        from src.data.dataset import prepare_datasets, load_waxal_dataset_clean, normalize_text

        logger.info("Building speaker-independent folds via GroupKFold...")
        train_df, _ = prepare_datasets(
            train_csv_path=self.data_cfg["train_csv"],
            test_csv_path=self.data_cfg["test_csv"],
            languages=[self.target_lang],
            k_folds=int(self.data_cfg.get("k_folds", 5)),
        )
        train_df = train_df[train_df["language"] == self.target_lang].reset_index(drop=True)
        train_split_df = train_df[train_df["fold"] != self.fold].reset_index(drop=True)
        val_split_df = train_df[train_df["fold"] == self.fold].reset_index(drop=True)
        logger.info(f"CSV fold split — train IDs: {len(train_split_df)} | val IDs: {len(val_split_df)}")

        full_ds = load_waxal_dataset_clean(self.target_lang)

        def _match_split(split_df):
            id_to_label = dict(zip(split_df["id"].astype(str), split_df["normalized_transcription"]))
            id_set = set(id_to_label.keys())
            parts = []
            for split_name in ("train", "validation"):
                if split_name not in full_ds:
                    continue
                ds = full_ds[split_name]
                id_cols = [c for c in ("id", "client_id") if c in ds.column_names]
                if not id_cols:
                    continue
                key = id_cols[0]
                filtered = ds.filter(
                    lambda batch, key=key, id_set=id_set: [str(x) in id_set for x in batch[key]],
                    batched=True,
                    batch_size=1000,
                    desc=f"Match {split_name}",
                )
                if len(filtered) == 0:
                    continue

                def _attach(batch, key=key, id_to_label=id_to_label):
                    batch["normalized_transcription"] = [
                        id_to_label.get(str(x), "") for x in batch[key]
                    ]
                    return batch

                filtered = filtered.map(_attach, batched=True, batch_size=1000)
                parts.append(filtered)
            if not parts:
                raise ValueError(f"No HF audio matched fold split for lang={self.target_lang}")
            out = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            return out.cast_column("audio", Audio(sampling_rate=16000))

        self.train_dataset = _match_split(train_split_df)
        self.val_dataset = _match_split(val_split_df)

        # Duration / WPS filter using decoded audio (once)
        dmin = float(self.data_cfg.get("duration_min", 1.5))
        dmax = float(self.data_cfg.get("duration_max", 16.0))
        wmin = float(self.data_cfg.get("wps_min", 1.0))
        wmax = float(self.data_cfg.get("wps_max", 8.0))

        def _ok(example):
            from src.data.dataset import get_audio_data

            y, sr = get_audio_data(example["audio"])
            if y is None or sr is None or sr <= 0:
                return False
            dur = len(y) / sr
            if dur < dmin or dur > dmax:
                return False
            text = example.get("normalized_transcription") or ""
            words = len(str(text).split())
            wps = words / dur if dur > 0 else 0
            return wmin <= wps <= wmax

        logger.info(f"Filtering train/val by duration [{dmin},{dmax}]s and WPS [{wmin},{wmax}]...")
        self.train_dataset = self.train_dataset.filter(_ok, desc="filter train")
        self.val_dataset = self.val_dataset.filter(_ok, desc="filter val")
        logger.info(f"After filter — train: {len(self.train_dataset)} | val: {len(self.val_dataset)}")

        if self.data_cfg.get("use_external_corpora", False):
            try:
                from src.data.external_corpora import load_external_corpus

                sources = self.data_cfg.get("external_corpora_sources", ["common_voice", "fleurs"])
                logger.info(f"Loading external corpora for {self.target_lang}: {sources}")
                external = load_external_corpus(self.target_lang, sources=sources)
                if external is not None and len(external) > 0:
                    max_ext = int(self.data_cfg.get("external_max_samples", 50000))
                    if len(external) > max_ext:
                        external = external.shuffle(seed=42).select(range(max_ext))
                    self.train_dataset = concatenate_datasets([self.train_dataset, external])
                    logger.info(f"Train size after external merge: {len(self.train_dataset)}")
            except Exception as exc:
                logger.warning(f"External corpora merge failed ({exc}); continuing with WAXAL only.")

        # Keep only columns needed for training
        keep = {"audio", "normalized_transcription"}
        for name, ds in (("train", self.train_dataset), ("val", self.val_dataset)):
            drop = [c for c in ds.column_names if c not in keep]
            if drop:
                if name == "train":
                    self.train_dataset = ds.remove_columns(drop)
                else:
                    self.val_dataset = ds.remove_columns(drop)

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        max_sec = float(self.train_cfg.get("max_audio_seconds", 16.0))
        collator = FixedLengthCTCCollator(
            processor=self.processor,
            max_samples=int(max_sec * 16000),
            augment=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.train_cfg.get("per_device_train_batch_size", 4)),
            shuffle=True,
            num_workers=int(self.train_cfg.get("num_workers", 0)),
            collate_fn=collator,
            drop_last=True,
            pin_memory=False,
            persistent_workers=False,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        max_sec = float(self.train_cfg.get("max_audio_seconds", 16.0))
        collator = FixedLengthCTCCollator(
            processor=self.processor,
            max_samples=int(max_sec * 16000),
            augment=False,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.train_cfg.get("per_device_eval_batch_size", 4)),
            shuffle=False,
            num_workers=int(self.train_cfg.get("num_workers", 0)),
            collate_fn=collator,
            drop_last=False,
            pin_memory=False,
            persistent_workers=False,
        )


def build_lightning_module(model, lr: float, weight_decay: float, warmup_steps: int, total_steps: int):
    import lightning.pytorch as pl

    class MMSCTCModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = model
            self.lr = lr
            self.weight_decay = weight_decay
            self.warmup_steps = warmup_steps
            self.total_steps = max(total_steps, 1)

        def forward(self, **batch):
            return self.model(**batch)

        def training_step(self, batch, batch_idx):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            return loss

        def configure_optimizers(self):
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            # Simple linear warmup + cosine decay without fancy schedulers that break on XLA
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step + 1) / float(max(1, self.warmup_steps))
                progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + np.cos(np.pi * progress))

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }

    return MMSCTCModule()


def save_best_model(model, processor, output_dir: str, transcripts: Optional[List[str]] = None, build_kenlm: bool = True):
    best_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    logger.info(f"Saving model + processor to {best_dir}")
    # Unwrap Lightning module if needed
    core = model.model if hasattr(model, "model") else model
    core.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)

    if build_kenlm and transcripts:
        try:
            from src.decoding.kenlm_utils import build_language_model

            lm_path = build_language_model(transcripts, best_dir, kenlm_dir="kenlm", order=5)
            if lm_path:
                logger.info(f"KenLM binary at {lm_path}")
        except Exception as exc:
            logger.warning(f"KenLM build skipped/failed: {exc}")


def train(args):
    _configure_env_for_tpu()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    data_cfg = config["data"]
    model_id = config["model_id"]
    target_lang = args.target_lang
    fold = args.fold

    output_dir = f"{get_outputs_dir()}/{target_lang}_mms-300m_fold{fold}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    from src.models.mms_model import get_mms_model_with_adapter, load_processor_for_mms

    logger.info(f"Loading processor/model for {target_lang}...")
    processor = load_processor_for_mms(model_id=model_id, target_lang=target_lang)
    model = get_mms_model_with_adapter(
        model_id=model_id,
        target_lang=target_lang,
        freeze_feature_extractor=True,
        processor=processor,
        torch_dtype=None,  # Lightning precision handles casting on TPU
    )
    # CTC stability knobs (prevent NaN loss on long/empty alignments)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.ctc_zero_infinity = True
    model.config.ctc_loss_reduction = "mean"

    dm = WaxalMMSDataModule(config, target_lang, fold, processor)
    dm.setup()

    # Estimate steps for scheduler
    steps_per_epoch = max(1, len(dm.train_dataset) // max(1, int(train_cfg.get("per_device_train_batch_size", 4)) * 8))
    max_epochs = int(args.epochs if args.epochs > 0 else train_cfg.get("num_train_epochs", 8))
    max_steps = int(args.max_steps if args.max_steps and args.max_steps > 0 else train_cfg.get("max_steps", -1))
    if max_steps and max_steps > 0:
        total_steps = max_steps
        max_epochs = 1000  # let max_steps dominate
    else:
        total_steps = steps_per_epoch * max_epochs

    lit_module = build_lightning_module(
        model=model,
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_steps=int(train_cfg.get("warmup_steps", 200)),
        total_steps=total_steps,
    )

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="mms-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=int(train_cfg.get("save_top_k", 2)),
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # accelerator:
    #   - "tpu" when --tpu or TPU hardware present
    #   - else GPU/CPU auto
    use_tpu = args.tpu or bool(
        os.environ.get("TPU_NAME")
        or os.environ.get("TPU_ACCELERATOR_TYPE")
        or os.path.exists("/usr/share/tpu-support")
        or os.path.exists("/dev/accel0")
    )
    if use_tpu:
        accelerator = "tpu"
        devices = args.devices if args.devices > 0 else 8
        precision = train_cfg.get("precision", "bf16-true")
        logger.info(f"TPU mode: devices={devices}, precision={precision}")
    else:
        accelerator = "auto"
        devices = "auto"
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"
        logger.info(f"Non-TPU mode: accelerator=auto, precision={precision}")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        default_root_dir=output_dir,
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 1.0)),
        accumulate_grad_batches=int(train_cfg.get("gradient_accumulation_steps", 1)),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 25)),
        val_check_interval=float(train_cfg.get("val_check_interval", 0.5)),
        callbacks=[checkpoint_cb, lr_monitor],
        enable_progress_bar=True,
        num_sanity_val_steps=0,  # skip sanity check recompilation cost on TPU
        deterministic=False,
    )

    logger.info("Starting Lightning training...")
    trainer.fit(lit_module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    # Save final weights from rank 0 only
    is_rank0 = True
    try:
        import torch_xla.core.xla_model as xm

        is_rank0 = xm.is_master_ordinal(local=False) if use_tpu else (trainer.global_rank == 0)
    except Exception:
        is_rank0 = getattr(trainer, "global_rank", 0) == 0

    if is_rank0:
        # Prefer best checkpoint weights when available
        best_ckpt = checkpoint_cb.best_model_path
        if best_ckpt and os.path.exists(best_ckpt):
            logger.info(f"Loading best checkpoint: {best_ckpt}")
            ckpt = torch.load(best_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # Strip Lightning prefix if present
            cleaned = {}
            for k, v in state.items():
                cleaned[k.replace("model.", "", 1) if k.startswith("model.") else k] = v
            try:
                lit_module.model.load_state_dict(cleaned, strict=False)
            except Exception as exc:
                logger.warning(f"Could not load best ckpt state into model ({exc}); saving current weights.")

        transcripts = []
        try:
            for ex in dm.train_dataset:
                t = ex.get("normalized_transcription") or ""
                if t:
                    transcripts.append(t)
        except Exception:
            pass

        build_kenlm = bool(config.get("decoding", {}).get("build_kenlm_after_train", True))
        save_best_model(lit_module, processor, output_dir, transcripts=transcripts, build_kenlm=build_kenlm)
        logger.info(f"Training complete for {target_lang}. Checkpoint: {output_dir}/best_model")

    return output_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Minimal Lightning MMS trainer (Kaggle TPU friendly)")
    p.add_argument("--config", type=str, default="config/base_mms_tpu.yaml")
    p.add_argument("--target_lang", type=str, default="lin", choices=["lin", "sna", "lug"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--tpu", action="store_true", help="Force TPU accelerator")
    p.add_argument("--devices", type=int, default=8, help="TPU cores (Kaggle v5e-8 => 8)")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=-1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
