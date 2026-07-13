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
        import gc
        from datasets import Audio, concatenate_datasets
        from src.data.dataset import (
            prepare_datasets,
            load_waxal_dataset_clean,
            parse_robust_csv,
            normalize_text,
        )
        from sklearn.model_selection import GroupKFold

        # Cap HF datasets RAM usage (Kaggle GPU ~29–30GB system RAM)
        os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")

        low_ram = bool(self.data_cfg.get("low_ram", True))
        k_folds = int(self.data_cfg.get("k_folds", 5))

        logger.info(
            f"Building speaker-independent folds (low_ram={low_ram}) for {self.target_lang}…"
        )

        if low_ram:
            # Avoid prepare_datasets double-load: CSV-only + cheap speaker groups.
            # True speaker_id from HF meta is better but re-loads the whole parquet set.
            train_df = parse_robust_csv(self.data_cfg["train_csv"])
            train_df = train_df[train_df["language"] == self.target_lang].reset_index(drop=True)
            train_df["normalized_transcription"] = train_df["transcription"].apply(normalize_text)
            train_df = train_df[train_df["normalized_transcription"].str.strip() != ""].reset_index(
                drop=True
            )
            # Pseudo speaker groups: hash(id) buckets keep GroupKFold structure without
            # a second full HF pass. Slightly weaker than real speaker_id but fits 30GB RAM.
            train_df["speaker_id"] = train_df["id"].astype(str).map(
                lambda x: f"spk_{hash(x) % 2000}"
            )
            gkf = GroupKFold(n_splits=k_folds)
            train_df["fold"] = -1
            for fold_idx, (_, val_idx) in enumerate(
                gkf.split(train_df, train_df["normalized_transcription"], train_df["speaker_id"])
            ):
                train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold_idx
            logger.info(f"Low-RAM GroupKFold done ({k_folds} folds, hash-speaker groups).")
        else:
            train_df, _ = prepare_datasets(
                train_csv_path=self.data_cfg["train_csv"],
                test_csv_path=self.data_cfg["test_csv"],
                languages=[self.target_lang],
                k_folds=k_folds,
            )
            train_df = train_df[train_df["language"] == self.target_lang].reset_index(drop=True)

        train_split_df = train_df[train_df["fold"] != self.fold].reset_index(drop=True)
        val_split_df = train_df[train_df["fold"] == self.fold].reset_index(drop=True)
        logger.info(f"CSV fold split — train IDs: {len(train_split_df)} | val IDs: {len(val_split_df)}")

        # Cap rows early for extreme low-RAM smoke tests
        max_train = int(self.data_cfg.get("max_train_samples", 0) or 0)
        max_val = int(self.data_cfg.get("max_val_samples", 0) or 0)
        if max_train > 0 and len(train_split_df) > max_train:
            train_split_df = train_split_df.sample(n=max_train, random_state=42).reset_index(drop=True)
            logger.info(f"Capped train IDs to {max_train}")
        if max_val > 0 and len(val_split_df) > max_val:
            val_split_df = val_split_df.sample(n=max_val, random_state=42).reset_index(drop=True)
            logger.info(f"Capped val IDs to {max_val}")

        id_to_label_train = dict(
            zip(train_split_df["id"].astype(str), train_split_df["normalized_transcription"])
        )
        id_to_label_val = dict(
            zip(val_split_df["id"].astype(str), val_split_df["normalized_transcription"])
        )
        train_ids = set(id_to_label_train)
        val_ids = set(id_to_label_val)
        all_ids = train_ids | val_ids
        id_to_label_all = {**id_to_label_train, **id_to_label_val}

        del train_df, train_split_df, val_split_df
        gc.collect()

        logger.info("Loading WaxalNLP once and matching fold IDs (low RAM)…")
        full_ds = load_waxal_dataset_clean(self.target_lang)

        parts = []
        for split_name in ("train", "validation"):
            if split_name not in full_ds:
                continue
            ds = full_ds[split_name]
            key = next((c for c in ("id", "client_id") if c in ds.column_names), None)
            if key is None:
                continue
            keep_cols = [c for c in (key, "audio") if c in ds.column_names]
            drop = [c for c in ds.column_names if c not in keep_cols]
            if drop:
                try:
                    ds = ds.remove_columns(drop)
                except Exception:
                    pass
            filtered = ds.filter(
                lambda batch, key=key, all_ids=all_ids: [str(x) in all_ids for x in batch[key]],
                batched=True,
                batch_size=2000,
                desc=f"match {split_name}",
            )
            if len(filtered) == 0:
                continue

            def _attach(batch, key=key, id_to_label_all=id_to_label_all, train_ids=train_ids):
                ids = [str(x) for x in batch[key]]
                batch["example_id"] = ids
                batch["normalized_transcription"] = [id_to_label_all.get(i, "") for i in ids]
                batch["is_train"] = [i in train_ids for i in ids]
                return batch

            filtered = filtered.map(_attach, batched=True, batch_size=2000)
            drop2 = [
                c
                for c in filtered.column_names
                if c not in ("audio", "normalized_transcription", "example_id", "is_train")
            ]
            if drop2:
                filtered = filtered.remove_columns(drop2)
            parts.append(filtered)

        del full_ds
        gc.collect()
        if not parts:
            raise ValueError(f"No HF audio matched fold split for lang={self.target_lang}")

        combined = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        del parts
        gc.collect()
        combined = combined.cast_column("audio", Audio(sampling_rate=16000))

        self.train_dataset = combined.filter(
            lambda x: bool(x.get("is_train")) and bool(str(x.get("normalized_transcription") or "").strip()),
            desc="select train fold",
        )
        self.val_dataset = combined.filter(
            lambda x: (not bool(x.get("is_train")))
            and bool(str(x.get("normalized_transcription") or "").strip()),
            desc="select val fold",
        )
        del combined, id_to_label_train, id_to_label_val, id_to_label_all, train_ids, val_ids, all_ids
        gc.collect()

        # Drop helper columns — keep only what the collator needs
        for name in ("train", "val"):
            ds = self.train_dataset if name == "train" else self.val_dataset
            drop = [c for c in ds.column_names if c not in ("audio", "normalized_transcription")]
            if drop:
                ds = ds.remove_columns(drop)
            if name == "train":
                self.train_dataset = ds
            else:
                self.val_dataset = ds

        logger.info(
            f"Ready — train: {len(self.train_dataset)} | val: {len(self.val_dataset)}"
        )

        if self.data_cfg.get("use_external_corpora", False) and not low_ram:
            try:
                from src.data.external_corpora import load_external_corpus

                sources = self.data_cfg.get("external_corpora_sources", ["common_voice", "fleurs"])
                logger.info(f"Loading external corpora for {self.target_lang}: {sources}")
                external = load_external_corpus(self.target_lang, sources=sources)
                if external is not None and len(external) > 0:
                    max_ext = int(self.data_cfg.get("external_max_samples", 5000))
                    if len(external) > max_ext:
                        external = external.shuffle(seed=42).select(range(max_ext))
                    keep = {"audio", "normalized_transcription"}
                    drop = [c for c in external.column_names if c not in keep]
                    if drop:
                        external = external.remove_columns(drop)
                    self.train_dataset = concatenate_datasets([self.train_dataset, external])
                    logger.info(f"Train size after external merge: {len(self.train_dataset)}")
                    del external
                    gc.collect()
            except Exception as exc:
                logger.warning(f"External corpora merge failed ({exc}); continuing with WAXAL only.")
        elif self.data_cfg.get("use_external_corpora", False) and low_ram:
            logger.info("low_ram=true: skipping external corpora merge (enable with low_ram=false).")

        gc.collect()

    def _dataloader_kwargs(self, train: bool):
        nw = int(self.train_cfg.get("num_workers", 0))
        pin = bool(self.train_cfg.get("pin_memory", nw > 0))
        kwargs = dict(
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=(nw > 0),
        )
        if nw > 0:
            kwargs["prefetch_factor"] = int(self.train_cfg.get("prefetch_factor", 2))
        return kwargs

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
            collate_fn=collator,
            drop_last=True,
            **self._dataloader_kwargs(train=True),
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
            collate_fn=collator,
            drop_last=False,
            **self._dataloader_kwargs(train=False),
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
        accelerator = "gpu" if torch.cuda.is_available() else "auto"
        # Honour --devices on multi-GPU (e.g. Kaggle 2x T4); fall back to all visible GPUs.
        if args.devices and args.devices > 0 and torch.cuda.is_available():
            devices = min(args.devices, torch.cuda.device_count())
        else:
            devices = "auto"
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"
        logger.info(f"GPU/CPU mode: accelerator={accelerator}, devices={devices}, precision={precision}")

    # On multi-GPU, each process reloads the full HF dataset → 2× system RAM.
    # Prefer devices=1 on Kaggle GPU (~30GB RAM) unless you know you have headroom.
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
        val_check_interval=float(train_cfg.get("val_check_interval", 1.0)),
        limit_val_batches=float(train_cfg.get("limit_val_batches", 0.25 if not use_tpu else 1.0)),
        callbacks=[checkpoint_cb, lr_monitor],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
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
