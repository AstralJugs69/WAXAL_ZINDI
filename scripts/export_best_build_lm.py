#!/usr/bin/env python3
"""
Export best (lowest val_loss) Lightning ckpt → best_model/ and build KenLM lm.bin.

Usage:
  python scripts/export_best_build_lm.py --langs lin,sna,lug --force
  python scripts/export_best_build_lm.py --langs sna --epochs-hint 5
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import torch


def log(msg: str):
    print(msg, flush=True)


def get_outputs_dir() -> Path:
    d = os.environ.get("WAXAL_OUTPUTS_DIR")
    if d:
        return Path(d).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "outputs"


def pick_best_ckpt(ckpt_dir: Path) -> Path | None:
    scored = []
    for p in ckpt_dir.glob("*.ckpt"):
        m = re.search(r"val_loss=([0-9.]+)", p.name)
        if m:
            try:
                scored.append((0, float(m.group(1)), p))
                continue
            except ValueError:
                pass
        ep = re.search(r"epoch[=_](\d+)", p.name, re.I)
        if ep:
            scored.append((1, -int(ep.group(1)), p))
        elif p.name.startswith("last"):
            scored.append((2, 0.0, p))
        else:
            scored.append((3, -p.stat().st_mtime, p))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def collect_transcripts(lang: str, train_csv: Path) -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.dataset import normalize_text

    texts: list[str] = []
    if not train_csv.exists():
        log(f"  Train.csv not found at {train_csv}")
        return texts
    try:
        import pandas as pd

        # Robust-ish read
        try:
            from src.data.dataset import parse_robust_csv

            df = parse_robust_csv(str(train_csv))
        except Exception:
            df = pd.read_csv(train_csv, on_bad_lines="skip", engine="python")
    except Exception as exc:
        log(f"  Failed reading Train.csv: {exc}")
        return texts

    id_col = "id" if "id" in df.columns else "ID" if "ID" in df.columns else df.columns[0]
    text_col = None
    for c in ("transcription", "Target", "text", "sentence", "normalized_transcription"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        return texts

    if "language" in df.columns:
        sub = df[df["language"].astype(str).str.lower() == lang]
    else:
        sub = df[df[id_col].astype(str).str.startswith(f"{lang}_")]

    for t in sub[text_col].astype(str).tolist():
        n = normalize_text(t)
        if n:
            texts.append(n)
    return texts


def export_lang(lang: str, outputs: Path, train_csv: Path, force: bool) -> bool:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    root = outputs / f"{lang}_mms-300m_fold0"
    ckpt_dir = root / "checkpoints"
    best_dir = root / "best_model"
    if not ckpt_dir.is_dir():
        log(f"[{lang}] no checkpoints under {ckpt_dir}")
        return False

    ckpt = pick_best_ckpt(ckpt_dir)
    if ckpt is None:
        log(f"[{lang}] no .ckpt files")
        return False
    log(f"[{lang}] best ckpt: {ckpt.name}")

    try:
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(str(ckpt), map_location="cpu")
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    cleaned = {}
    for k, v in state.items():
        nk = k[6:] if isinstance(k, str) and k.startswith("model.") else k
        cleaned[nk] = v

    # Same path as training: processor from mms-1b-all, backbone mms-300m, resized head
    processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all", target_lang=lang)
    try:
        processor.tokenizer.set_target_lang(lang)
    except Exception:
        pass
    vocab_size = len(processor.tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-300m",
        ignore_mismatched_sizes=True,
        vocab_size=vocab_size,
    )
    miss, unexp = model.load_state_dict(cleaned, strict=False)
    log(f"  load_state_dict missing={len(miss)} unexpected={len(unexp)}")
    if hasattr(model, "config"):
        model.config.apply_spec_augment = False

    if best_dir.exists() and force:
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    processor.save_pretrained(str(best_dir))
    log(f"  wrote weights → {best_dir}")

    # KenLM
    texts = collect_transcripts(lang, train_csv)
    log(f"  LM corpus lines: {len(texts)}")
    if texts:
        try:
            from src.decoding.kenlm_utils import build_language_model

            path = build_language_model(
                texts, str(best_dir), kenlm_dir="kenlm", order=5, force=force
            )
            log(f"  KenLM → {path}")
        except Exception as exc:
            log(f"  KenLM failed (beam-without-LM still OK): {exc}")
    else:
        log("  skip KenLM (no transcripts)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="lin,sna,lug")
    ap.add_argument("--force", action="store_true", help="Overwrite best_model + rebuild lm.bin")
    ap.add_argument("--train-csv", default="Train.csv")
    ap.add_argument(
        "--outputs",
        default=os.environ.get("WAXAL_OUTPUTS_DIR", ""),
    )
    args = ap.parse_args()
    outputs = Path(args.outputs).expanduser().resolve() if args.outputs else get_outputs_dir()
    train_csv = Path(args.train_csv)
    if not train_csv.is_absolute():
        # try repo root
        cand = Path(__file__).resolve().parents[1] / args.train_csv
        if cand.exists():
            train_csv = cand

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    log(f"outputs={outputs}")
    ok = 0
    for lang in langs:
        if export_lang(lang, outputs, train_csv, force=args.force):
            ok += 1
    log(f"DONE {ok}/{len(langs)} languages")
    if ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
