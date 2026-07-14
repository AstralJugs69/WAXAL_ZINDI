#!/usr/bin/env python3
"""
WAXAL — Lightning AI studio bootstrap (CPU data prep OR GPU train helpers)

Paste/run this once per studio session instead of ad-hoc command dumps.

Usage
-----
CPU prep (no GPU credits):
  python lightning_studio_bootstrap.py prep

GPU train all langs (after prep):
  python lightning_studio_bootstrap.py train --langs lin,sna,lug

GPU train one lang:
  python lightning_studio_bootstrap.py train --lang lug --epochs 10 --batch-size 16 --resume

Upload checkpoints to Kaggle dataset (preferred):
  python lightning_studio_bootstrap.py upload

Env / files expected
--------------------
  kaggle.json  → /teamspace/studios/this_studio/kaggle.json
                 (cache download + checkpoint dataset upload)
  HF_TOKEN     → optional; pass --hf_token hf_... (never commit tokens)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

STUDIO = Path(os.environ.get("LIGHTNING_STUDIO", "/teamspace/studios/this_studio"))
REPO_URL = "https://github.com/AstralJugs69/WAXAL_ZINDI.git"
REPO = STUDIO / "WAXAL_ZINDI"
HF_HOME = Path(os.environ.get("HF_HOME", STUDIO / "hf_home"))
OUTPUTS = Path(os.environ.get("WAXAL_OUTPUTS_DIR", REPO / "outputs"))


def log(msg: str):
    print(msg, flush=True)


def apply_hf_token(token: str | None):
    if not token or not str(token).strip():
        return
    t = str(token).strip()
    os.environ["HF_TOKEN"] = t
    os.environ["HUGGING_FACE_HUB_TOKEN"] = t
    log("HF_TOKEN set from --hf_token (value not printed)")


def run(cmd, check=True, env=None):
    # Redact secrets next to --hf_token / --token in logs
    display = []
    skip = False
    for c in map(str, cmd):
        if skip:
            display.append("<redacted>")
            skip = False
            continue
        if c in ("--hf_token", "--token"):
            display.append(c)
            skip = True
            continue
        display.append(c)
    log("\n>>> " + " ".join(display))
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(list(map(str, cmd)), env=e)
    if check and r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode})")
    return r.returncode


def configure_env(hf_token: str | None = None):
    HF_HOME.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WAXAL_OUTPUTS_DIR"] = str(OUTPUTS)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    apply_hf_token(hf_token)
    log(f"HF_HOME={HF_HOME}")
    log(f"WAXAL_OUTPUTS_DIR={OUTPUTS}")
    log(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")


def ensure_repo():
    STUDIO.mkdir(parents=True, exist_ok=True)
    if REPO.exists() and (REPO / ".git").exists():
        run(["git", "-C", str(REPO), "pull", "origin", "main"], check=False)
    else:
        if REPO.exists():
            shutil.rmtree(REPO, ignore_errors=True)
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    log(f"REPO={REPO} HEAD={head}")
    return REPO


def setup_kaggle_json():
    dest = Path.home() / ".kaggle" / "kaggle.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    for src in (
        STUDIO / "kaggle.json",
        REPO / "kaggle.json",
        Path("kaggle.json"),
    ):
        if src.exists():
            shutil.copy2(src, dest)
            try:
                dest.chmod(0o600)
            except Exception:
                pass
            log(f"Installed kaggle.json from {src}")
            return True
    log("WARNING: kaggle.json not found (cache download may fail)")
    return False


def cmd_prep(args):
    """CPU-only: cache extract + external + fold CSVs (no training)."""
    configure_env(getattr(args, "hf_token", None))
    ensure_repo()
    setup_kaggle_json()
    os.chdir(REPO)
    prep_cmd = [sys.executable, "scripts/setup_data_cpu.py", "--repo", str(REPO)]
    if getattr(args, "hf_token", None):
        prep_cmd.extend(["--hf_token", args.hf_token])
    run(prep_cmd, check=False)
    log("\n=== PREP DONE (no training) ===")
    log("Next (GPU machine): python lightning_studio_bootstrap.py train --lang lug --epochs 10")


def cmd_train(args):
    """GPU train one or more languages with the proven recipe."""
    configure_env(getattr(args, "hf_token", None))
    ensure_repo()
    os.chdir(REPO)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "lightning",
            "transformers>=4.40.0",
            "datasets>=2.19.0,<4.0.0",
            "accelerate",
            "librosa",
            "soundfile",
            "jiwer",
            "pyyaml",
            "scikit-learn",
            "evaluate",
            "tqdm",
        ],
        check=False,
    )

    if args.lang:
        langs = [args.lang]
    else:
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    for lang in langs:
        epochs = args.epochs
        cmd = [
            sys.executable,
            "-m",
            "src.training.lightning_mms",
            "--config",
            args.config,
            "--target_lang",
            lang,
            "--fold",
            str(args.fold),
            "--devices",
            str(args.devices),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--unfreeze_feature_encoder",
        ]
        # Default: keep early stopping (config). Only disable if user asks.
        if getattr(args, "no_early_stopping", False):
            cmd.append("--no_early_stopping")
        if args.resume:
            cmd.append("--resume")
        if getattr(args, "ckpt_path", None):
            cmd.extend(["--ckpt_path", args.ckpt_path])
        log(f"\n===== TRAIN {lang} epochs={epochs} batch={args.batch_size} lr={args.lr} =====")
        run(cmd, check=False)

    log("\n=== TRAIN COMMANDS FINISHED ===")
    log(f"Checkpoints: {OUTPUTS}/{{lin,sna,lug}}_mms-300m_fold{args.fold}/")


def cmd_upload(args):
    """Upload checkpoints to Kaggle as a private dataset (uses kaggle.json)."""
    configure_env(getattr(args, "hf_token", None))
    ensure_repo()
    os.chdir(REPO)
    cmd = [
        sys.executable,
        "scripts/upload_checkpoints_kaggle.py",
        "--outputs",
        str(OUTPUTS),
        "--langs",
        args.langs,
        "--slug",
        args.slug,
        "--message",
        args.message,
    ]
    if args.skip_pack:
        cmd.append("--skip-pack")
    if args.public:
        cmd.append("--public")
    run(cmd, check=False)


def cmd_download(args):
    """Restore checkpoints from Kaggle (ignores broken size=0 list column)."""
    configure_env(getattr(args, "hf_token", None))
    ensure_repo()
    os.chdir(REPO)
    cmd = [
        sys.executable,
        "scripts/download_checkpoints_kaggle.py",
        "--dataset",
        args.dataset,
        "--outputs",
        str(OUTPUTS),
        "--download-dir",
        args.download_dir,
    ]
    if args.list_only:
        cmd.append("--list-only")
    if getattr(args, "force_per_file", False):
        cmd.append("--force-per-file")
    if getattr(args, "langs", None):
        cmd.extend(["--langs", args.langs])
    run(cmd, check=False)


def cmd_submit(args):
    configure_env(getattr(args, "hf_token", None))
    ensure_repo()
    os.chdir(REPO)
    sub_cmd = [
        sys.executable,
        "generate_submission.py",
        "--max-blank-frac",
        str(args.max_blank_frac),
    ]
    if getattr(args, "hf_token", None):
        sub_cmd.extend(["--hf_token", args.hf_token])
    run(sub_cmd, check=False)
    sub = REPO / "submission.csv"
    if sub.exists():
        dest = STUDIO / "submission.csv"
        shutil.copy2(sub, dest)
        log(f"submission.csv → {dest}")


def main():
    parser = argparse.ArgumentParser(description="WAXAL Lightning studio bootstrap")
    # Global token arg works before subcommand too if placed after subcommand name
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_hf_token(p):
        p.add_argument(
            "--hf_token",
            type=str,
            default=None,
            help="Hugging Face token (optional; for Common Voice / gated assets). Do not commit.",
        )

    p_prep = sub.add_parser("prep", help="CPU data prep (Kaggle cache + external + folds)")
    add_hf_token(p_prep)
    p_prep.set_defaults(func=cmd_prep)

    p_train = sub.add_parser("train", help="GPU train MMS (Lightning)")
    add_hf_token(p_train)
    p_train.add_argument("--lang", type=str, default=None, help="Single language lin|sna|lug")
    p_train.add_argument("--langs", type=str, default="lin,sna,lug")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--fold", type=int, default=0)
    p_train.add_argument("--devices", type=int, default=1)
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument(
        "--no-early-stopping",
        dest="no_early_stopping",
        action="store_true",
        help="Disable early stopping (not recommended for fresh trains)",
    )
    p_train.add_argument(
        "--ckpt-path",
        dest="ckpt_path",
        type=str,
        default=None,
        help="Explicit Lightning .ckpt (overrides auto-pick of furthest epoch)",
    )
    p_train.add_argument("--config", type=str, default="config/base_mms_lightning_96gb.yaml")
    p_train.set_defaults(func=cmd_train)

    p_up = sub.add_parser("upload", help="Upload checkpoints to Kaggle dataset")
    add_hf_token(p_up)
    p_up.add_argument("--langs", type=str, default="lin,sna,lug")
    p_up.add_argument(
        "--slug",
        type=str,
        default="waxal-mms-checkpoints",
        help="Kaggle dataset slug under your username",
    )
    p_up.add_argument("--message", type=str, default="Update WAXAL MMS checkpoints")
    p_up.add_argument("--skip-pack", action="store_true", help="Reuse existing .tar.gz packs")
    p_up.add_argument("--public", action="store_true")
    p_up.set_defaults(func=cmd_upload)

    p_dl = sub.add_parser("download", help="Download/restore checkpoints from Kaggle")
    add_hf_token(p_dl)
    p_dl.add_argument(
        "--dataset",
        default="cashgenenator/waxal-mms-checkpoints",
    )
    p_dl.add_argument("--download-dir", default="./ckpt_dl")
    p_dl.add_argument("--list-only", action="store_true")
    p_dl.add_argument(
        "--force-per-file",
        action="store_true",
        help="Skip bulk download; always use per-file mode (needed when bulk 404s)",
    )
    p_dl.add_argument(
        "--langs",
        default="lin,sna,lug",
        help="Comma-separated languages to restore (e.g. sna only after partial restore)",
    )
    p_dl.set_defaults(func=cmd_download)

    p_sub = sub.add_parser("submit", help="Generate submission.csv")
    add_hf_token(p_sub)
    p_sub.add_argument("--max-blank-frac", type=float, default=0.05)
    p_sub.set_defaults(func=cmd_submit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
