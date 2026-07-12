#!/usr/bin/env python3
"""
Kaggle TPU entrypoint for WAXAL MMS training (clean path).

Usage on Kaggle (TPU VM / TPU v5e-8 notebook):
    python run_tpu_train.py --lang all --tpu
    python run_tpu_train.py --lang lin --tpu --max_steps 4000
    python run_tpu_train.py --lang sna --tpu --epochs 10

This path deliberately avoids:
  - manual xmp.spawn
  - HF Trainer TPU hacks
  - mid-train git kill/restart
  - package reinstall of torch/torch_xla on Kaggle
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def detect_tpu() -> bool:
    return bool(
        os.environ.get("TPU_NAME")
        or os.environ.get("TPU_ACCELERATOR_TYPE")
        or os.path.exists("/usr/share/tpu-support")
        or os.path.exists("/dev/accel0")
        or os.path.exists("/dev/accel")
    )


def ensure_lightning():
    try:
        import lightning.pytorch  # noqa: F401
        return
    except ImportError:
        print("Installing lightning (does NOT reinstall torch/torch_xla)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightning>=2.2.0"])


def ensure_project_deps_light():
    """Install only missing ASR deps; never force-reinstall torch on TPU."""
    try:
        import transformers, datasets, jiwer, librosa, soundfile, yaml, sklearn  # noqa: F401
    except ImportError:
        print("Installing lightweight Python deps (constraints: do not upgrade torch)...")
        pkgs = [
            "transformers>=4.40.0",
            "datasets>=2.19.0,<4.0.0",
            "jiwer",
            "librosa",
            "soundfile",
            "pyyaml",
            "scikit-learn",
            "evaluate",
            "accelerate",
        ]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def configure_kaggle_hf_home() -> str:
    """
    Point HF caches at a large writable volume.

    Kaggle layout that matters for the ~54GB split-tar cache:
      /kaggle/input/<dataset>/hf_cache.tar.aa..am  — read-only mount (source chunks)
      /kaggle/temp/hf_home/                        — large writable extract target
      /kaggle/working/                             — only ~20GB; NEVER extract here

    The dataset was built offline via scripts/download_hf_assets.py, then packed as:
      tar of $HF_HOME  →  split into 5GB pieces (hf_cache.tar.aa, .ab, ...)
    Runtime reassembly is a streaming pipe (no intermediate 54GB concat file):
      cat hf_cache.tar.a* | tar -xf - -C $HF_HOME
    implemented in kaggle_bootstrapper.extract_cache_chunks().
    """
    if os.path.exists("/kaggle/temp"):
        hf_home = "/kaggle/temp/hf_home"
    elif os.path.exists("/kaggle/working"):
        # Fallback only — may OOM the 20GB working disk on full extract
        hf_home = "/tmp/hf_home"
        print("WARNING: /kaggle/temp missing; extracting to /tmp/hf_home (disk risk).")
    else:
        hf_home = str(Path.cwd() / "hf_home")

    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print(f"HF cache root: {hf_home}")
    print(f"  HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}")
    print(f"  HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")
    return hf_home


def extract_cache_if_present():
    """
    Locate cashgenenator/waxal-hf-cache-chunks under /kaggle/input and stream-extract
    into HF_HOME. Idempotent via extraction_completed.txt + hub/datasets non-empty checks.
    """
    from kaggle_bootstrapper import check_extraction_valid, extract_cache_chunks, find_cache_chunks_dir

    hf_home = configure_kaggle_hf_home()

    if check_extraction_valid(hf_home):
        print(f"HF cache already valid at {hf_home} (sentinel + hub/datasets present). Skipping extract.")
        return

    chunks = find_cache_chunks_dir()
    if not chunks:
        print(
            "No hf_cache.tar.aa under /kaggle/input (or . /content). "
            "Attach dataset: cashgenenator/waxal-hf-cache-chunks — "
            "otherwise training will download from Hugging Face Hub (slower)."
        )
        return

    print(f"Found split-tar cache chunks in: {chunks}")
    try:
        names = sorted(f for f in os.listdir(chunks) if f.startswith("hf_cache.tar.a"))
        print(f"  {len(names)} chunk files: {names[0]} … {names[-1]}")
    except Exception:
        pass

    try:
        # Streams aa|ab|… into `tar -xf - -C hf_home` then writes extraction_completed.txt
        extract_cache_chunks(chunks, hf_home)
        if check_extraction_valid(hf_home):
            print(f"Cache extraction OK → {hf_home}")
        else:
            print("WARNING: extract finished but hub/datasets validation failed.")
    except Exception as exc:
        print(f"Cache extraction failed ({exc}); continuing with Hub downloads.")


def parse_args():
    p = argparse.ArgumentParser(description="WAXAL MMS TPU trainer (Lightning, minimal)")
    p.add_argument("--config", default="config/base_mms_tpu.yaml")
    p.add_argument("--lang", default="all", help="lin|sna|lug|all")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--tpu", action="store_true")
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=-1, help="Per-language max steps (-1 = use epochs)")
    p.add_argument(
        "--epochs",
        type=int,
        default=-1,
        help="Per-language epochs (-1 = config default). Use high values to burn TPU quota.",
    )
    p.add_argument("--prefetch_external", action="store_true", default=True)
    p.add_argument("--skip_prefetch", action="store_true")
    p.add_argument("--skip_cache_extract", action="store_true")
    p.add_argument("--hf_token", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    elif "HF_TOKEN" not in os.environ:
        # Kaggle secrets optional
        try:
            from kaggle_secrets import UserSecretsClient

            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
            print("Loaded HF_TOKEN from Kaggle secrets.")
        except Exception:
            print("HF_TOKEN not set. Common Voice (gated) may be skipped; FLEURS still works.")

    os.environ.setdefault("WAXAL_OUTPUTS_DIR", str(root / "outputs"))
    Path(os.environ["WAXAL_OUTPUTS_DIR"]).mkdir(parents=True, exist_ok=True)

    use_tpu = args.tpu or detect_tpu()
    print(f"=== WAXAL MMS Lightning trainer | TPU={use_tpu} | lang={args.lang} ===")

    # Always pin HF_HOME before any HF import/download (even if extract is skipped).
    configure_kaggle_hf_home()

    if not args.skip_cache_extract:
        try:
            extract_cache_if_present()
        except Exception as exc:
            print(f"Cache step warning: {exc}")

    ensure_project_deps_light()
    ensure_lightning()

    if args.prefetch_external and not args.skip_prefetch:
        print("=== Prefetching external corpora (Common Voice + FLEURS) ===")
        subprocess.check_call(
            [sys.executable, "scripts/prefetch_external_data.py", "--langs", "lin,sna,lug"]
        )

    langs = ["lin", "sna", "lug"] if args.lang == "all" else [args.lang]
    # Aggressive default step budgets when user wants to burn weekly quota on all langs.
    # Override with --max_steps / --epochs.
    default_steps = {"lin": 5000, "sna": 5000, "lug": 3500}

    for lang in langs:
        print("")
        print("=" * 60)
        print(f"Training language: {lang}")
        print("=" * 60)
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
        ]
        if use_tpu:
            cmd.append("--tpu")
        if args.max_steps and args.max_steps > 0:
            cmd.extend(["--max_steps", str(args.max_steps)])
        elif args.epochs and args.epochs > 0:
            cmd.extend(["--epochs", str(args.epochs)])
        else:
            # Default: high step budget to use TPU time productively
            cmd.extend(["--max_steps", str(default_steps.get(lang, 4000))])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("WAXAL_OUTPUTS_DIR", os.environ.get("WAXAL_OUTPUTS_DIR", str(root / "outputs")))
        print("Running:", " ".join(cmd))
        ret = subprocess.call(cmd, env=env)
        if ret != 0:
            print(f"ERROR: training failed for {lang} with exit code {ret}")
            # Continue other languages so one failure does not waste remaining quota
            continue
        print(f"Finished {lang}. Checkpoint expected at outputs/{lang}_mms-300m_fold{args.fold}/best_model")

    print("")
    print("All requested languages processed.")
    print("Next: python generate_submission.py")


if __name__ == "__main__":
    main()
