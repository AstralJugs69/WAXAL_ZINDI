#!/usr/bin/env python3
"""
CPU-only environment + data preparation for Lightning AI (no training).

Use on a free/cheap CPU machine so GPU credits stay for training:
  1) Point HF caches at a persistent disk
  2) Download + stream-extract cashgenenator/waxal-hf-cache-chunks from Kaggle
  3) Prefetch external corpora (FLEURS + Common Voice if HF_TOKEN works)
  4) Pre-download MMS weights/processor into the cache
  5) Smoke-load WaxalNLP per language + write fold CSVs for fold 0

Example (Lightning Studio terminal):
  export HF_TOKEN=hf_...
  export KAGGLE_USERNAME=...
  export KAGGLE_KEY=...
  python scripts/setup_data_cpu.py
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path


def log(msg: str):
    print(msg, flush=True)


def run(cmd, check=True, env=None):
    log(f"\n>>> {' '.join(cmd)}")
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e)
    if check and r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


def resolve_studio_root() -> Path:
    for p in (
        Path("/teamspace/studios/this_studio"),
        Path.home() / "studio",
        Path.cwd(),
    ):
        if p.exists():
            return p
    return Path.cwd()


def configure_paths(studio: Path) -> dict:
    # Prefer large persistent studio disk; never fill tiny ephemeral /tmp only.
    hf_home = Path(os.environ.get("HF_HOME") or (studio / "hf_home"))
    data_prep = Path(os.environ.get("WAXAL_PREP_DIR") or (studio / "waxal_prep"))
    outputs = Path(os.environ.get("WAXAL_OUTPUTS_DIR") or (studio / "WAXAL_ZINDI" / "outputs"))
    kaggle_dl = Path(os.environ.get("WAXAL_KAGGLE_DL") or (studio / "kaggle_dl"))

    for p in (hf_home, data_prep, outputs, kaggle_dl):
        p.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WAXAL_OUTPUTS_DIR"] = str(outputs)
    os.environ["WAXAL_PREP_DIR"] = str(data_prep)

    log(f"STUDIO      = {studio}")
    log(f"HF_HOME     = {hf_home}")
    log(f"PREP_DIR    = {data_prep}")
    log(f"OUTPUTS     = {outputs}")
    log(f"KAGGLE_DL   = {kaggle_dl}")
    return {
        "studio": studio,
        "hf_home": hf_home,
        "data_prep": data_prep,
        "outputs": outputs,
        "kaggle_dl": kaggle_dl,
    }


def setup_kaggle_credentials():
    home = Path.home()
    dest = home / ".kaggle" / "kaggle.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        import json

        dest.write_text(json.dumps({"username": user, "key": key}), encoding="utf-8")
        try:
            dest.chmod(0o600)
        except Exception:
            pass
        log(f"Wrote Kaggle credentials to {dest}")
        return True

    # Search common places (repo / studio)
    candidates = [
        Path("kaggle.json"),
        Path("../kaggle.json"),
        Path("/teamspace/studios/this_studio/kaggle.json"),
        Path("/teamspace/studios/this_studio/WAXAL_ZINDI/kaggle.json"),
        Path("/teamspace/studios/this_studio/WAXAL_ZINDI/waxal_asr_challenge/kaggle.json"),
        home / "kaggle.json",
        home / ".kaggle" / "kaggle.json",
    ]
    for src in candidates:
        if src.exists():
            shutil.copy2(src, dest)
            try:
                dest.chmod(0o600)
            except Exception:
                pass
            log(f"Copied Kaggle credentials from {src}")
            return True

    log("WARNING: No Kaggle credentials found. Cache dataset download will be skipped.")
    return False


def extract_valid(hf_home: Path) -> bool:
    sentinel = hf_home / "extraction_completed.txt"
    if not sentinel.exists():
        return False
    for sub in ("hub", "datasets"):
        p = hf_home / sub
        if not p.exists() or not any(p.iterdir()):
            return False
    return True


def download_and_extract_kaggle_cache(paths: dict, dataset_id: str):
    hf_home: Path = paths["hf_home"]
    kaggle_dl: Path = paths["kaggle_dl"]

    if extract_valid(hf_home):
        log(f"HF cache already valid at {hf_home} — skip Kaggle download/extract.")
        return

    if not setup_kaggle_credentials():
        log("Skipping Kaggle cache; will rely on HF Hub downloads later.")
        return

    try:
        import kaggle  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=False)

    kaggle_dl.mkdir(parents=True, exist_ok=True)
    log(f"Downloading Kaggle dataset {dataset_id} → {kaggle_dl}")
    # kaggle CLI writes a zip named after the dataset slug
    rc = run(
        ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(kaggle_dl), "--force"],
        check=False,
    )
    if rc != 0:
        log("kaggle CLI failed; trying python API…")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset_id, path=str(kaggle_dl), unzip=False, quiet=False)
        except Exception as exc:
            log(f"ERROR: Kaggle download failed ({exc}). Continue without cache.")
            return

    # Find zip
    zips = sorted(kaggle_dl.glob("*.zip"))
    if not zips:
        # maybe already extracted chunks present
        chunks_here = sorted(kaggle_dl.glob("hf_cache.tar.a*"))
        if chunks_here:
            chunks_dir = kaggle_dl
        else:
            log(f"ERROR: No zip or chunks in {kaggle_dl}")
            return
    else:
        zip_path = zips[0]
        extracted = kaggle_dl / "extracted"
        if extracted.exists():
            shutil.rmtree(extracted, ignore_errors=True)
        extracted.mkdir(parents=True, exist_ok=True)
        log(f"Unzipping {zip_path.name}…")
        run(["unzip", "-q", "-o", str(zip_path), "-d", str(extracted)], check=False)
        # chunks may be nested one level
        chunks_dir = extracted
        if not list(extracted.glob("hf_cache.tar.a*")):
            for root, _, files in os.walk(extracted):
                if "hf_cache.tar.aa" in files:
                    chunks_dir = Path(root)
                    break

    chunks = sorted(chunks_dir.glob("hf_cache.tar.a*"))
    if not chunks:
        log(f"ERROR: No hf_cache.tar.a* under {chunks_dir}")
        return

    log(f"Streaming {len(chunks)} chunks into {hf_home} (no intermediate 54GB concat)…")
    hf_home.mkdir(parents=True, exist_ok=True)
    tar = subprocess.Popen(["tar", "-xf", "-", "-C", str(hf_home)], stdin=subprocess.PIPE)
    assert tar.stdin is not None
    try:
        for ch in chunks:
            log(f"  streaming {ch.name}…")
            with open(ch, "rb") as f:
                shutil.copyfileobj(f, tar.stdin)
        tar.stdin.close()
        ret = tar.wait()
        if ret != 0:
            raise RuntimeError(f"tar extract failed code={ret}")
    except Exception as exc:
        if tar.poll() is None:
            tar.terminate()
        log(f"ERROR: extract failed: {exc}")
        return

    (hf_home / "extraction_completed.txt").write_text("ok\n", encoding="utf-8")
    log("Cache extraction OK.")

    # Free download zip to reclaim disk (keep extracted chunks optional)
    try:
        for z in kaggle_dl.glob("*.zip"):
            z.unlink(missing_ok=True)
            log(f"Removed {z.name} to free disk.")
    except Exception:
        pass


def install_python_deps():
    log("Installing light CPU prep deps (no torch reinstall)…")
    pkgs = [
        "kaggle",
        "hf_transfer",
        "datasets>=2.19.0,<4.0.0",
        "transformers>=4.40.0",
        "librosa",
        "soundfile",
        "pyyaml",
        "scikit-learn",
        "pandas",
        "tqdm",
        "jiwer",
        "accelerate",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def prefetch_external(repo: Path):
    log("Prefetching external corpora (FLEURS + Common Voice if gated access works)…")
    script = repo / "scripts" / "prefetch_external_data.py"
    if not script.exists():
        log(f"Missing {script}; skip external prefetch.")
        return
    run(
        [sys.executable, str(script), "--langs", "lin,sna,lug"],
        check=False,
        env={"PYTHONPATH": str(repo)},
    )


def prefetch_models(token: str | None):
    log("Pre-downloading MMS backbone + per-lang processors into HF cache…")
    code = r"""
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
token = os.environ.get("HF_TOKEN")
print("Loading facebook/mms-300m …")
Wav2Vec2ForCTC.from_pretrained("facebook/mms-300m", token=token)
for lang in ("lin", "sna", "lug"):
    print(f"Loading processor mms-1b-all target_lang={lang} …")
    p = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all", target_lang=lang, token=token)
    p.tokenizer.set_target_lang(lang)
print("Model/processor cache warm OK.")
"""
    env = {"HF_HOME": os.environ["HF_HOME"], "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"]}
    if token:
        env["HF_TOKEN"] = token
    run([sys.executable, "-c", code], check=False, env=env)


def prepare_fold_csvs(repo: Path, paths: dict, fold: int = 0):
    """Write per-language fold CSVs without decoding audio (low RAM)."""
    log("Writing fold assignment CSVs (no audio decode)…")
    sys.path.insert(0, str(repo))
    from src.data.dataset import parse_robust_csv, normalize_text
    from sklearn.model_selection import GroupKFold

    train_csv = repo / "Train.csv"
    if not train_csv.exists():
        # try parent
        alt = repo.parent / "Train.csv"
        train_csv = alt if alt.exists() else train_csv
    if not train_csv.exists():
        log(f"WARNING: Train.csv not found at {train_csv}; skip fold CSVs.")
        return

    df = parse_robust_csv(str(train_csv))
    df["normalized_transcription"] = df["transcription"].apply(normalize_text)
    df = df[df["normalized_transcription"].str.strip() != ""].reset_index(drop=True)

    prep = paths["data_prep"]
    for lang in ("lin", "sna", "lug"):
        sub = df[df["language"] == lang].reset_index(drop=True)
        if len(sub) == 0:
            log(f"  {lang}: no rows")
            continue
        sub = sub.copy()
        sub["speaker_id"] = sub["id"].astype(str).map(lambda x: f"spk_{hash(x) % 2000}")
        gkf = GroupKFold(n_splits=5)
        sub["fold"] = -1
        for fi, (_, val_idx) in enumerate(
            gkf.split(sub, sub["normalized_transcription"], sub["speaker_id"])
        ):
            sub.iloc[val_idx, sub.columns.get_loc("fold")] = fi
        train_part = sub[sub["fold"] != fold]
        val_part = sub[sub["fold"] == fold]
        tpath = prep / f"temp_train_{lang}_fold{fold}.csv"
        vpath = prep / f"temp_val_{lang}_fold{fold}.csv"
        train_part.to_csv(tpath, index=False)
        val_part.to_csv(vpath, index=False)
        log(f"  {lang}: train={len(train_part)} val={len(val_part)} → {tpath.name}, {vpath.name}")
    gc.collect()


def smoke_load_waxal(repo: Path):
    log("Smoke-loading WaxalNLP parquet configs (metadata only)…")
    sys.path.insert(0, str(repo))
    from src.data.dataset import load_waxal_dataset_clean

    for lang in ("lin", "sna", "lug"):
        try:
            ds = load_waxal_dataset_clean(lang)
            n_tr = len(ds["train"]) if "train" in ds else 0
            n_va = len(ds["validation"]) if "validation" in ds else 0
            log(f"  {lang}: train={n_tr} validation={n_va} cols={ds['train'].column_names[:8] if n_tr else []}")
            del ds
            gc.collect()
        except Exception as exc:
            log(f"  {lang}: FAILED ({exc})")


def disk_report(path: Path):
    try:
        usage = shutil.disk_usage(path)
        log(
            f"[disk] {path} used={usage.used/1e9:.1f}GB free={usage.free/1e9:.1f}GB "
            f"total={usage.total/1e9:.1f}GB"
        )
    except Exception as exc:
        log(f"[disk] {path}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="CPU-only WAXAL data prep (no training)")
    parser.add_argument("--dataset", default="cashgenenator/waxal-hf-cache-chunks")
    parser.add_argument("--skip-kaggle", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-folds", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--repo", type=str, default=None, help="Path to WAXAL_ZINDI checkout")
    args = parser.parse_args()

    studio = resolve_studio_root()
    paths = configure_paths(studio)
    disk_report(studio)

    # Resolve repo root (script may live in scripts/)
    if args.repo:
        repo = Path(args.repo).resolve()
    else:
        here = Path(__file__).resolve()
        repo = here.parents[1] if here.parent.name == "scripts" else Path.cwd()
    if not (repo / "src").exists():
        # common Lightning layout
        cand = studio / "WAXAL_ZINDI"
        if (cand / "src").exists():
            repo = cand
    log(f"REPO        = {repo}")
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    token = os.environ.get("HF_TOKEN")
    if not token:
        log("WARNING: HF_TOKEN unset — Common Voice may fail; FLEURS/MMS usually still work.")

    install_python_deps()

    if not args.skip_kaggle:
        download_and_extract_kaggle_cache(paths, args.dataset)
        disk_report(studio)
    else:
        log("Skipping Kaggle cache step.")

    if not args.skip_models:
        prefetch_models(token)
    else:
        log("Skipping model cache warm.")

    if not args.skip_external:
        prefetch_external(repo)
    else:
        log("Skipping external corpora.")

    smoke_load_waxal(repo)

    if not args.skip_folds:
        prepare_fold_csvs(repo, paths, fold=args.fold)
    else:
        log("Skipping fold CSV write.")

    log("")
    log("=" * 72)
    log("CPU DATA PREP COMPLETE — no training was run.")
    log(f"  HF cache:     {paths['hf_home']}")
    log(f"  Fold CSVs:    {paths['data_prep']}")
    log(f"  Outputs dir:  {paths['outputs']}")
    log("Next (on GPU machine, same studio disk):")
    log("  export HF_HOME=/teamspace/studios/this_studio/hf_home")
    log("  export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs")
    log("  python -m src.training.lightning_mms --config config/base_mms_lightning_96gb.yaml \\")
    log("      --target_lang lin --fold 0 --devices 1 --max_steps 5000")
    log("=" * 72)


if __name__ == "__main__":
    main()
