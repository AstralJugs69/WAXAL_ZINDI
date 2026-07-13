# ==============================================================================
# WAXAL ASR — ONE CELL, END-TO-END (Kaggle GPU 2× T4)
# ==============================================================================
# Run this in a SEPARATE GPU notebook while the TPU notebook is also running.
# Sessions do not share /kaggle/working or /kaggle/temp — fully independent.
#
# Pipeline (same bootstrap culture as TPU cell):
#   0) secrets + disk layout
#   1) fresh git clone of main
#   2) HF_HOME → /kaggle/temp (NEVER /kaggle/working)
#   3) stream-extract cashgenenator/waxal-hf-cache-chunks
#   4) light Python deps (do NOT reinstall torch — keep Kaggle CUDA wheels)
#   5) optional KenLM
#   6) prefetch external corpora
#   7) train MMS-300M fold0 for lin → sna → lug (Lightning DDP, 2× T4)
#   8) generate submission.csv
#
# Notebook setup BEFORE Run:
#   • Accelerator: GPU T4 x2
#   • Input data:  cashgenenator/waxal-hf-cache-chunks
#   • Secrets:     HF_TOKEN
#   • Internet:    ON
#
# Parallel strategy tip:
#   TPU notebook → train languages A (heavy steps)
#   GPU notebook → train other fold / complementary steps / or same langs as backup
#   Default below trains fold 0 with GPU-friendly step counts + builds submission
#   (submission is much happier on GPU than on TPU host CPU).
# ==============================================================================

import os
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/AstralJugs69/WAXAL_ZINDI.git"
FOLD = 0
# devices=1 avoids Lightning DDP forking 2 processes that each load the full
# dataset (~2× system RAM). Kaggle GPU RAM is ~30GB — dual-process OOMs before train.
DEVICES = 1
LANG_STEPS = {"lin": 4000, "sna": 4000, "lug": 2500}
CONFIG = "config/base_mms_gpu.yaml"
RUN_SUBMISSION = True
SKIP_EXISTING = True
# Do NOT prefetch CV/FLEURS up-front on 30GB RAM — loads multi-GB tables.
SKIP_PREFETCH = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)

def run(cmd, cwd=None, check=True, env=None):
    log(f"\n>>> {' '.join(cmd)}")
    e = os.environ.copy()
    if env:
        e.update(env)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=e,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode

def disk_report(path="/kaggle"):
    try:
        total, used, free = shutil.disk_usage(path if os.path.exists(path) else "/")
        log(f"[disk] {path} used={used/1e9:.1f}GB free={free/1e9:.1f}GB total={total/1e9:.1f}GB")
    except Exception as exc:
        log(f"[disk] unavailable: {exc}")

# ---------------------------------------------------------------------------
# 0) Paths + secrets
# ---------------------------------------------------------------------------
WORKING = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
REPO = os.path.join(WORKING, "WAXAL_ZINDI")
t0 = time.time()
log("=" * 72)
log("WAXAL ASR one-cell bootstrap | GPU 2×T4 | MMS-300M | lin+sna+lug")
log("=" * 72)
disk_report("/kaggle/working")
disk_report("/kaggle/temp")

# CUDA sanity
try:
    import torch
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpus={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"  gpu[{i}]={torch.cuda.get_device_name(i)}")
except Exception as exc:
    log(f"torch import warning: {exc}")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token and os.path.exists("/kaggle"):
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        log("HF_TOKEN loaded from Kaggle Secrets.")
    except Exception as exc:
        log(f"HF_TOKEN secret missing ({exc}). Common Voice may be skipped.")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

if os.path.exists("/kaggle/temp"):
    HF_HOME = "/kaggle/temp/hf_home"
else:
    HF_HOME = "/tmp/hf_home"
    log("WARNING: /kaggle/temp missing — using /tmp/hf_home")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Ensure we do NOT force TPU mode if this image has leftover TPU env vars
for k in list(os.environ.keys()):
    if k.startswith("TPU_") or k in ("PJRT_DEVICE", "XRT_TPU_CONFIG"):
        # Keep harmless vars; clear device force so Lightning stays on GPU
        if k in ("PJRT_DEVICE",):
            os.environ.pop(k, None)

log(f"HF_HOME={HF_HOME}")

# ---------------------------------------------------------------------------
# 1) Fresh code
# ---------------------------------------------------------------------------
os.chdir(WORKING)
if os.path.exists(REPO):
    log(f"Removing old clone for clean bootstrap: {REPO}")
    shutil.rmtree(REPO, ignore_errors=True)
log(f"Cloning {REPO_URL} …")
run(["git", "clone", "--depth", "1", REPO_URL, REPO])
os.chdir(REPO)
sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "outputs")
os.makedirs(OUT, exist_ok=True)
os.environ["WAXAL_OUTPUTS_DIR"] = OUT
log(f"WAXAL_OUTPUTS_DIR={OUT}")
log(f"Repo HEAD: {subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()}")

# ---------------------------------------------------------------------------
# 2) Extract split-tar HF cache
# ---------------------------------------------------------------------------
from kaggle_bootstrapper import (  # noqa: E402
    check_extraction_valid,
    extract_cache_chunks,
    find_cache_chunks_dir,
)

if check_extraction_valid(HF_HOME):
    log(f"HF cache already valid at {HF_HOME} — skip extract.")
else:
    chunks_dir = find_cache_chunks_dir()
    if chunks_dir:
        names = sorted(f for f in os.listdir(chunks_dir) if f.startswith("hf_cache.tar.a"))
        log(f"Found {len(names)} cache chunks in {chunks_dir}: {names[0]} … {names[-1]}")
        log("Streaming split-tar extract into HF_HOME…")
        extract_cache_chunks(chunks_dir, HF_HOME)
        assert check_extraction_valid(HF_HOME), "Extract finished but hub/datasets invalid"
        log("Cache extraction OK.")
    else:
        log(
            "NOTICE: hf_cache.tar.aa not found. Attach cashgenenator/waxal-hf-cache-chunks.\n"
            "  Continuing with Hugging Face Hub downloads (slower)."
        )
disk_report("/kaggle/temp")

# ---------------------------------------------------------------------------
# 3) Light deps — NEVER reinstall torch on Kaggle GPU
# ---------------------------------------------------------------------------
log("Installing Python deps (keep preinstalled CUDA torch)…")
run([
    sys.executable, "-m", "pip", "install", "-q",
    "lightning>=2.2.0",
    "transformers>=4.40.0",
    "datasets>=2.19.0,<4.0.0",
    "accelerate",
    "jiwer",
    "librosa",
    "soundfile",
    "pyyaml",
    "scikit-learn",
    "evaluate",
    "hf_transfer",
    "pyctcdecode",
], check=False)

# Optional KenLM — soft-fail; training does not need it.
log("Compiling KenLM (optional; greedy CTC if this fails)…")
run(
    [
        sys.executable,
        "-c",
        (
            "from src.decoding.kenlm_utils import compile_kenlm; "
            "p=compile_kenlm('kenlm', strict=False); "
            "print('KenLM OK' if p[0] else 'KenLM skipped (no boost libs) — training continues')"
        ),
    ],
    check=False,
)

# ---------------------------------------------------------------------------
# 4) Prefetch external corpora (optional — OOMs 30GB RAM if enabled)
# ---------------------------------------------------------------------------
if not SKIP_PREFETCH:
    log("Prefetching external corpora…")
    run([sys.executable, "scripts/prefetch_external_data.py", "--langs", "lin,sna,lug"], check=False)
else:
    log("SKIP_PREFETCH=True — training WAXAL-only (fits ~30GB system RAM).")

# ---------------------------------------------------------------------------
# 5) Train languages on GPU (NO --tpu). Default 1 process to save RAM.
# ---------------------------------------------------------------------------
env = {
    "HF_HOME": os.environ["HF_HOME"],
    "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"],
    "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
    "WAXAL_OUTPUTS_DIR": OUT,
    "PYTHONPATH": REPO,
    "TOKENIZERS_PARALLELISM": "false",
    "HF_DATASETS_IN_MEMORY_MAX_SIZE": "0",
    # Single visible GPU when DEVICES=1; both when using 2 (not recommended for RAM)
    "CUDA_VISIBLE_DEVICES": "0" if DEVICES <= 1 else "0,1",
}
if hf_token:
    env["HF_TOKEN"] = hf_token

for lang, steps in LANG_STEPS.items():
    ckpt = os.path.join(OUT, f"{lang}_mms-300m_fold{FOLD}", "best_model")
    if SKIP_EXISTING and os.path.isdir(ckpt) and os.listdir(ckpt):
        log(f"SKIP {lang}: existing checkpoint at {ckpt}")
        continue

    log("")
    log("=" * 72)
    log(f"TRAIN {lang} | fold={FOLD} | max_steps={steps} | devices={DEVICES} (GPU)")
    log("=" * 72)
    cmd = [
        sys.executable, "-m", "src.training.lightning_mms",
        "--config", CONFIG,
        "--target_lang", lang,
        "--fold", str(FOLD),
        "--devices", str(DEVICES),
        "--max_steps", str(steps),
        # deliberately NO --tpu
    ]
    rc = run(cmd, env=env, check=False)
    log(f"[{lang}] exit={rc} checkpoint_dir={ckpt} exists={os.path.isdir(ckpt)}")

# ---------------------------------------------------------------------------
# 6) Submission (GPU is the better host for inference)
# ---------------------------------------------------------------------------
if RUN_SUBMISSION:
    log("")
    log("=" * 72)
    log("GENERATE submission.csv (GPU-friendly)")
    log("=" * 72)
    run([sys.executable, "generate_submission.py", "--max-blank-frac", "0.05"], env=env, check=False)
    sub_path = os.path.join(REPO, "submission.csv")
    if os.path.isfile(sub_path):
        dest = os.path.join(WORKING, "submission.csv")
        shutil.copy2(sub_path, dest)
        log(f"submission.csv ready: {dest}")
    else:
        log("WARNING: submission.csv was not produced.")

elapsed = (time.time() - t0) / 60.0
log("")
log("=" * 72)
log(f"GPU ONE-CELL FINISHED in {elapsed:.1f} min")
log(f"Checkpoints: {OUT}/{{lin,sna,lug}}_mms-300m_fold{FOLD}/best_model/")
log(f"Submission:  {WORKING}/submission.csv")
log("This session is independent of any parallel TPU notebook.")
log("=" * 72)
