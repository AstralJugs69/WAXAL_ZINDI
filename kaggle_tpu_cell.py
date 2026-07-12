# ==============================================================================
# WAXAL ASR — ONE CELL, END-TO-END (Kaggle TPU v5e-8)
# ==============================================================================
# Handover "bootstrapping culture": every session is ephemeral, so this cell
# rebuilds the world from scratch in order:
#
#   0) secrets + disk layout
#   1) fresh git clone of main (or pull)
#   2) HF_HOME → /kaggle/temp (NEVER /kaggle/working — 20GB limit)
#   3) stream-extract cashgenenator/waxal-hf-cache-chunks (split tar → HF cache)
#   4) light Python deps (do NOT reinstall torch/torch_xla — Kaggle TPU image)
#   5) optional KenLM compile (for post-train LM)
#   6) prefetch external corpora (Common Voice + FLEURS)
#   7) train MMS-300M fold0 for lin → sna → lug (Lightning, 8 TPU cores)
#   8) generate submission.csv (MMS-first, blank-ID retry)
#
# Notebook setup BEFORE Run All:
#   • Accelerator: TPU v5e-8
#   • Input data:  cashgenenator/waxal-hf-cache-chunks
#   • Secrets:     HF_TOKEN  (accept Common Voice license on HF)
#   • Internet:    ON
# ==============================================================================

import os
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Tunables (edit here if you want different quota burn)
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/AstralJugs69/WAXAL_ZINDI.git"
FOLD = 0
DEVICES = 8
# High step budgets to use weekly TPU quota productively
LANG_STEPS = {"lin": 5000, "sna": 5000, "lug": 3500}
CONFIG = "config/base_mms_tpu.yaml"
RUN_SUBMISSION = True  # set False if you only want checkpoints this session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)

def run(cmd, cwd=None, check=True, env=None):
    """Stream subprocess output live (handover-style; no hidden hangs)."""
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
log("WAXAL ASR one-cell bootstrap | TPU-first | MMS-300M | lin+sna+lug")
log("=" * 72)
disk_report("/kaggle/working")
disk_report("/kaggle/temp")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token and os.path.exists("/kaggle"):
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        log("HF_TOKEN loaded from Kaggle Secrets.")
    except Exception as exc:
        log(f"HF_TOKEN secret missing ({exc}). Common Voice may be skipped; FLEURS still works.")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# HF cache MUST live outside /kaggle/working (20GB quota)
if os.path.exists("/kaggle/temp"):
    HF_HOME = "/kaggle/temp/hf_home"
else:
    HF_HOME = "/tmp/hf_home"
    log("WARNING: /kaggle/temp not found — using /tmp/hf_home (risk of OOM disk).")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["WAXAL_OUTPUTS_DIR"] = os.path.join(REPO, "outputs")  # set after clone path known
log(f"HF_HOME={HF_HOME}")
log(f"HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}")
log(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")

# ---------------------------------------------------------------------------
# 1) Fresh code (handover: reproducible clone every session)
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
# 2) Extract split-tar HF cache (cashgenenator/waxal-hf-cache-chunks)
#    /kaggle/input/.../hf_cache.tar.aa..am  →  stream pipe →  $HF_HOME
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
        log("Streaming split-tar extract into HF_HOME (no intermediate 54GB file)…")
        extract_cache_chunks(chunks_dir, HF_HOME)
        assert check_extraction_valid(HF_HOME), "Extract finished but hub/datasets invalid"
        log("Cache extraction OK.")
    else:
        log(
            "NOTICE: hf_cache.tar.aa not found under /kaggle/input.\n"
            "  Attach dataset cashgenenator/waxal-hf-cache-chunks for fast start.\n"
            "  Continuing with live Hugging Face Hub downloads (slower)."
        )
disk_report("/kaggle/temp")

# ---------------------------------------------------------------------------
# 3) Light deps — NEVER reinstall torch / torch_xla on Kaggle TPU
# ---------------------------------------------------------------------------
log("Installing Python deps (lightning + ASR stack; torch left as Kaggle image provides)…")
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

# KenLM (optional — post-train LM). Fail soft if cmake/build tools missing.
try:
    log("Compiling KenLM (optional)…")
    run([sys.executable, "-c",
         "from src.decoding.kenlm_utils import compile_kenlm; compile_kenlm('kenlm')"],
        check=False)
except Exception as exc:
    log(f"KenLM compile skipped: {exc}")

# ---------------------------------------------------------------------------
# 4) Prefetch external corpora (Common Voice + FLEURS) into HF cache
# ---------------------------------------------------------------------------
log("Prefetching external corpora for lin/sna/lug…")
run([sys.executable, "scripts/prefetch_external_data.py", "--langs", "lin,sna,lug"], check=False)

# ---------------------------------------------------------------------------
# 5) Train all three languages (speaker-independent fold0, Lightning TPU)
# ---------------------------------------------------------------------------
tpu_present = bool(
    os.environ.get("TPU_NAME")
    or os.environ.get("TPU_ACCELERATOR_TYPE")
    or os.path.exists("/usr/share/tpu-support")
    or os.path.exists("/dev/accel0")
)
log(f"TPU hardware detected: {tpu_present}")

env = {
    "HF_HOME": os.environ["HF_HOME"],
    "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"],
    "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
    "WAXAL_OUTPUTS_DIR": OUT,
    "PYTHONPATH": REPO,
    "TOKENIZERS_PARALLELISM": "false",
    "PJRT_DEVICE": "TPU",
}
if hf_token:
    env["HF_TOKEN"] = hf_token

for lang, steps in LANG_STEPS.items():
    log("")
    log("=" * 72)
    log(f"TRAIN {lang} | fold={FOLD} | max_steps={steps} | devices={DEVICES}")
    log("=" * 72)
    cmd = [
        sys.executable, "-m", "src.training.lightning_mms",
        "--config", CONFIG,
        "--target_lang", lang,
        "--fold", str(FOLD),
        "--devices", str(DEVICES),
        "--max_steps", str(steps),
        "--tpu",
    ]
    # Continue other languages if one fails so remaining TPU quota is not wasted
    rc = run(cmd, env=env, check=False)
    ckpt = os.path.join(OUT, f"{lang}_mms-300m_fold{FOLD}", "best_model")
    log(f"[{lang}] exit={rc} checkpoint_dir={ckpt} exists={os.path.isdir(ckpt)}")

# ---------------------------------------------------------------------------
# 6) Submission (VAD + per-lang MMS route + blank retry)
# ---------------------------------------------------------------------------
if RUN_SUBMISSION:
    log("")
    log("=" * 72)
    log("GENERATE submission.csv")
    log("=" * 72)
    # Prefer a slightly looser blank threshold on first full run
    run([sys.executable, "generate_submission.py", "--max-blank-frac", "0.05"], env=env, check=False)
    sub_path = os.path.join(REPO, "submission.csv")
    if os.path.isfile(sub_path):
        # Copy to /kaggle/working root for easy Download
        dest = os.path.join(WORKING, "submission.csv")
        shutil.copy2(sub_path, dest)
        log(f"submission.csv ready: {dest}")
    else:
        log("WARNING: submission.csv was not produced.")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
elapsed = (time.time() - t0) / 60.0
log("")
log("=" * 72)
log(f"ONE-CELL PIPELINE FINISHED in {elapsed:.1f} min")
log(f"Checkpoints under: {OUT}/{{lin,sna,lug}}_mms-300m_fold{FOLD}/best_model/")
log(f"Submission: {WORKING}/submission.csv (if generated)")
log("Download submission.csv from the notebook Output panel → upload to Zindi.")
log("=" * 72)
