# ==============================================================================
# WAXAL ASR — Kaggle TPU v5e-8 one-cell launcher (paste into a Kaggle Notebook)
# ==============================================================================
# Setup before Run:
#   1. Accelerator: TPU v5e-8  (v3-8 is phased out on Kaggle)
#   2. Attach dataset: cashgenenator/waxal-hf-cache-chunks  (optional but faster)
#   3. Add Kaggle secret HF_TOKEN (required for Common Voice; FLEURS works without)
#   4. Internet ON
#
# This cell:
#   - clones / updates the repo
#   - does NOT reinstall torch / torch_xla (Kaggle image already has TPU wheels)
#   - prefetches external corpora
#   - trains lin → sna → lug with the minimal Lightning MMS path
# ==============================================================================
import os
import subprocess
import sys

WORKING = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
REPO = os.path.join(WORKING, "WAXAL_ZINDI")
REPO_URL = "https://github.com/AstralJugs69/WAXAL_ZINDI.git"

# --- HF token from secrets ---
hf_token = os.environ.get("HF_TOKEN")
if not hf_token and os.path.exists("/kaggle"):
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"HF_TOKEN secret not available: {e}")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# --- clone / update ---
os.chdir(WORKING)
if os.path.exists(REPO):
    print("Updating existing clone...")
    subprocess.run(["git", "-C", REPO, "pull"], check=False)
else:
    print("Cloning repo...")
    subprocess.run(["git", "clone", REPO_URL, REPO], check=True)

os.chdir(REPO)
sys.path.insert(0, REPO)

# Persist outputs in working dir
os.environ["WAXAL_OUTPUTS_DIR"] = os.path.join(REPO, "outputs")
os.makedirs(os.environ["WAXAL_OUTPUTS_DIR"], exist_ok=True)

# HF cache on /kaggle/temp (larger, not counted against 20GB working)
if os.path.exists("/kaggle/temp"):
    os.environ["HF_HOME"] = "/kaggle/temp/hf_home"
else:
    os.environ["HF_HOME"] = "/tmp/hf_home"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")

print("=== Launching run_tpu_train.py for ALL languages ===")
# Default budgets burn meaningful TPU time (override as needed):
#   lin/sna ~5000 steps, lug ~3500 steps with batch 4 * 8 cores
cmd = [
    sys.executable,
    "run_tpu_train.py",
    "--lang", "all",
    "--tpu",
    "--devices", "8",
    "--fold", "0",
    "--config", "config/base_mms_tpu.yaml",
]
if hf_token:
    cmd.extend(["--hf_token", hf_token])

print("Running:", " ".join(cmd))
ret = subprocess.call(cmd)
print("run_tpu_train exit code:", ret)

# After training, optionally generate submission (needs GPU or long CPU time)
# subprocess.call([sys.executable, "generate_submission.py"])
