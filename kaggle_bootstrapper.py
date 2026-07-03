# ==============================================================================
# KAGGLE BOOTSTRAPPER & PIPELINE RUNNER (Run this in a Kaggle Notebook Cell)
# ==============================================================================
import os
import shutil
import subprocess
import sys

def run_command_live(cmd, cwd=None):
    """
    Runs a shell command and streams its stdout/stderr live to the console
    to prevent blocking or hidden hangs in notebook environments.
    """
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd
    )
    
    # Read output line-by-line as it streams
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

def main():
    repo_url = "https://github.com/AstralJugs69/WAXAL_ZINDI.git"
    working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
    project_dir = os.path.join(working_dir, "WAXAL_ZINDI")

    # Load HF_TOKEN from Kaggle Secrets or Colab Secrets if configured
    hf_token = None
    if os.path.exists("/kaggle/working"):
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
        except Exception as e:
            print(f"Warning: Failed to load HF_TOKEN from Kaggle Secrets: {e}")
    else:
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
        except Exception as e:
            print(f"Warning: Failed to load HF_TOKEN from Colab Secrets: {e}")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("Successfully configured HF_TOKEN environment variable.")
    else:
        print("Notice: HF_TOKEN env var not set. Gated HuggingFace datasets (Common Voice) will be skipped.")

    # Configure HuggingFace cache to persist on the working disk drive (preserves datasets/models between runs)
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        hf_home = os.path.join(working_dir, "hf_home")
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    else:
        if "HF_DATASETS_CACHE" not in os.environ:
            os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    print(f"HuggingFace cache configured to persist at: {hf_home}")

    print("=== Step 0: Wiping Old Project & Freeing Space ===")
    # Clear old project directory to ensure fresh clone (do NOT wipe hf_home cache)
    if os.path.exists(project_dir):
        print(f"Removing old project directory at: {project_dir}")
        try:
            shutil.rmtree(project_dir)
            print("Project directory cleared successfully.")
        except Exception as e:
            print(f"Warning: Failed to clear project directory: {e}")
            
    # Report available disk space
    try:
        total, used, free = shutil.disk_usage(working_dir)
        print(f"Disk Space Telemetry | Used: {used/(1024**3):.1f}GB | Free: {free/(1024**3):.1f}GB | Total: {total/(1024**3):.1f}GB")
    except Exception:
        pass

    print("\n=== Step 1: Retrieving Codebase ===")
    os.chdir(working_dir)
    run_command_live(["git", "clone", repo_url, project_dir])
    os.chdir(project_dir)
    print(f"Current working directory set to: {os.getcwd()}")

    print("\n=== Step 2: Making Shell Scripts Executable ===")
    for script in ["install_dependencies.sh", "run_training.sh"]:
        script_path = os.path.join("scripts", script)
        if os.path.exists(script_path):
            os.chmod(script_path, 0o755)

    scan_root = "/kaggle" if os.path.exists("/kaggle") else "/content"
    print(f"\n=== Scanning {scan_root} Workspace Files ===")
    for root, dirs, files in os.walk(scan_root):
        # Limit depth to keep logs concise
        depth = root.replace(scan_root, "").count(os.sep)
        if depth <= 3 and ".git" not in root and "__pycache__" not in root:
            print(f"{'  ' * depth}[DIR] {root}")
            for f in files[:10]:
                print(f"{'  ' * (depth + 1)}[FILE] {f}")
                    
    # Detect if TPU environment is active
    tpu_active = False
    if os.environ.get("TPU_NAME") or os.environ.get("TPU_ACCELERATOR_TYPE") or os.path.exists("/usr/share/tpu-support"):
        tpu_active = True
    print(f"\n=== TPU Accelerator Detected: {tpu_active} ===")
    
    if tpu_active:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            print("torch_xla is already installed and importable. Skipping wheels installation.")
        except ImportError:
            print("Installing PyTorch/XLA 2.8.0 wheels for TPU VMs (Python 3.12)...")
            run_command_live([
                "pip", "install", "--upgrade",
                "torch==2.8.0", "torch_xla[tpu]==2.8.0", 
                "-f", "https://storage.googleapis.com/libtpu-releases/index.html"
            ])
        
    # Detect if we are on a Tesla P100 GPU and need to install sm_60 (Pascal) compatible PyTorch wheels
    p100_detected = False
    if not tpu_active:
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0).lower()
                if "p100" in device_name:
                    p100_detected = True
        except Exception:
            pass

    if p100_detected:
        print("\n=== Tesla P100 GPU Detected ===")
        print("Installing sm_60 (Pascal) compatible PyTorch, Torchaudio, and Torchvision wheels...")
        run_command_live([
            "pip", "install", "--force-reinstall",
            "torch", "torchaudio", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])

    print("\n=== Step 3: Installing Dependencies & Compiling KenLM ===")
    run_command_live(["bash", "scripts/install_dependencies.sh"])

    print("\n=== Step 4: Kickstarting Model Training Pipeline ===")
    tpu_flag = "--tpu" if tpu_active else ""
    run_command_live(["bash", "scripts/run_training.sh", "config/base_mms.yaml", "0", "lin", tpu_flag])

    print("\n=== Bootstrapping and Training Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
