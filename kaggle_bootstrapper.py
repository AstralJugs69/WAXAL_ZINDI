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
        bufsize=0, # Unbuffered pipe
        cwd=cwd
    )
    
    # Read output character-by-character to stream progress bars (\r) and logs (\n) in real-time
    while True:
        char = process.stdout.read(1)
        if not char:
            break
        sys.stdout.write(char)
        if char in ('\n', '\r'):
            sys.stdout.flush()
            
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

def find_cache_chunks_dir():
    """
    Recursively scans /kaggle/input, /content, and current directory
    to locate the folder containing the hf_cache.tar.aa chunk.
    """
    search_roots = ["/kaggle/input", "/content", "."]
    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            # Skip code repo itself to avoid matching downloaded assets recursively
            if ".git" in root or "hf_home" in root or "WAXAL_ZINDI" in root:
                continue
            if "hf_cache.tar.aa" in files:
                print(f"Discovered cache chunks folder at: {root}")
                return root
    return None

def check_extraction_valid(hf_home_dir):
    """
    Checks if the cache extraction was already performed and is valid.
    """
    sentinel_path = os.path.join(hf_home_dir, "extraction_completed.txt")
    if not os.path.exists(sentinel_path):
        return False
    # Verify subdirectories exist and have content
    for sub in ["hub", "datasets"]:
        sub_path = os.path.join(hf_home_dir, sub)
        if not os.path.exists(sub_path) or not os.listdir(sub_path):
            return False
    return True

def extract_cache_chunks(chunks_dir, hf_home_dir):
    """
    Streams and extracts split tar chunks directly into the tar pipe
    to prevent disk I/O bottlenecks and OOM errors.
    """
    chunks = sorted([f for f in os.listdir(chunks_dir) if f.startswith("hf_cache.tar.a")])
    if not chunks:
        raise FileNotFoundError(f"No hf_cache.tar.a* chunks found in {chunks_dir}")
        
    chunk_paths = [os.path.join(chunks_dir, c) for c in chunks]
    print(f"Extracting {len(chunks)} chunks directly via pipe into {hf_home_dir}...")
    os.makedirs(hf_home_dir, exist_ok=True)
    
    tar_cmd = ["tar", "-xf", "-", "-C", hf_home_dir]
    tar_proc = subprocess.Popen(tar_cmd, stdin=subprocess.PIPE)
    
    try:
        for chunk_path in chunk_paths:
            print(f"  Streaming {os.path.basename(chunk_path)}...")
            with open(chunk_path, "rb") as infile:
                shutil.copyfileobj(infile, tar_proc.stdin)
        # Close stdin to signal EOF to tar
        tar_proc.stdin.close()
        ret = tar_proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, tar_cmd)
    except Exception as e:
        print(f"Extraction failed: {e}")
        if tar_proc.poll() is None:
            tar_proc.terminate()
            tar_proc.wait()
        sentinel_path = os.path.join(hf_home_dir, "extraction_completed.txt")
        if os.path.exists(sentinel_path):
            os.remove(sentinel_path)
        raise e
        
    sentinel_path = os.path.join(hf_home_dir, "extraction_completed.txt")
    with open(sentinel_path, "w") as f:
        f.write("extraction completed successfully")
        
    print("HuggingFace cache extraction completed successfully!")

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

    # Configure HuggingFace cache to persist outside of /kaggle/working to bypass the 20GB output disk limit on Kaggle
    if os.path.exists("/kaggle/temp"):
        hf_home = "/kaggle/temp/hf_home"
    elif os.path.exists("/kaggle/working"):
        hf_home = "/tmp/hf_home"
    else:
        hf_home = "/content/hf_home"
        
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
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
    for script in ["install_dependencies.sh", "run_training.sh", "bootstrap_and_train.py", "run_kaggle.py", "run_lightning.py"]:
        script_path = os.path.join("scripts", script)
        if os.path.exists(script_path):
            os.chmod(script_path, 0o755)
        # Check root of repository as well
        root_script_path = os.path.join(".", script)
        if os.path.exists(root_script_path):
            os.chmod(root_script_path, 0o755)

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

    print("\n=== Step 3.5: Extracting HuggingFace Cache Chunks ===")
    chunks_dir = find_cache_chunks_dir()
    if chunks_dir:
        if check_extraction_valid(hf_home):
            print(f"Valid extraction sentinel found at {hf_home}. Skipping extraction.")
        else:
            try:
                extract_cache_chunks(chunks_dir, hf_home)
            except Exception as e:
                print(f"Warning: Cache extraction failed: {e}. Training will continue using default Hugging Face download.")
    else:
        print("Notice: hf_cache.tar.aa chunk was not found in standard directories. Skipping local cache extraction.")

    print("\n=== Step 4: Kickstarting Model Training Pipeline ===")
    # Run the multilingual sequential runner for all three languages
    run_command_live([
        "bash", "run_all_languages.sh",
        "config/base_gemma.yaml",
        "0",
        hf_token or "",
        os.environ.get("KAGGLE_USERNAME", ""),
        os.environ.get("KAGGLE_KEY", "")
    ])

    print("\n=== Bootstrapping and Training Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
