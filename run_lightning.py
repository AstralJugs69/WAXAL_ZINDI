#!/usr/bin/env python3
"""
Lightning AI VM Bootstrapper and Training Launcher
--------------------------------------------------
Automates git pulling, Kaggle API setup, cache downloading and extraction,
installing system dependencies, and executing training.
"""
import os
import sys
import shutil
import subprocess

# We dynamically extract the HuggingFace token from CLI arguments to avoid hardcoding secrets
# and triggering GitHub's Push Protection policies.
def extract_hf_token():
    token = os.environ.get("HF_TOKEN")
    
    # Check for --hf_token in sys.argv directly to avoid index mismatch
    if "--hf_token" in sys.argv:
        idx = sys.argv.index("--hf_token")
        if idx + 1 < len(sys.argv):
            token = sys.argv[idx + 1]
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
            
    return token

def self_update():
    """
    Checks for outstanding git updates on main and reloads the process if updated.
    """
    if not os.path.exists(".git"):
        return
    print("=== Step 1: Checking Git Remote for Updates ===")
    try:
        # Fetch latest changes
        subprocess.run(["git", "fetch", "origin"], check=True)
        # Check if local is behind remote
        local_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        remote_hash = subprocess.run(["git", "rev-parse", "@{u}"], capture_output=True, text=True, check=True).stdout.strip()
        
        if local_hash != remote_hash:
            print("Updates found on remote! Pulling changes...")
            subprocess.run(["git", "pull"], check=True)
            print("Code updated. Reloading bootstrapper...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            print("Code is up to date.")
    except Exception as e:
        print(f"Warning: Failed to check for git updates: {e}")

def setup_kaggle_credentials():
    """
    Configures ~/.kaggle/kaggle.json from command line arguments, environment,
    or robust filesystem search.
    """
    print("=== Step 2: Configuring Kaggle Credentials ===")
    home_dir = os.path.expanduser("~")
    kaggle_dir = os.path.join(home_dir, ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    dest_json = os.path.join(kaggle_dir, "kaggle.json")
    
    # 1. Parse command-line args for credentials
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    
    # Use sys.argv directly to prevent index shifts
    if "--kaggle_username" in sys.argv:
        idx = sys.argv.index("--kaggle_username")
        if idx + 1 < len(sys.argv):
            username = sys.argv[idx + 1]
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
            
    if "--kaggle_key" in sys.argv:
        idx = sys.argv.index("--kaggle_key")
        if idx + 1 < len(sys.argv):
            key = sys.argv[idx + 1]
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
            
    if username and key:
        import json
        try:
            with open(dest_json, "w") as f:
                json.dump({"username": username, "key": key}, f)
            os.chmod(dest_json, 0o600)
            print(f"Successfully configured Kaggle credentials via CLI arguments at {dest_json}")
            return
        except Exception as e:
            print(f"Warning: Failed to write Kaggle credentials from arguments: {e}")

    # 2. Search filesystem for kaggle.json
    search_paths = [
        "kaggle.json",
        "../kaggle.json",
        "../../kaggle.json",
        "/teamspace/studios/this_studio/kaggle.json",
        "/teamspace/studios/this_studio/WAXAL_ZINDI/kaggle.json",
        "/teamspace/studios/this_studio/WAXAL_ZINDI/waxal_asr_challenge/kaggle.json",
        os.path.join(home_dir, "kaggle.json")
    ]
    
    src_json = None
    for path in search_paths:
        if os.path.exists(path):
            src_json = path
            break
            
    if src_json:
        print(f"Found kaggle.json at: {src_json}")
        try:
            shutil.copy(src_json, dest_json)
            os.chmod(dest_json, 0o600)
            print(f"Successfully configured Kaggle credentials at {dest_json}")
        except Exception as e:
            print(f"Warning: Failed to copy kaggle.json from {src_json}: {e}")
    else:
        print("Warning: kaggle.json not found in search paths and no CLI credentials provided.")

def download_and_extract_cache(hf_home_dir):
    """
    Downloads the cache chunks from Kaggle using the Kaggle API and extracts them.
    """
    print("=== Step 3: Downloading HF Cache Chunks from Kaggle ===")
    
    # 1. Check if cache already extracted and valid
    sentinel_path = os.path.join(hf_home_dir, "extraction_completed.txt")
    if os.path.exists(sentinel_path):
        print(f"Valid cache extraction sentinel found at {hf_home_dir}. Skipping download and extraction.")
        return

    # 2. Ensure kaggle CLI package is installed
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle API package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "--quiet"], check=True)
        # Re-import
        import kaggle

    # 3. Download the dataset
    dataset_id = "cashgenenator/waxal-hf-cache-chunks"
    download_dir = "/tmp/kaggle_download"
    os.makedirs(download_dir, exist_ok=True)
    
    zip_path = os.path.join(download_dir, "waxal-hf-cache-chunks.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading dataset {dataset_id} via Kaggle API to {download_dir}...")
        try:
            subprocess.run([
                "kaggle", "datasets", "download", "-d", dataset_id, "-p", download_dir
            ], check=True)
        except Exception as e:
            print(f"Error: Kaggle dataset download failed: {e}")
            sys.exit(1)
    else:
        print(f"Found existing zip archive at {zip_path}.")

    # 4. Unzip the dataset chunks
    extracted_dir = os.path.join(download_dir, "extracted")
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)
    os.makedirs(extracted_dir, exist_ok=True)
    
    print(f"Unzipping dataset chunks into {extracted_dir}...")
    try:
        subprocess.run(["unzip", "-q", zip_path, "-d", extracted_dir], check=True)
    except Exception as e:
        print(f"Error: Failed to unzip cache chunks: {e}")
        sys.exit(1)

    # 5. Concatenate and stream extract chunks directly to hf_home_dir
    chunks = sorted([f for f in os.listdir(extracted_dir) if f.startswith("hf_cache.tar.a")])
    if not chunks:
        print(f"Error: No hf_cache.tar.a* chunks found in extracted zip.")
        sys.exit(1)
        
    chunk_paths = [os.path.join(extracted_dir, c) for c in chunks]
    print(f"Extracting {len(chunks)} chunks directly via pipe into {hf_home_dir}...")
    os.makedirs(hf_home_dir, exist_ok=True)
    
    tar_cmd = ["tar", "-xf", "-", "-C", hf_home_dir]
    tar_proc = subprocess.Popen(tar_cmd, stdin=subprocess.PIPE)
    
    try:
        for chunk_path in chunk_paths:
            print(f"  Streaming {os.path.basename(chunk_path)}...")
            with open(chunk_path, "rb") as infile:
                shutil.copyfileobj(infile, tar_proc.stdin)
        tar_proc.stdin.close()
        ret = tar_proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, tar_cmd)
    except Exception as e:
        print(f"Extraction failed: {e}")
        if tar_proc.poll() is None:
            tar_proc.terminate()
            tar_proc.wait()
        raise e

    # Write sentinel file
    with open(sentinel_path, "w") as f:
        f.write("extraction completed successfully")
    print("HuggingFace cache extraction completed successfully!")

    # Cleanup temporary zip downloads to free space
    print("Cleaning up temporary zip files...")
    try:
        shutil.rmtree(download_dir)
    except Exception:
        pass

def main():
    # 1. Update bootstrapper code from origin
    self_update()
    
    # 2. Setup credentials
    setup_kaggle_credentials()
    
    # Configure token
    hf_token = extract_hf_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("Configured HF_TOKEN from execution parameters.")
    else:
        print("Warning: HF_TOKEN not set. Online dataset loading may fail for gated datasets.")
    
    # Set default HF home to /tmp/hf_home for Lightning AI
    hf_home = "/tmp/hf_home"
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    
    # 3. Pull and extract cache
    download_and_extract_cache(hf_home)
    
    # 4. Make scripts executable
    for script in ["install_dependencies.sh", "run_training.sh", "bootstrap_and_train.py", "run_kaggle.py", "run_lightning.py"]:
        script_path = os.path.join("scripts", script)
        if os.path.exists(script_path):
            os.chmod(script_path, 0o755)
        root_script_path = os.path.join(".", script)
        if os.path.exists(root_script_path):
            os.chmod(root_script_path, 0o755)

    # 5. Run the dependency installation
    print("=== Step 4: Installing Dependencies ===")
    try:
        subprocess.run(["bash", "scripts/install_dependencies.sh"], check=True)
    except Exception as e:
        print(f"Warning: Dependency installation reported errors: {e}")

    # 6. Execute training via bootstrap_and_train.py
    cmd_args = sys.argv[1:]
    run_cmd = [sys.executable, "scripts/bootstrap_and_train.py"] + cmd_args
    print(f"=== Step 5: Launching Training via bootstrap_and_train.py ===")
    print(f"Running: {' '.join(run_cmd)}")
    
    os.execv(sys.executable, run_cmd)

if __name__ == "__main__":
    main()
