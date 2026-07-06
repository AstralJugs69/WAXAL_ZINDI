#!/usr/bin/env python3
"""
Unified ASR Bootstrap & Training Runner
---------------------------------------
Combines:
1. Environment setup & dependency installation (GPU vs TPU checks).
2. Automated local Hugging Face cache chunks discovery, merge, extraction, and validation.
3. Subprocess execution of the training loop.
4. Git auto-update watcher that checks for remote commits, gracefully kills active runs,
   pulls changes, and hot-reloads the process via execv.
"""
import os
import sys
import time
import shutil
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Unified ASR Bootstrap & Training Runner")
    parser.add_argument("--config", type=str, default="config/base_mms.yaml", help="Path to config YAML file")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to train (0 to k_folds-1)")
    parser.add_argument("--target_lang", type=str, default="lin", help="Target language (lin, sna, lug)")
    parser.add_argument("--tpu", action="store_true", help="Launch training on Google TPU VM cores")
    parser.add_argument("--git_poll_interval", type=int, default=30, help="Git polling interval in seconds (0 to disable)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override training max steps")
    return parser.parse_args()

def check_and_update_git():
    """
    Checks if there are remote git changes at startup.
    If so, pulls changes and re-executes the script.
    """
    if not os.path.exists(".git"):
        return False
    try:
        print("Checking remote git repository for updates at startup...")
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        branch = branch_res.stdout.strip() or "main"
        
        subprocess.run(["git", "fetch", "origin", branch], capture_output=True)
        
        local_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        remote_hash = subprocess.run(["git", "rev-parse", f"origin/{branch}"], capture_output=True, text=True).stdout.strip()
        
        if local_hash and remote_hash and local_hash != remote_hash:
            print(f"Git updates detected: Local {local_hash[:7]} != Remote origin/{branch} {remote_hash[:7]}")
            print("Pulling changes...")
            subprocess.run(["git", "pull", "origin", branch], check=True)
            print("Restarting script with updated code...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
            return True
    except Exception as e:
        print(f"Warning: Failed to check/apply git updates at startup: {e}")
    return False

def check_and_update_git_during_training(proc):
    """
    Checks git during active training. If new commits are detected, 
    terminates the training process group, pulls, and restarts the script.
    """
    if not os.path.exists(".git"):
        return False
    try:
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        branch = branch_res.stdout.strip() or "main"
        
        subprocess.run(["git", "fetch", "origin", branch], capture_output=True)
        
        local_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        remote_hash = subprocess.run(["git", "rev-parse", f"origin/{branch}"], capture_output=True, text=True).stdout.strip()
        
        if local_hash and remote_hash and local_hash != remote_hash:
            print(f"Git updates detected while training! Local: {local_hash[:7]} | Remote: {remote_hash[:7]}")
            print("Terminating active training process group...")
            
            if os.name != "nt":
                import signal
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(20):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.5)
                if proc.poll() is None:
                    print("Process did not exit. Sending SIGKILL...")
                    os.killpg(pgid, signal.SIGKILL)
            else:
                proc.terminate()
                proc.wait()
                
            print("Pulling changes...")
            subprocess.run(["git", "pull", "origin", branch], check=True)
            
            print("Restarting script with updated code...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
            return True
    except Exception as e:
        print(f"Error checking git updates during training: {e}")
    return False

def bootstrap_environment(tpu_enabled):
    """
    Ensures environment dependencies, hardware packages, and KenLM are set up.
    """
    print("\n=== [Bootstrap] Aligning Dependencies ===")
    
    # 1. Check/Install TPU packages
    tpu_active = tpu_enabled or os.environ.get("TPU_NAME") or os.environ.get("TPU_ACCELERATOR_TYPE") or os.path.exists("/usr/share/tpu-support")
    if tpu_active:
        try:
            import torch_xla
            print("torch_xla is already installed.")
        except ImportError:
            print("Installing PyTorch/XLA 2.8.0 wheels for TPU VMs (Python 3.12)...")
            subprocess.run([
                "pip", "install", "--upgrade",
                "torch==2.8.0", "torch_xla[tpu]==2.8.0", 
                "-f", "https://storage.googleapis.com/libtpu-releases/index.html"
            ], check=True)
            
    # 2. Run install_dependencies.sh
    if os.path.exists("scripts/install_dependencies.sh"):
        os.chmod("scripts/install_dependencies.sh", 0o755)
        print("Running install_dependencies.sh...")
        subprocess.run(["bash", "scripts/install_dependencies.sh"], check=True)
    else:
        print("Warning: scripts/install_dependencies.sh not found. Skipping.")

def find_cache_chunks_dir():
    search_roots = ["/kaggle/input", "/content", "."]
    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            if ".git" in root or "hf_home" in root or "WAXAL_ZINDI" in root:
                continue
            if "hf_cache.tar.aa" in files:
                print(f"Discovered cache chunks folder at: {root}")
                return root
    return None

def check_extraction_valid(hf_home_dir):
    sentinel_path = os.path.join(hf_home_dir, "extraction_completed.txt")
    if not os.path.exists(sentinel_path):
        return False
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

def start_training_subprocess(args):
    env = os.environ.copy()
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        if os.path.exists("/kaggle/temp"):
            hf_home = "/kaggle/temp/hf_home"
        elif os.path.exists("/kaggle/working") or os.path.exists("/content"):
            hf_home = "/tmp/hf_home"
        else:
            hf_home = "./hf_home"
        
    env["HF_HOME"] = hf_home
    env["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    env["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["PYTHONPATH"] = "."
    
    tpu_active = args.tpu or os.environ.get("TPU_NAME") or os.environ.get("TPU_ACCELERATOR_TYPE") or os.path.exists("/usr/share/tpu-support")
    
    if tpu_active:
        cmd = [
            sys.executable, "src/training/trainer.py",
            "--config", args.config,
            "--fold", str(args.fold),
            "--target_lang", args.target_lang,
            "--tpu"
        ]
        if args.max_steps > 0:
            cmd.extend(["--max_steps", str(args.max_steps)])
    else:
        if shutil.which("bash"):
            cmd = [
                "bash", "scripts/run_training.sh",
                args.config,
                str(args.fold),
                args.target_lang
            ]
            if args.max_steps > 0:
                cmd.extend(["--max_steps", str(args.max_steps)])
        else:
            cmd = [
                sys.executable, "src/training/trainer.py",
                "--config", args.config,
                "--fold", str(args.fold),
                "--target_lang", args.target_lang
            ]
            if args.max_steps > 0:
                cmd.extend(["--max_steps", str(args.max_steps)])
            
    print(f"Launching training subprocess: {' '.join(cmd)}")
    
    kwargs = {}
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
        
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        **kwargs
    )

def main():
    args = parse_args()
    
    # 1. Check for remote git updates at startup
    check_and_update_git()
    
    # 2. Bootstrap environment and compile KenLM
    bootstrap_environment(args.tpu)
    
    # 3. Extract HuggingFace cache chunks if available (skip if running in Kaggle environment to avoid duplicate extraction)
    if os.path.exists("/kaggle"):
        print("Kaggle environment detected. Skipping duplicate cache extraction in bootstrap_and_train.py (handled by bootstrapper).")
    else:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            if os.path.exists("/kaggle/temp"):
                hf_home = "/kaggle/temp/hf_home"
            elif os.path.exists("/kaggle/working") or os.path.exists("/content"):
                hf_home = "/tmp/hf_home"
            else:
                hf_home = "./hf_home"
            
        chunks_dir = find_cache_chunks_dir()
        if chunks_dir:
            if check_extraction_valid(hf_home):
                print(f"Valid extraction sentinel found at {hf_home}. Skipping extraction.")
            else:
                try:
                    extract_cache_chunks(chunks_dir, hf_home)
                except Exception as e:
                    print(f"Warning: Cache extraction failed: {e}. Training will fallback to online HF hub.")
        else:
            print("Notice: hf_cache.tar.aa chunk was not found in standard directories. Skipping local cache extraction.")
        
    # 4. Kickstart training and watch for remote git changes
    proc = start_training_subprocess(args)
    
    if args.git_poll_interval > 0:
        print(f"Training started in background. Polling remote git every {args.git_poll_interval}s for updates...")
        try:
            while True:
                ret = proc.poll()
                if ret is not None:
                    print(f"Training completed with exit code: {ret}")
                    sys.exit(ret)
                
                time.sleep(args.git_poll_interval)
                
                # Check for git updates
                check_and_update_git_during_training(proc)
        except KeyboardInterrupt:
            print("Terminating training subprocess on keyboard interrupt...")
            if os.name != "nt":
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
            sys.exit(1)
    else:
        # One-shot run: wait for completion synchronously
        print("Training started. Polling disabled. Waiting for training to complete...")
        ret = proc.wait()
        print(f"Training completed with exit code: {ret}")
        sys.exit(ret)

if __name__ == "__main__":
    main()
