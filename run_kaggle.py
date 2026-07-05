#!/usr/bin/env python3
"""
Kaggle Notebook Startup Shell Wrapper
-------------------------------------
Pulls any outstanding git updates and starts the unified bootstrap and training pipeline.
"""
import os
import sys
import subprocess

def main():
    # 1. Sync repository changes from origin branch
    if os.path.exists(".git"):
        print("Checking remote git repository for updates...")
        try:
            branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
            branch = branch_res.stdout.strip() or "main"
            subprocess.run(["git", "pull", "origin", branch], check=False)
        except Exception as e:
            print(f"Warning: Git pull failed: {e}")

    # 2. Launch the unified bootstrap and training script with CLI arguments
    cmd_args = sys.argv[1:]
    script_path = "scripts/bootstrap_and_train.py"
    
    print(f"Launching unified training script: {script_path} {' '.join(cmd_args)}")
    
    os.execv(sys.executable, [sys.executable, script_path] + cmd_args)

if __name__ == "__main__":
    main()
