#!/usr/bin/env python3
"""
Watcher and process manager for live-reloading ASR training on Lightning AI.
Monitors the local filesystem for edits and polls the git remote for new commits.
Automatically cleans up running child processes and VRAM on updates, runs git pull,
aligns dependencies if requirements change, and restarts the training pipeline.
"""
import os
import sys
import time
import signal
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [Watcher] - %(levelname)s - %(message)s")
logger = logging.getLogger("watcher")

class TrainingWatcher:
    def __init__(self, config, fold, target_lang, tpu=False, poll_interval=10):
        self.config = config
        self.fold = fold
        self.target_lang = target_lang
        self.tpu = tpu
        self.poll_interval = poll_interval
        
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.process = None
        self.last_mtime = self.get_max_mtime()
        
        # Git tracking branch info
        self.current_branch = self.get_git_branch()
        logger.info(f"Initialized watcher on branch: '{self.current_branch}'")
        logger.info(f"Targeting: config={config}, fold={fold}, lang={target_lang}, tpu={tpu}")
        
    def get_git_branch(self):
        try:
            res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=self.project_root)
            return res.stdout.strip()
        except Exception:
            return "main"

    def get_max_mtime(self):
        """
        Recursively finds the maximum modification time among source files (.py, .yaml, .sh).
        """
        max_time = 0.0
        # Watch src/ and config/ directories
        watch_dirs = [
            os.path.join(self.project_root, "src"),
            os.path.join(self.project_root, "config"),
            os.path.join(self.project_root, "scripts")
        ]
        
        for watch_dir in watch_dirs:
            if not os.path.exists(watch_dir):
                continue
            for root, _, files in os.walk(watch_dir):
                # Skip cache directories
                if "__pycache__" in root or ".git" in root or "outputs" in root:
                    continue
                for f in files:
                    if f.endswith((".py", ".yaml", ".yml", ".sh", "requirements.txt")):
                        filepath = os.path.join(root, f)
                        try:
                            mtime = os.path.getmtime(filepath)
                            if mtime > max_time:
                                max_time = mtime
                        except OSError:
                            pass
        return max_time

    def check_git_updates(self):
        """
        Polls the remote git repository by running git fetch and comparing HEAD vs tracking branch.
        """
        try:
            # git fetch to update tracking branches
            subprocess.run(["git", "fetch", "origin", self.current_branch], capture_output=True, cwd=self.project_root)
            
            # Compare local HEAD with origin/branch
            local_hash_res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=self.project_root)
            remote_hash_res = subprocess.run(["git", "rev-parse", f"origin/{self.current_branch}"], capture_output=True, text=True, cwd=self.project_root)
            
            local_hash = local_hash_res.stdout.strip()
            remote_hash = remote_hash_res.stdout.strip()
            
            if local_hash and remote_hash and local_hash != remote_hash:
                logger.info(f"Git updates detected: local {local_hash[:7]} != remote {remote_hash[:7]}")
                return True
        except Exception as e:
            logger.warning(f"Error checking git updates: {e}")
        return False

    def pull_and_install_deps(self):
        """
        Pulls latest commits and installs dependencies if requirements.txt changed.
        """
        logger.info("Executing git pull...")
        
        # Check requirements mtime before pull
        req_file = os.path.join(self.project_root, "requirements.txt")
        req_mtime_before = os.path.getmtime(req_file) if os.path.exists(req_file) else 0
        
        pull_res = subprocess.run(["git", "pull", "origin", self.current_branch], capture_output=True, text=True, cwd=self.project_root)
        logger.info(pull_res.stdout)
        
        # Check requirements mtime after pull
        req_mtime_after = os.path.getmtime(req_file) if os.path.exists(req_file) else 0
        if req_mtime_after > req_mtime_before:
            logger.info("requirements.txt changed. Updating python dependencies...")
            subprocess.run(["bash", "scripts/install_dependencies.sh"], cwd=self.project_root)

    def terminate_process(self):
        """
        Gracefully terminates the training subprocess and all its children.
        """
        if not self.process:
            return
            
        pid = self.process.pid
        logger.info(f"Terminating active training process group (PID: {pid})...")
        
        try:
            if os.name != "nt":
                # Send SIGTERM to the process group (negative PID)
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
                
                # Wait for process to clean up
                for _ in range(20):
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.5)
                
                # If still running, send SIGKILL
                if self.process.poll() is None:
                    logger.warning("Process group did not exit gracefully. Sending SIGKILL...")
                    os.killpg(pgid, signal.SIGKILL)
            else:
                self.process.terminate()
                self.process.wait(timeout=10)
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
            
        self.process = None
        logger.info("Training process cleaned up successfully.")

    def start_training(self):
        """
        Launches the training pipeline in a separate process group.
        """
        logger.info("=== Spawning Training Pipeline ===")
        
        # Setup environment variables for offline mode and target cache
        env = os.environ.copy()
        env["HF_HOME"] = "/teamspace/studios/this_studio/hf_home"
        env["HF_HUB_CACHE"] = "/teamspace/studios/this_studio/hf_home/hub"
        env["HF_DATASETS_CACHE"] = "/teamspace/studios/this_studio/hf_home/datasets"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONPATH"] = self.project_root
        
        # Build training command
        if self.tpu:
            cmd = [
                "python", "src/training/trainer.py",
                "--config", self.config,
                "--fold", str(self.fold),
                "--target_lang", self.target_lang,
                "--tpu"
            ]
        else:
            # GPU/CPU DDP via run_training.sh
            cmd = [
                "bash", "scripts/run_training.sh",
                self.config,
                str(self.fold),
                self.target_lang
            ]
            
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Start the process in a new session (process group) to easily kill child processes
        kwargs = {}
        if os.name != "nt":
            kwargs["preexec_fn"] = os.setsid
            
        self.process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            **kwargs
        )
        logger.info(f"Training started (PID: {self.process.pid})")

    def run(self):
        # Start initial run
        self.start_training()
        
        try:
            while True:
                time.sleep(self.poll_interval)
                
                # Check for process exit (completion or crash)
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.poll()
                    logger.info(f"Training process exited with code: {exit_code}")
                    self.process = None
                
                # Check for changes
                mtime_changed = False
                git_changed = False
                
                # 1. Local filesystem modifications
                current_max_mtime = self.get_max_mtime()
                if current_max_mtime > self.last_mtime:
                    logger.info("Local file modification detected.")
                    self.last_mtime = current_max_mtime
                    mtime_changed = True
                    
                # 2. Remote git updates
                if self.check_git_updates():
                    git_changed = True
                    
                if mtime_changed or git_changed:
                    logger.info("Triggering reload...")
                    self.terminate_process()
                    
                    if git_changed:
                        self.pull_and_install_deps()
                        # Reset mtime baseline to avoid double triggers from pulled code
                        self.last_mtime = self.get_max_mtime()
                        
                    self.start_training()
                    
        except KeyboardInterrupt:
            logger.info("Watcher interrupted by user. Cleaning up...")
            self.terminate_process()
            sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live-Reloading Training Runner")
    parser.add_argument("--config", type=str, default="config/base_mms.yaml", help="Path to config file")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0 to 4)")
    parser.add_argument("--target_lang", type=str, default="lin", help="Target language (lin, sna, lug)")
    parser.add_argument("--tpu", action="store_true", help="Launch on Google TPU")
    parser.add_argument("--poll_interval", type=int, default=10, help="Polling interval in seconds")
    args = parser.parse_args()
    
    watcher = TrainingWatcher(
        config=args.config,
        fold=args.fold,
        target_lang=args.target_lang,
        tpu=args.tpu,
        poll_interval=args.poll_interval
    )
    watcher.run()
