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
    working_dir = "/kaggle/working"
    project_dir = os.path.join(working_dir, "WAXAL_ZINDI")

    print("=== Step 1: Retrieving Codebase ===")
    if os.path.exists(project_dir):
        print("Found existing project directory. Removing to get a fresh clone...")
        shutil.rmtree(project_dir)

    # Clone repository
    run_command_live(["git", "clone", repo_url, project_dir])
    os.chdir(project_dir)
    print(f"Current working directory set to: {os.getcwd()}")

    print("\n=== Step 2: Making Shell Scripts Executable ===")
    for script in ["install_dependencies.sh", "run_training.sh"]:
        script_path = os.path.join("scripts", script)
        if os.path.exists(script_path):
            os.chmod(script_path, 0o755)

    print("\n=== Step 3: Installing Dependencies & Compiling KenLM ===")
    run_command_live(["bash", "scripts/install_dependencies.sh"])

    print("\n=== Step 4: Kickstarting Model Training Pipeline ===")
    # Starts fine-tuning the MMS-300M model on Lingala for fold 0
    # Arguments: [config_path] [fold_index] [target_language]
    run_command_live(["bash", "scripts/run_training.sh", "config/base_mms.yaml", "0", "lin"])

    print("\n=== Bootstrapping and Training Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
