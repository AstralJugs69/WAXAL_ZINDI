import os
import tarfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bundler")

def filter_tar(tarinfo):
    # Exclude temporary cache folders and intermediate checkpoints
    exclude_patterns = [
        "__pycache__",
        ".git",
        ".ipynb_checkpoints",
        "outputs",
        "temp_repo",
        "kenlm/build",
        "target_lm.bin"
    ]
    for pattern in exclude_patterns:
        if pattern in tarinfo.name:
            return None
    return tarinfo

def main():
    archive_name = "src.tar.gz"
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Target /kaggle/working/src.tar.gz if running on Kaggle, otherwise write to repo root
    if os.path.exists("/kaggle/working"):
        output_path = "/kaggle/working/src.tar.gz"
    else:
        output_path = os.path.join(repo_dir, archive_name)
    
    logger.info(f"Bundling workspace from {repo_dir} into {output_path}...")
    
    # We include src/, config/, scripts/, and requirements.txt
    include_paths = ["src", "config", "scripts", "requirements.txt"]
    
    with tarfile.open(output_path, "w:gz") as tar:
        for path in include_paths:
            full_path = os.path.join(repo_dir, path)
            if os.path.exists(full_path):
                logger.info(f"Adding: {path}")
                tar.add(full_path, arcname=path, filter=filter_tar)
            else:
                logger.warning(f"Path does not exist, skipping: {path}")
                
    logger.info(f"Successfully created bundle: {output_path} ({os.path.getsize(output_path) / 1024:.2f} KB)")

if __name__ == "__main__":
    main()
