import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def compile_kenlm(kenlm_dir="kenlm"):
    """
    Downloads and compiles the KenLM C++ codebase in the environment.
    """
    logger.info("Checking for KenLM binaries...")
    lmplz_path = os.path.join(kenlm_dir, "build", "bin", "lmplz")
    build_binary_path = os.path.join(kenlm_dir, "build", "bin", "build_binary")
    
    if os.path.exists(lmplz_path) and os.path.exists(build_binary_path):
        logger.info("KenLM binaries already compiled and available.")
        return lmplz_path, build_binary_path
        
    logger.info("KenLM binaries not found. Cloning and compiling KenLM...")
    
    # 1. Clone repo if it doesn't exist
    if not os.path.exists(kenlm_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/kpu/kenlm.git", kenlm_dir],
            check=True
        )
        
    # 2. Compile KenLM using cmake
    build_dir = os.path.join(kenlm_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    logger.info("Running cmake for KenLM compilation...")
    # On Windows/Linux containers we run cmake and make
    try:
        subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
        # Check system cores to speed up compilation
        import multiprocessing
        cores = multiprocessing.cpu_count()
        subprocess.run(["make", f"-j{cores}"], cwd=build_dir, check=True)
    except Exception as e:
        logger.error(f"Failed to compile KenLM: {e}. Please ensure build-essential and cmake are installed.")
        raise e
        
    logger.info("KenLM compiled successfully.")
    return lmplz_path, build_binary_path

def build_interpolated_text_corpus(conversational_path, formal_path, output_path, lambda_val=0.7):
    """
    Constructs a weighted, interpolated corpus by combining conversational and formal text.
    We replicate conversational lines to match the target lambda weight:
    ratio = lambda_val / (1.0 - lambda_val)
    """
    logger.info(f"Building interpolated corpus. Lambda={lambda_val} (conversational weight)")
    
    with open(conversational_path, "r", encoding="utf-8") as f:
        conversational_lines = [line.strip() for line in f.readlines() if line.strip()]
        
    with open(formal_path, "r", encoding="utf-8") as f:
        formal_lines = [line.strip() for line in f.readlines() if line.strip()]
        
    # Calculate duplication factor
    # Weight of conversational = lambda_val
    # Weight of formal = 1.0 - lambda_val
    # Let N_conv * c_factor / (N_conv * c_factor + N_formal) = lambda_val
    # N_conv * c_factor = lambda_val * N_conv * c_factor + lambda_val * N_formal
    # (1 - lambda_val) * N_conv * c_factor = lambda_val * N_formal
    # c_factor = (lambda_val / (1 - lambda_val)) * (N_formal / N_conv)
    
    n_conv = len(conversational_lines)
    n_formal = len(formal_lines)
    
    if n_conv == 0:
        combined_lines = formal_lines
    elif n_formal == 0:
        combined_lines = conversational_lines
    else:
        ratio = lambda_val / (1.0 - lambda_val)
        c_factor = max(1, round(ratio * (n_formal / n_conv)))
        logger.info(f"Replicating conversational lines {c_factor} times to achieve weight.")
        combined_lines = conversational_lines * c_factor + formal_lines
        
    # Shuffle to mix text patterns naturally
    import random
    random.shuffle(combined_lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for line in combined_lines:
            f.write(line + "\n")
            
    logger.info(f"Interpolated text corpus saved to {output_path} (Total lines: {len(combined_lines)})")

def train_kenlm_model(text_path, arpa_path, binary_path, kenlm_dir="kenlm"):
    """
    Runs KenLM binaries to compile the text corpus into a compressed trie binary model.
    """
    lmplz_path, build_binary_path = compile_kenlm(kenlm_dir)
    
    logger.info(f"Training 5-gram language model using {lmplz_path}...")
    # Run lmplz command
    with open(arpa_path, "w", encoding="utf-8") as arpa_file:
        subprocess.run(
            [lmplz_path, "-o", "5"],
            stdin=open(text_path, "r", encoding="utf-8"),
            stdout=arpa_file,
            check=True
        )
        
    logger.info(f"Compressing language model to trie binary format using {build_binary_path}...")
    # Run build_binary command
    subprocess.run(
        [build_binary_path, "trie", arpa_path, binary_path],
        check=True
    )
    
    logger.info(f"Successfully compiled KenLM model: {binary_path}")
    return binary_path


def build_language_model(transcripts, output_dir, kenlm_dir="kenlm", order=5):
    """
    High-level helper: builds a KenLM n-gram LM binary from a list of transcript
    strings extracted from the training set.

    Workflow:
        transcripts → lm_corpus.txt → lmplz → lm.arpa → build_binary → lm.bin

    Skips re-compilation if lm.bin already exists (safe across training restarts).

    Parameters
    ----------
    transcripts : list[str]  — raw/normalised transcript strings
    output_dir  : str        — directory where lm_corpus.txt, lm.arpa, lm.bin are written
    kenlm_dir   : str        — path to the compiled KenLM source tree
    order       : int        — n-gram order (default 5)

    Returns
    -------
    Path to the compiled lm.bin, or None if compilation fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    text_path   = os.path.join(output_dir, "lm_corpus.txt")
    arpa_path   = os.path.join(output_dir, "lm.arpa")
    binary_path = os.path.join(output_dir, "lm.bin")

    if os.path.exists(binary_path):
        logger.info(f"LM binary already exists at {binary_path}. Skipping rebuild.")
        return binary_path

    try:
        lmplz_path, build_binary_path = compile_kenlm(kenlm_dir)
    except Exception as exc:
        logger.warning(f"KenLM binaries not available ({exc}). Skipping LM build.")
        return None

    # 1. Write transcript corpus to plain-text file
    clean_lines = [t.strip() for t in transcripts if t and t.strip()]
    logger.info(f"Writing {len(clean_lines)} transcript lines to {text_path}")
    with open(text_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(clean_lines) + "\n")

    # 2. Run lmplz (--discount_fallback prevents crashes on small corpora)
    logger.info(f"Running lmplz (order={order}) to produce ARPA file...")
    try:
        with open(text_path, "rb") as stdin_fh, open(arpa_path, "wb") as arpa_fh:
            subprocess.run(
                [lmplz_path, "-o", str(order), "--discount_fallback"],
                stdin=stdin_fh,
                stdout=arpa_fh,
                check=True,
            )
        logger.info(f"ARPA model written to {arpa_path}")
    except subprocess.CalledProcessError as exc:
        logger.error(f"lmplz failed: {exc}. LM build aborted.")
        return None

    # 3. Compress ARPA → trie binary
    logger.info("Compressing ARPA to trie binary with build_binary...")
    try:
        subprocess.run(
            [build_binary_path, "trie", arpa_path, binary_path],
            check=True,
        )
        logger.info(f"KenLM binary ready at {binary_path}")
    except subprocess.CalledProcessError as exc:
        logger.error(f"build_binary failed: {exc}. LM build aborted.")
        return None

    return binary_path
