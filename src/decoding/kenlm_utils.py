import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def _try_install_kenlm_system_deps():
    """Best-effort apt install of Boost + Eigen on Debian/Ubuntu (e.g. Kaggle)."""
    if os.name == "nt":
        return
    need_boost = not os.path.exists("/usr/include/boost/program_options.hpp")
    need_eigen = not (
        os.path.exists("/usr/include/eigen3")
        or os.path.exists("/usr/local/include/eigen3")
    )
    if not need_boost and not need_eigen:
        return
    pkgs = []
    if need_boost:
        pkgs.extend(["libboost-all-dev", "libboost-program-options-dev"])
    if need_eigen:
        pkgs.append("libeigen3-dev")
    if not pkgs:
        return
    # Prefer sudo when available (Kaggle notebooks often allow passwordless sudo).
    for prefix in (["sudo", "-n"], []):
        cmd = prefix + ["apt-get", "update", "-qq"]
        try:
            subprocess.run(cmd, check=False, timeout=120, capture_output=True)
            install = prefix + ["apt-get", "install", "-y", "-qq"] + pkgs
            r = subprocess.run(install, check=False, timeout=300, capture_output=True)
            if r.returncode == 0:
                logger.info(f"Installed system deps for KenLM: {pkgs}")
                return
        except Exception as exc:
            logger.debug(f"apt via {prefix or 'user'}: {exc}")
    logger.warning(
        "Could not apt-install KenLM deps (boost/eigen). "
        "LM build will be skipped unless they are already present."
    )


def compile_kenlm(kenlm_dir="kenlm", strict=False):
    """
    Downloads and compiles the KenLM C++ codebase.

    Parameters
    ----------
    kenlm_dir : str
        Checkout / build directory.
    strict : bool
        If True, re-raise on failure. If False (default), log and return (None, None)
        so training can continue with greedy CTC decode.

    Returns
    -------
    (lmplz_path, build_binary_path) or (None, None) on soft failure.
    """
    logger.info("Checking for KenLM binaries...")
    lmplz_path = os.path.join(kenlm_dir, "build", "bin", "lmplz")
    build_binary_path = os.path.join(kenlm_dir, "build", "bin", "build_binary")

    if os.path.exists(lmplz_path) and os.path.exists(build_binary_path):
        logger.info("KenLM binaries already compiled and available.")
        return lmplz_path, build_binary_path

    logger.info("KenLM binaries not found. Cloning and compiling KenLM...")

    try:
        _try_install_kenlm_system_deps()

        if not os.path.exists(kenlm_dir):
            subprocess.run(
                ["git", "clone", "https://github.com/kpu/kenlm.git", kenlm_dir],
                check=True,
            )

        build_dir = os.path.join(kenlm_dir, "build")
        os.makedirs(build_dir, exist_ok=True)

        logger.info("Running cmake for KenLM compilation...")
        subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
        import multiprocessing

        cores = multiprocessing.cpu_count()
        subprocess.run(["make", f"-j{cores}"], cwd=build_dir, check=True)
    except Exception as e:
        msg = (
            f"KenLM compile skipped ({e}). "
            "Training continues with greedy CTC decode (no lm.bin). "
            "Optional fix: sudo apt-get install -y libboost-all-dev libeigen3-dev"
        )
        if strict:
            logger.error(msg)
            raise
        logger.warning(msg)
        return None, None

    if not (os.path.exists(lmplz_path) and os.path.exists(build_binary_path)):
        msg = "KenLM build finished but lmplz/build_binary missing."
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return None, None

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
    Returns None if KenLM is unavailable.
    """
    lmplz_path, build_binary_path = compile_kenlm(kenlm_dir, strict=False)
    if not lmplz_path or not build_binary_path:
        logger.warning("KenLM unavailable — skipping train_kenlm_model.")
        return None

    logger.info(f"Training 5-gram language model using {lmplz_path}...")
    with open(text_path, "r", encoding="utf-8") as stdin_fh, open(
        arpa_path, "w", encoding="utf-8"
    ) as arpa_file:
        subprocess.run(
            [lmplz_path, "-o", "5", "--discount_fallback"],
            stdin=stdin_fh,
            stdout=arpa_file,
            check=True,
        )

    logger.info(f"Compressing language model to trie binary format using {build_binary_path}...")
    subprocess.run(
        [build_binary_path, "trie", arpa_path, binary_path],
        check=True,
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

    lmplz_path, build_binary_path = compile_kenlm(kenlm_dir, strict=False)
    if not lmplz_path or not build_binary_path:
        logger.warning("KenLM binaries not available. Skipping LM build.")
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
