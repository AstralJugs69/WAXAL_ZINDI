import os
import re
import csv
import logging
import unicodedata
import pandas as pd
from sklearn.model_selection import GroupKFold
from datasets import load_dataset, Dataset, Audio

logger = logging.getLogger(__name__)

def get_audio_data(audio_info):
    """
    Robustly extracts the raw audio numpy array and sampling rate from an audio feature representation.
    Supports standard dict format, custom properties, and datasets >= 4.0.0 AudioDecoder object (based on torchcodec).
    """
    if audio_info is None:
        return None, None
        
    # Case 1: AudioDecoder object (datasets >= 4.0.0 / torchcodec integration)
    if hasattr(audio_info, "get_all_samples"):
        try:
            samples = audio_info.get_all_samples()
            array = samples.data
            # If the data is a PyTorch tensor, convert it to a NumPy array
            if hasattr(array, "numpy"):
                array = array.numpy()
            
            # Ensure it is a 1D mono array (channels, frames) -> (frames,)
            if array.ndim > 1:
                if array.shape[0] < array.shape[-1]:
                    array = array[0]  # Select first channel
                else:
                    array = array[:, 0]
            array = array.flatten()
            return array, samples.sample_rate
        except Exception as e:
            logger.warning(f"Error decoding via AudioDecoder.get_all_samples: {e}")
            
    # Case 2: Standard dictionary representation
    if isinstance(audio_info, dict):
        array = audio_info.get("array")
        sr = audio_info.get("sampling_rate")
        if array is not None:
            import numpy as np
            array = np.array(array)
            if array.ndim > 1:
                if array.shape[0] < array.shape[-1]:
                    array = array[0]
                else:
                    array = array[:, 0]
            array = array.flatten()
        return array, sr
        
    # Case 3: Duck typing for objects with attributes
    if hasattr(audio_info, "array") and hasattr(audio_info, "sampling_rate"):
        return audio_info.array, audio_info.sampling_rate
        
    # Case 4: Subscription fallback (in case it is dictionary-like)
    try:
        return audio_info["array"], audio_info["sampling_rate"]
    except Exception:
        pass
        
    return None, None

def normalize_text(text):
    """
    ASR text normalization aligned with common Zindi/WAXAL eval practice:
      - NFKC + lowercase
      - keep unicode letters, digits, whitespace
      - keep apostrophe (') — critical for Luganda (g'ennyanja, eby'enjawulo)
      - strip other punctuation
      - collapse whitespace

    Older versions stripped apostrophes entirely, which merged word pieces and
    inflated CER/WER vs references that retain them (or vs scorers that keep ').
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. NFKC Unicode Normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Normalize quote-like chars to ASCII apostrophe (Luganda clitics)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    text = text.replace("\u02bc", "'").replace("\u00b4", "'")

    # 4. Drop control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # 5. Keep letters / digits / whitespace / apostrophe only
    #    \w is unicode-aware with re.UNICODE (letters from African orthographies stay)
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    # drop bare underscores that \w keeps (not part of spoken orthography here)
    text = text.replace("_", " ")

    # 6. Collapse whitespace; strip dangling apostrophes used as quotes
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"'+", "'", text)
    text = re.sub(r"\s'\s", " ", text)
    text = text.strip(" '")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def resolve_csv_path(csv_path):
    """
    Resolves the actual location of the CSV files case-insensitively by checking
    the current directory, parent directory, /kaggle/working, or searching /kaggle/input.
    """
    if os.path.exists(csv_path):
        return csv_path
        
    basename_lower = os.path.basename(csv_path).lower()
    
    # 1. Try current directory case-insensitively
    try:
        for f in os.listdir("."):
            if f.lower() == basename_lower:
                return f
    except Exception:
        pass
        
    # 2. Try parent directory case-insensitively
    try:
        for f in os.listdir(".."):
            if f.lower() == basename_lower:
                return os.path.join("..", f)
    except Exception:
        pass
        
    # 3. Try /kaggle/working/ case-insensitively
    if os.path.exists("/kaggle/working"):
        try:
            for f in os.listdir("/kaggle/working"):
                if f.lower() == basename_lower:
                    return os.path.join("/kaggle/working", f)
        except Exception:
            pass
            
    # 4. Search inside /kaggle/input/ case-insensitively
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f.lower() == basename_lower:
                    resolved = os.path.join(root, f)
                    logger.info(f"Auto-discovered CSV file at: {resolved}")
                    return resolved
                    
    return csv_path

def parse_robust_csv(csv_path):
    """
    Parses a Zindi CSV file with robust handling of unescaped quotes,
    commas inside fields, and multiline cells.
    """
    csv_path = resolve_csv_path(csv_path)
    logger.info(f"Parsing CSV file robustly: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    rows = []
    current_row = ""
    parsed_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    header_line = lines[0].strip().strip('"')
    headers = header_line.split('","')
    is_test = len(headers) == 1 and headers[0].lower() in ["id", "test_id"]
    
    for line in lines[1:]:
        if not current_row:
            current_row = line
        else:
            current_row += line
            
        stripped = current_row.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            parts = stripped[1:-1].split('","')
            if is_test:
                rows.append({"id": parts[0]})
                current_row = ""
                parsed_count += 1
            else:
                if len(parts) >= 4:
                    # In case the transcription contains '","' itself:
                    id_val = parts[0]
                    original_split = parts[-1]
                    language = parts[-2]
                    transcription = '","'.join(parts[1:-2])
                    rows.append({
                        "id": id_val,
                        "transcription": transcription,
                        "language": language,
                        "original_split": original_split
                    })
                    current_row = ""
                    parsed_count += 1
        elif is_test and not current_row.strip().startswith('"'):
            # Sometimes test set doesn't use quotes
            rows.append({"id": current_row.strip()})
            current_row = ""
            parsed_count += 1
            
    df = pd.DataFrame(rows)
    logger.info(f"Successfully parsed {len(df)} rows from {csv_path}")
    return df

# Static mapping of language codes to parquet shard counts in google/WaxalNLP
WAXAL_PARQUET_COUNTS = {
    "lin": {"train": 8, "validation": 2},
    "lug": {"train": 5, "validation": 1},
    "sna": {"train": 9, "validation": 2}
}

def load_waxal_dataset_clean(lang):
    """
    Loads only train and validation parquet files directly from Hugging Face Hub
    for the specified language. This completely bypasses the 40+ unlabeled files,
    preventing RAM OOMs and massive disk/network waste.
    """
    from huggingface_hub import list_repo_files
    from datasets import load_dataset, Audio
    
    repo_id = "google/WaxalNLP"
    logger.info(f"Retrieving file list for {repo_id} to load {lang} dataset...")
    
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset")
        lang_dir = f"data/ASR/{lang}"
        train_patterns = [f for f in all_files if f.startswith(lang_dir) and f"{lang}-train-" in f and f.endswith(".parquet")]
        val_patterns = [f for f in all_files if f.startswith(lang_dir) and f"{lang}-validation-" in f and f.endswith(".parquet")]
    except Exception as e:
        logger.warning(f"Failed to list files from Hugging Face Hub online: {e}. Falling back to static file list for offline mode.")
        if lang in WAXAL_PARQUET_COUNTS:
            train_patterns = [f"data/ASR/{lang}/{lang}-train-{i:05d}.parquet" for i in range(WAXAL_PARQUET_COUNTS[lang]["train"])]
            val_patterns = [f"data/ASR/{lang}/{lang}-validation-{i:05d}.parquet" for i in range(WAXAL_PARQUET_COUNTS[lang]["validation"])]
        else:
            logger.error(f"No static parquet config for language {lang} and online listing failed.")
            raise e
    
    if not train_patterns or not val_patterns:
        raise ValueError(f"Could not find train/validation parquet files for language {lang} in {repo_id}")
        
    train_urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}" for f in train_patterns]
    val_urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}" for f in val_patterns]
    
    # Try to resolve remote URLs to local cached paths to support offline mode
    local_train_paths = []
    local_val_paths = []
    
    cache_dirs = []
    hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_datasets_cache:
        cache_dirs.append(hf_datasets_cache)
        cache_dirs.append(os.path.join(hf_datasets_cache, "downloads"))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_dirs.append(hf_home)
        cache_dirs.append(os.path.join(hf_home, "datasets", "downloads"))
    cache_dirs.append(os.path.expanduser("~/.cache/huggingface"))
    cache_dirs.append(os.path.expanduser("~/.cache/huggingface/datasets/downloads"))
    
    # Debug log directories contents count
    for c_dir in cache_dirs:
        if os.path.exists(c_dir):
            try:
                f_count = 0
                for r, d, files in os.walk(c_dir):
                    f_count += len(files)
                logger.info(f"Cache telemetry: Directory '{c_dir}' exists with {f_count} total files.")
            except Exception:
                pass
                
    def resolve_url_to_local(url):
        basename = os.path.basename(url)
        escaped_url = url.replace("/", "\\/")
        for cache_dir in cache_dirs:
            if not os.path.exists(cache_dir):
                continue
            for root, _, files in os.walk(cache_dir):
                for f in files:
                    if f.endswith(".json"):
                        json_path = os.path.join(root, f)
                        try:
                            with open(json_path, "r", encoding="utf-8") as jf:
                                content = jf.read()
                                # Match by full URL, escaped URL, or basename of the shard
                                if url in content or escaped_url in content or basename in content:
                                    data_file_path = json_path[:-5]
                                    if os.path.exists(data_file_path):
                                        return data_file_path
                        except Exception:
                            pass
        return None

    logger.info("Checking local datasets cache for offline execution...")
    for url in train_urls:
        local_path = resolve_url_to_local(url)
        if local_path:
            local_train_paths.append(local_path)
            
    for url in val_urls:
        local_path = resolve_url_to_local(url)
        if local_path:
            local_val_paths.append(local_path)
            
    token = os.environ.get("HF_TOKEN")
    if len(local_train_paths) == len(train_urls) and len(local_val_paths) == len(val_urls):
        logger.info("All files found in local cache. Loading completely offline using local paths.")
        ds = load_dataset("parquet", data_files={"train": local_train_paths, "validation": local_val_paths}, token=token)
    else:
        logger.warning("Some files not found in local cache. Falling back to remote HF Hub URLs (requires online connection).")
        logger.info(f"Loading {len(train_urls)} train files and {len(val_urls)} validation files directly...")
        ds = load_dataset("parquet", data_files={"train": train_urls, "validation": val_urls}, token=token)
    
    # Cast audio column to Audio feature
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def get_speaker_metadata(languages=["lin", "sna", "lug"]):
    """
    Loads google/WaxalNLP metadata (id → speaker_id) without decoding audio.

    Low-RAM path: select only id/speaker columns and convert to pandas once per
    split instead of Python-iterating every row (which OOMs on ~30GB Kaggle GPU).
    """
    logger.info(f"Fetching speaker metadata from Hugging Face google/WaxalNLP for {languages}")
    id_to_meta = {}

    for lang in languages:
        lang_count = 0
        try:
            ds = load_waxal_dataset_clean(lang)
            for split_name in ["train", "validation"]:
                if split_name not in ds:
                    continue
                split_ds = ds[split_name]
                id_col = next((c for c in ("id", "client_id") if c in split_ds.column_names), None)
                if id_col is None:
                    continue
                keep = [id_col]
                if "speaker_id" in split_ds.column_names:
                    keep.append("speaker_id")
                elif "client_id" in split_ds.column_names and "client_id" not in keep:
                    keep.append("client_id")
                # Drop everything else (especially audio) before materializing
                try:
                    meta = split_ds.select_columns(keep)
                except Exception:
                    drop = [c for c in split_ds.column_names if c not in keep]
                    meta = split_ds.remove_columns(drop) if drop else split_ds
                try:
                    pdf = meta.to_pandas()
                except Exception as e:
                    logger.warning(f"to_pandas meta failed for {lang}/{split_name}: {e}; falling back to row iter")
                    pdf = None
                    for example in meta:
                        ex_id = example.get(id_col)
                        if not ex_id:
                            continue
                        spk = example.get("speaker_id") or example.get("client_id") or "unknown_speaker"
                        id_to_meta[str(ex_id)] = {"speaker_id": str(spk)}
                        lang_count += 1
                if pdf is not None:
                    spk_col = "speaker_id" if "speaker_id" in pdf.columns else (
                        "client_id" if "client_id" in pdf.columns else None
                    )
                    for _, row in pdf.iterrows():
                        ex_id = row.get(id_col)
                        if ex_id is None or (isinstance(ex_id, float) and str(ex_id) == "nan"):
                            continue
                        if spk_col:
                            spk = row.get(spk_col) or "unknown_speaker"
                        else:
                            spk = "unknown_speaker"
                        id_to_meta[str(ex_id)] = {"speaker_id": str(spk)}
                        lang_count += 1
                    del pdf
                del meta
            del ds
            import gc
            gc.collect()
        except Exception as e:
            logger.warning(f"Could not load clean dataset for language {lang} metadata mapping: {e}")

        logger.info(f"Loaded {lang_count} metadata entries for language '{lang}'")

    return id_to_meta



def prepare_datasets(train_csv_path, test_csv_path, languages=["lin", "sna", "lug"], k_folds=5):
    """
    Loads train and test sets, normalizes text, fetches speaker metadata, 
    and applies GroupKFold partitioning on speaker_id.
    """
    # 1. Parse Train and Test CSVs
    train_df = parse_robust_csv(train_csv_path)
    test_df = parse_robust_csv(test_csv_path)
    
    # 2. Normalize Text
    train_df["normalized_transcription"] = train_df["transcription"].apply(normalize_text)
    # Remove nulls or empty transcriptions after normalization
    train_df = train_df[train_df["normalized_transcription"].str.strip() != ""]
    
    # 3. Retrieve Speaker IDs
    hf_meta = get_speaker_metadata(languages)
    
    # Assign speaker_id
    def map_meta(row, key):
        meta = hf_meta.get(row["id"])
        if meta:
            return meta.get(key)
        # Fallback speaker ID based on ID prefix if metadata fails
        if key == "speaker_id":
            return f"spk_{hash(row['id']) % 1000}"
        return None
        
    train_df["speaker_id"] = train_df.apply(lambda r: map_meta(r, "speaker_id"), axis=1)
    test_df["speaker_id"] = test_df.apply(lambda r: map_meta(r, "speaker_id"), axis=1)
    
    # 4. GroupKFold Cross-Validation Splitting
    # We group strictly by speaker_id to ensure 0% speaker intersection between folds
    gkf = GroupKFold(n_splits=k_folds)
    train_df["fold"] = -1
    
    # We split based on the speaker_id groups
    groups = train_df["speaker_id"].values
    for fold_idx, (train_indices, val_indices) in enumerate(gkf.split(train_df, train_df["normalized_transcription"], groups)):
        train_df.iloc[val_indices, train_df.columns.get_loc("fold")] = fold_idx
        
    logger.info(f"GroupKFold splits completed across {k_folds} folds using speaker_id.")
    return train_df, test_df
