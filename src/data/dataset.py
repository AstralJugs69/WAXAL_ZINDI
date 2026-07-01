import os
import re
import csv
import logging
import unicodedata
import pandas as pd
from sklearn.model_selection import GroupKFold
from datasets import load_dataset, Dataset, Audio

logger = logging.getLogger(__name__)

def normalize_text(text):
    """
    Standardizes punctuation, converts to lowercase, applies NFKC unicode normalization,
    and strips special symbols keeping only letters and spaces.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. NFKC Unicode Normalization
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Lowercase conversion
    text = text.lower()
    
    # 3. Clean spacing
    text = re.sub(r"\s+", " ", text)
    
    # 4. Strip punctuation and special symbols (keeping unicode letters and whitespace)
    # [^\w\s] removes punctuation but keeps letters in low-resource orthographies
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    
    # 5. Remove digits if transcriptions shouldn't have them, or keep them if they are read speech numbers.
    # For ASR challenge, standard is keeping letters and spacing
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

def get_speaker_metadata(languages=["lin", "sna", "lug"]):
    """
    Loads google/WaxalNLP metadata from Hugging Face for specified configs
    to map ID to speaker_id.
    Does NOT load the audio column to prevent downloading/reading any audio bytes into RAM.
    """
    logger.info(f"Fetching speaker metadata from Hugging Face google/WaxalNLP for {languages}")
    id_to_meta = {}

    for lang in languages:
        config_name = f"{lang}_asr"
        lang_count = 0
        for split_name in ["train", "validation"]:
            try:
                # Load one split at a time
                split_ds = load_dataset("google/WaxalNLP", config_name, split=split_name)
                # Drop the audio column immediately to prevent loading raw bytes into RAM
                split_ds = split_ds.remove_columns(["audio"])
                for example in split_ds:
                    ex_id = example.get("id") or example.get("client_id")
                    if ex_id:
                        id_to_meta[ex_id] = {
                            "speaker_id": example.get("speaker_id") or example.get("client_id") or "unknown_speaker"
                        }
                        lang_count += 1
            except Exception as e:
                logger.warning(f"Could not load split '{split_name}' for {config_name}: {e}")

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
