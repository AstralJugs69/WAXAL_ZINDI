"""
External corpora loader for augmenting training data with open-source
speech datasets (Mozilla Common Voice, Google FLEURS) per target language.

Each source is normalized into the unified WAXAL schema:
    audio (Audio @ 16kHz), normalized_transcription (str), id (str)
"""
import os
import logging
from datasets import Audio, concatenate_datasets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language → dataset configuration table
# ---------------------------------------------------------------------------
# Each source entry specifies:
#   dataset_id   – HuggingFace dataset repo
#   config_name  – dataset config / language code passed to load_dataset()
#   text_column  – column that holds the raw transcript text
#   splits       – which HF splits to pull from (all go into train pool)
# ---------------------------------------------------------------------------
EXTERNAL_CORPUS_CONFIGS = {
    "lin": {  # Lingala
        "common_voice": {
            "dataset_id": "mozilla-foundation/common_voice_17_0",
            "config_name": "ln",
            "text_column": "sentence",
            "splits": ["train", "validation", "test"],
        },
        "fleurs": {
            "dataset_id": "google/fleurs",
            "config_name": "ln_cd",
            "text_column": "transcription",
            "splits": ["train", "validation"],
        },
    },
    "lug": {  # Luganda
        "common_voice": {
            "dataset_id": "mozilla-foundation/common_voice_17_0",
            "config_name": "lg",
            "text_column": "sentence",
            "splits": ["train", "validation", "test"],
        },
        "fleurs": {
            "dataset_id": "google/fleurs",
            "config_name": "lg_ug",
            "text_column": "transcription",
            "splits": ["train", "validation"],
        },
    },
    "sna": {  # Shona — no Common Voice; FLEURS only
        "fleurs": {
            "dataset_id": "google/fleurs",
            "config_name": "sn_zw",
            "text_column": "transcription",
            "splits": ["train", "validation"],
        },
    },
}


def load_external_corpus(lang: str, sources: list = None) -> object:
    """
    Loads and normalises external open-source speech corpora for *lang*.

    Parameters
    ----------
    lang    : ISO 639-3 language code used by the WAXAL challenge ("lin", "lug", "sna")
    sources : list of source keys to load, e.g. ["common_voice", "fleurs"].
              If None, all configured sources for the language are loaded.

    Returns
    -------
    A HuggingFace Dataset with columns: audio, normalized_transcription, id
    or None if no data could be loaded.
    """
    from datasets import load_dataset
    from src.data.dataset import normalize_text

    if lang not in EXTERNAL_CORPUS_CONFIGS:
        logger.warning(f"No external corpus config for language '{lang}'. Skipping.")
        return None

    lang_configs = EXTERNAL_CORPUS_CONFIGS[lang]
    if sources is None:
        sources = list(lang_configs.keys())

    all_datasets = []
    global_idx = 0  # Used to produce stable, unique ID strings

    for source in sources:
        if source not in lang_configs:
            logger.warning(f"Source '{source}' not configured for '{lang}'. Skipping.")
            continue

        cfg = lang_configs[source]
        dataset_id = cfg["dataset_id"]
        config_name = cfg["config_name"]
        text_column = cfg["text_column"]

        try:
            logger.info(f"Loading external corpus: {dataset_id} [{config_name}] for '{lang}'...")
            token = os.environ.get("HF_TOKEN")
            # Prefer split-by-split loads so a missing split does not fail the whole source.
            split_parts = []
            load_attempts = [
                # Modern parquet datasets (no custom script)
                dict(path=dataset_id, name=config_name, trust_remote_code=True, token=token),
                dict(path=dataset_id, name=config_name, token=token),
            ]
            for split in cfg.get("splits", ["train"]):
                loaded = False
                last_err = None
                for base in load_attempts:
                    try:
                        part = load_dataset(**base, split=split)
                        split_parts.append(part)
                        logger.info(f"  Loaded split '{split}': {len(part)} examples")
                        loaded = True
                        break
                    except Exception as split_exc:
                        last_err = split_exc
                if not loaded:
                    logger.warning(f"  Split '{split}' failed for {dataset_id}: {last_err}")

            if not split_parts:
                # Fallback: load whole dataset dict
                try:
                    ds = load_dataset(
                        dataset_id,
                        config_name,
                        trust_remote_code=True,
                        token=token,
                    )
                    for split in cfg.get("splits", ["train"]):
                        if hasattr(ds, "keys") and split in ds:
                            split_parts.append(ds[split])
                        elif split == "train" and not hasattr(ds, "keys"):
                            split_parts.append(ds)
                except Exception as whole_exc:
                    logger.warning(f"  Full load failed for {dataset_id}: {whole_exc}")

            if not split_parts:
                logger.warning(f"  No valid splits found in {dataset_id}. Skipping.")
                continue

            from datasets import concatenate_datasets as _cat
            combined = _cat(split_parts) if len(split_parts) > 1 else split_parts[0]
            logger.info(f"  Raw examples: {len(combined)}")

            # ----------------------------------------------------------------
            # Normalise to unified schema in a single batched map pass
            # ----------------------------------------------------------------
            start_idx = global_idx

            def _normalise(batch, idx_offset=start_idx):
                texts = batch[text_column]
                batch["normalized_transcription"] = [
                    normalize_text(t) if t else "" for t in texts
                ]
                batch["id"] = [
                    f"ext_{source}_{lang}_{idx_offset + i}"
                    for i in range(len(texts))
                ]
                return batch

            # NOTE: do NOT pass decode_audio= to map() — removed/unsupported in many
            # datasets versions and was killing FLEURS loads with:
            #   Dataset.map() got an unexpected keyword argument 'decode_audio'
            map_kwargs = dict(
                batched=True,
                batch_size=1000,
                desc=f"Normalising {source}/{lang}",
            )
            # Prefer not decoding audio during the text-only normalisation pass.
            try:
                combined = combined.map(_normalise, **map_kwargs, load_from_cache_file=True)
            except TypeError:
                combined = combined.map(_normalise, **map_kwargs)
            global_idx += len(combined)

            # Drop every column except the three we need
            keep = {"audio", "normalized_transcription", "id"}
            drop = [c for c in combined.column_names if c not in keep]
            if drop:
                combined = combined.remove_columns(drop)

            # Resample audio to 16 kHz to match WAXAL (lazy; does not decode all bytes yet)
            if "audio" in combined.column_names:
                combined = combined.cast_column("audio", Audio(sampling_rate=16000))

            # Drop rows with empty transcription without decoding audio
            combined = combined.filter(
                lambda text: bool(text and str(text).strip()),
                input_columns=["normalized_transcription"],
                desc=f"Removing empty transcriptions in {source}",
            )

            logger.info(f"  Final examples after normalisation: {len(combined)}")
            all_datasets.append(combined)

        except Exception as exc:
            logger.warning(
                f"Failed to load '{source}' for '{lang}' ({dataset_id}): {exc}. "
                "Skipping this source — training will continue without it."
            )

    if not all_datasets:
        logger.warning(f"No external corpora loaded for language '{lang}'.")
        return None

    result = concatenate_datasets(all_datasets)
    logger.info(
        f"External corpus total for '{lang}': {len(result)} examples "
        f"from sources: {sources}"
    )
    return result
