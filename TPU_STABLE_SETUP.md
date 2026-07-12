# Kaggle TPU Stable Setup (WAXAL MMS)

## Why the old path kept failing

| Failure mode | Where | Fix in new path |
|---|---|---|
| Manual `xmp.spawn` + HF `Trainer` double orchestration | `src/training/trainer.py --tpu` | Lightning owns TPU process group |
| Variable audio lengths → endless XLA recompiles | collators / `group_by_length` | Fixed pad/truncate (`max_audio_seconds: 16`) |
| `num_workers>0` + pin_memory on TPU | TrainingArguments | `num_workers=0`, `pin_memory=False` |
| Mid-train `git pull` kill/restart every 5s | `run_all_languages.sh` + bootstrap | Removed from TPU path |
| Reinstalling `torch`/`torch_xla` on Kaggle | `install_dependencies.sh` / bootstrap | `run_tpu_train.py` never reinstalls torch |
| Default Gemma config on TPU | `kaggle_bootstrapper.py` | MMS + `config/base_mms_tpu.yaml` |
| Wrong JIT warmup shapes | trainer TPU branch | Removed entirely |
| Aggressive TPU env scrubbing | trainer imports | Minimal `PJRT_DEVICE=TPU` only |

## Recommended Kaggle hardware

- **Use TPU v5e-8** (Kaggle phased out v3-8).
- Attach dataset: `cashgenenator/waxal-hf-cache-chunks` (optional, faster WAXAL/MMS download).
- Secrets: `HF_TOKEN` (required for **Common Voice**; FLEURS works without).
- Internet ON.

## HF cache dataset extraction (critical)

The private Kaggle dataset `cashgenenator/waxal-hf-cache-chunks` (~54GB) is **not** a normal HF folder.
It is a **split tar** of a pre-populated `HF_HOME` built by `scripts/download_hf_assets.py`.

### Build pipeline (offline, already done)

1. Download models + datasets into a local `HF_HOME` (`hub/`, `datasets/`).
2. `tar` the whole tree.
3. `split` into ~5GB pieces: `hf_cache.tar.aa` … `hf_cache.tar.am` (13 chunks).
4. Upload pieces as a Kaggle dataset (avoids single-file upload limits / FUSE timeouts).

### Runtime pipeline (every Kaggle session)

```
/kaggle/input/<slug>/hf_cache.tar.aa … .am   (read-only mount)
                    │
                    │  find_cache_chunks_dir() walks /kaggle/input for hf_cache.tar.aa
                    ▼
        extract_cache_chunks(chunks_dir, hf_home)
                    │
                    │  for each chunk in sorted order:
                    │      stream bytes into tar stdin
                    │  tar -xf - -C $HF_HOME
                    │  write extraction_completed.txt
                    ▼
/kaggle/temp/hf_home/{hub,datasets}/         (writable, large)
```

Why this design:

| Constraint | Choice |
|---|---|
| `/kaggle/working` ≈ 20GB | Extract to **`/kaggle/temp/hf_home`**, not working |
| `/kaggle/input` read-only | Only source of chunks; never write there |
| Cannot materialize 54GB concat | **Pipe** chunks into `tar` (no intermediate file) |
| Session restarts | Sentinel `extraction_completed.txt` + non-empty `hub/` + `datasets/` skip re-extract |
| HF libraries | Must set `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE` **before** training |

Code owners:

- Discover + extract: `kaggle_bootstrapper.find_cache_chunks_dir` / `extract_cache_chunks` / `check_extraction_valid`
- TPU entry calls the same helpers: `run_tpu_train.extract_cache_if_present`
- Offline packer of contents: `scripts/download_hf_assets.py`

## One-cell launch

Paste / run `kaggle_tpu_cell.py`, or:

```bash
cd /kaggle/working/WAXAL_ZINDI
python run_tpu_train.py --lang all --tpu --devices 8 --fold 0
```

Per-language override (burn more quota):

```bash
python run_tpu_train.py --lang lin --tpu --max_steps 8000
python run_tpu_train.py --lang sna --tpu --max_steps 8000
python run_tpu_train.py --lang lug --tpu --max_steps 5000
```

Default budgets in `run_tpu_train.py`: lin/sna 5000 steps, lug 3500 steps  
(global batch ≈ 4 × 8 cores = 32).

## External data

Prefetch (automatic in `run_tpu_train.py`):

```bash
python scripts/prefetch_external_data.py --langs lin,sna,lug
```

| Lang | Sources |
|------|---------|
| lin | Common Voice `ln` + FLEURS `ln_cd` |
| lug | Common Voice `lg` + FLEURS `lg_ug` |
| sna | FLEURS `sn_zw` only |

## After training

Checkpoints:

```
outputs/{lin,sna,lug}_mms-300m_fold0/best_model/
```

Submission (MMS-first, blank retry, hard fail on high blank rate):

```bash
python generate_submission.py
# optional: --max-blank-frac 0.05
```

## Do not use for TPU

- `python src/training/trainer.py --tpu` (legacy over-engineered path)
- `run_lightning.py` git hot-reload loop during long TPU jobs
- Reinstalling PyTorch wheels mid-session on Kaggle TPU
