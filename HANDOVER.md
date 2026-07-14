# HANDOVER — WAXAL ASR Challenge (Session 2026-07-12 → 2026-07-14+)

**Read this first** before changing training, inference, or infrastructure.  
This document is the operating bible for the next human developer or AI agent.

| | |
|--|--|
| **Repo** | https://github.com/AstralJugs69/WAXAL_ZINDI.git (`main`) |
| **Local path (Windows)** | `C:\dev\WAXAL\waxal_asr_challenge` |
| **Bootstrap notes (parent)** | `C:\dev\WAXAL\bootstrap\` (research plan + starter Gemma notebook) |
| **Competition** | Zindi Google WAXAL ASR — Phase 1 public LB ~0.5 historically; target Phase 2 generalization |
| **Languages** | Lingala (`lin`), Shona (`sna`), Luganda (`lug`) |
| **Primary model** | **Per-language fine-tuned MMS-300M CTC** (not Gemma) |
| **Eval metric** | ~0.5 × WER + 0.5 × CER (Zindi-style blend) |

---

## 1. Strategic decisions (do not reverse without strong reason)

1. **Primary ASR = MMS-300M CTC per language**  
   - Better character stability on Bantu/agglutinative languages than autoregressive models.  
   - **Gemma is deprecated as primary** (hallucinations, “No audio” refusals, ~0.5 LB scores).

2. **Whisper-Small** = secondary / future ensemble only.

3. **Speaker-independent validation**  
   - Default HF train/val splits **leak speakers** (hundreds of overlapping IDs).  
   - Use **GroupKFold-style** folds by speaker (or low-RAM hash groups). Never trust default HF split scores for Phase 2.

4. **External data allowed and used on Lightning**  
   - Common Voice + FLEURS (see `src/data/external_corpora.py`).  
   - Shona: mostly FLEURS (little/no CV).

5. **Phase 2** will have **no language labels** → need VAD + LID + per-lang routing later.  
   Phase 1 test IDs still have prefixes (`lin_`, `sna_`, `lug_`).

6. **Budget is extremely tight**  
   - Prefer free Kaggle/Colab; paid only for short GPU bursts.  
   - **Never** run open-ended epoch counts (no “train to 999”).

---

## 2. What this session accomplished

### 2.1 Strategy & audit
- Confirmed Gemma path is a dead end for primary scoring.
- Audited `generate_submission.py` missing-ID issues, TPU over-engineering, remote artifacts.
- Aligned plan with `bootstrap/WAXAL Speech AI Research Plan.md`.

### 2.2 Inference hardening (`generate_submission.py`)
- **MMS-first** model selection (Gemma only if forced / no MMS).
- Canonical IDs: **never** use `speaker_id` as cache key (caused silent blanks).
- Blank-target **retry** pass (replaces need for `transcribe_missing.py`).
- Fail if blank rate too high (`--max-blank-frac`).
- HF cache env is portable (no hard-coded Lightning-only paths as sole option).

### 2.3 Training stack pivot
| Path | Status | Use when |
|------|--------|----------|
| `src/training/trainer.py` + `xmp.spawn` | Legacy / fragile | Avoid for new work |
| `src/training/lightning_mms.py` | **Primary** | All new training |
| Gemma / TRL SFT | Deprioritized | Do not restart as main line |

### 2.4 Lightning AI workflow (credit-saving)
1. **CPU machine**: data prep only (`scripts/setup_data_cpu.py` / `lightning_studio_bootstrap.py prep`).  
2. **GPU machine (RTXP 6000 ~96–102GB VRAM, ~125GB RAM)**: train only.  
3. **Checkpoints on Kaggle dataset** so studio wipe ≠ total loss (Google Drive SA upload abandoned).

### 2.5 Languages trained (approximate outcomes)

| Lang | Outcome | Notes |
|------|---------|--------|
| **lin** | Strong | Long early run (~$5); train_loss got very low (~0.3 regime). Do **not** retrain unless needed. |
| **sna** | Recovered + continued | Stuck ~2.8 with batch 48; fixed with **batch 16, lr 3e-4, unfreeze FE** → train~0.59 / val~1.02; resumed past PT2.6 ckpt load bugs. |
| **lug** | Trained (pack uploaded) | Same recipe; packed as `lug_mms-300m_fold0.tar.gz` (~20GB) into Kaggle dataset. |

Exact epoch counts may vary; **trust Kaggle dataset tarballs + `outputs/`**, not chat memory.

### 2.6 Checkpoint backup decision (important)

| Attempt | Result |
|---------|--------|
| Google Drive + **service account** | **FAILED** — `Service Accounts do not have storage quota` on personal My Drive even when folder is shared |
| Google Drive + OAuth | Documented but user abandoned (confusing setup) |
| **Kaggle dataset** | **SUCCESS** — uses existing `kaggle.json` |

**Live checkpoint dataset (as of 2026-07-14):**  
https://www.kaggle.com/datasets/cashgenenator/waxal-mms-checkpoints  

Contains resumable packs: `lin_mms-300m_fold0.tar.gz`, `sna_…`, `lug_…` (each ~20GB).

### 2.7 Infrastructure locations

| Asset | Location |
|-------|----------|
| GitHub | `AstralJugs69/WAXAL_ZINDI` `main` |
| Kaggle HF cache dataset | `cashgenenator/waxal-hf-cache-chunks` (~54GB split tar) |
| Kaggle user | `cashgenenator` |
| **Checkpoints dataset** | **`cashgenenator/waxal-mms-checkpoints`** |
| Upload script | `scripts/upload_checkpoints_kaggle.py` |
| Upload docs | `docs/KAGGLE_CHECKPOINT_UPLOAD.md` |
| Drive (legacy/optional) | `1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru` — not required |
| HF cache on Lightning | `/teamspace/studios/this_studio/hf_home` |
| Outputs on Lightning | `/teamspace/studios/this_studio/WAXAL_ZINDI/outputs` |

---

## 3. Time-saving & credit-saving techniques (use these)

### 3.1 Never burn GPU on data prep
- Extract Kaggle cache + download external + fold CSVs on **CPU**.
- GPU session only: train / submit / upload.

### 3.2 Split-tar HF cache (huge time saver)
```
/kaggle/input/.../hf_cache.tar.aa … am  →  stream pipe →  $HF_HOME/{hub,datasets}
```
- Implemented in `kaggle_bootstrapper.extract_cache_chunks` + `scripts/setup_data_cpu.py`.
- **Never** extract into `/kaggle/working` (20GB) or small disks; use `/kaggle/temp` or Lightning studio disk.
- Sentinel: `$HF_HOME/extraction_completed.txt`.

### 3.3 Hard training caps
- Always set **`--epochs N`** or **`--max_steps M`**.  
- UI once showed `Epoch x/999` (bad display hack) — **fixed**. Still: never assume open-ended training.
- Use **EarlyStopping** unless you explicitly need a hard epoch target (`--no_early_stopping`).

### 3.4 Batch size vs learning (critical lesson)
| Mistake | Effect |
|---------|--------|
| batch 48, lr 1e-4, ~6 epochs, frozen FE | sna **stuck** ~2.8 loss (~950 steps total) |
| batch 16, lr 3e-4, unfreeze FE, ~5–7 epochs | sna **learned** ~0.6 train / ~1.0 val |

**Rule:** On this task, prefer **enough optimizer steps** over maxing VRAM.  
RTXP 6000 can take batch 32–48, but **scale LR** and ensure **≥~400–500 steps/epoch** when data allows.

### 3.5 CLI overrides (don’t re-edit YAML every time)
```bash
python -m src.training.lightning_mms \
  --config config/base_mms_lightning_96gb.yaml \
  --target_lang sna --fold 0 --devices 1 \
  --epochs 7 --batch_size 16 --lr 3e-4 \
  --unfreeze_feature_encoder --resume --no_early_stopping
```

### 3.6 Resume after crash
- Checkpoints: `outputs/{lang}_mms-300m_fold0/checkpoints/*.ckpt`
- PyTorch 2.6+ requires `weights_only=False` for Lightning ckpts — **already patched** in:
  - post-train export load
  - `trainer.fit(..., weights_only=False)` / torch.load fallback
- `--resume` loads `last.ckpt` or newest `.ckpt`.

### 3.7 Studio wipe recovery (current procedure)
1. Clone repo + place `kaggle.json`.  
2. Download **`cashgenenator/waxal-mms-checkpoints`** → extract tarballs into `outputs/`.  
3. CPU prep if `hf_home` missing (`lightning_studio_bootstrap.py prep` or attach/extract HF cache).  
4. Continue train / submit / re-upload new versions to the same Kaggle dataset.

### 3.8 Kaggle constraints (if using free GPUs)
| Pool | Limit | Implication |
|------|-------|-------------|
| System RAM | ~30 GB | `devices=1`, `num_workers=0`, `low_ram: true`, no external prefetch |
| VRAM | 2×T4 ~15–16 GB each | batch 1–2, ≤12–16s audio |
| TPU | v5e-8, ~20h/week | Prefer Lightning MMS path; old `xmp.spawn` trainer is fragile |

### 3.9 Do not reinstall torch on Kaggle TPU/GPU images
- Breaks prebuilt CUDA/XLA wheels. Install only lightning/transformers/datasets extras.

### 3.10 KenLM is optional
- Missing Boost on Kaggle → soft-fail; greedy CTC still works.  
- Do not block training on KenLM compile.

---

## 4. Fresh Lightning VM bootstrap (after crash) — COPY/PASTE

**Prereqs on the new studio:** upload `kaggle.json` to  
`/teamspace/studios/this_studio/kaggle.json`

### 4.1 One block: clone + restore checkpoints + env (start here)

**Note:** `kaggle datasets list -m` often shows **size=0** even when the dataset is ~59GB
(web Data Explorer is correct: folders `lin/sna/lug_mms-300m_fold0`). Use the download helper.

```bash
# === NEW LIGHTNING VM — restore from Kaggle checkpoints ===
export STUDIO=/teamspace/studios/this_studio
export HF_HOME=$STUDIO/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export WAXAL_OUTPUTS_DIR=$STUDIO/WAXAL_ZINDI/outputs
export TOKENIZERS_PARALLELISM=false

mkdir -p $HOME/.kaggle
cp -f $STUDIO/kaggle.json $HOME/.kaggle/kaggle.json
chmod 600 $HOME/.kaggle/kaggle.json

cd $STUDIO
if [ -d WAXAL_ZINDI/.git ]; then
  cd WAXAL_ZINDI && git pull origin main
else
  git clone --depth 1 https://github.com/AstralJugs69/WAXAL_ZINDI.git
  cd WAXAL_ZINDI
fi
GIT_PAGER=cat git log -1 --oneline

pip install -q kaggle
# Reliable restore (handles list size=0 quirk + folder layout, not tar.gz)
python scripts/download_checkpoints_kaggle.py \
  --dataset cashgenenator/waxal-mms-checkpoints \
  --outputs "$WAXAL_OUTPUTS_DIR" \
  --download-dir ./ckpt_dl

# or: python lightning_studio_bootstrap.py download

ls -la outputs/
ls outputs/*/checkpoints 2>/dev/null
ls outputs/*/best_model 2>/dev/null
echo "Checkpoints restored."
```

### 4.2 CPU prep (only if HF cache missing — free/cheap machine)

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
# needs kaggle.json for waxal-hf-cache-chunks
python lightning_studio_bootstrap.py prep
# or: python scripts/setup_data_cpu.py
test -f /teamspace/studios/this_studio/hf_home/extraction_completed.txt && echo HF_CACHE_OK
```

### 4.3 GPU: install train deps + submit (if all three best_model exist)

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
export HF_HOME=/teamspace/studios/this_studio/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs

pip install -q lightning "transformers>=4.40.0" "datasets>=2.19.0,<4.0.0" \
  accelerate librosa soundfile jiwer pyyaml scikit-learn evaluate tqdm

python generate_submission.py --max-blank-frac 0.05
ls -la submission.csv
```

### 4.4 GPU: resume a language if needed

```bash
python -m src.training.lightning_mms \
  --config config/base_mms_lightning_96gb.yaml \
  --target_lang lug --fold 0 --devices 1 \
  --epochs 10 --batch_size 16 --lr 3e-4 \
  --unfreeze_feature_encoder --resume --no_early_stopping
```

### 4.5 Re-upload after any new training

```bash
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs
python scripts/upload_checkpoints_kaggle.py --langs lin,sna,lug
# creates a NEW VERSION of cashgenenator/waxal-mms-checkpoints
```

---

## 5. Key files map

| Path | Role |
|------|------|
| `lightning_studio_bootstrap.py` | **Session entry**: prep / train / upload / submit |
| `scripts/setup_data_cpu.py` | CPU: Kaggle cache + models + external + fold CSVs |
| `scripts/upload_checkpoints_kaggle.py` | **Preferred** checkpoint backup → Kaggle dataset |
| `docs/KAGGLE_CHECKPOINT_UPLOAD.md` | How to upload/download ckpts via Kaggle |
| `scripts/upload_checkpoints_gdrive.py` | Legacy Drive upload (OAuth only on personal Drive) |
| `src/training/lightning_mms.py` | **Main trainer** (Lightning + MMS CTC) |
| `config/base_mms_lightning_96gb.yaml` | High-VRAM Lightning config (external ON) |
| `config/base_mms_gpu.yaml` | Kaggle 2×T4 / 30GB RAM (low_ram, external OFF) |
| `config/base_mms_tpu.yaml` | Kaggle TPU-oriented config |
| `generate_submission.py` | MMS-first inference + blank retry |
| `src/data/external_corpora.py` | CV + FLEURS loaders |
| `src/data/dataset.py` | Waxal load, GroupKFold, text norm |
| `src/inference/pipeline.py` | Phase-2 VAD+LID scaffold (not fully wired to submit) |
| `kaggle_tpu_cell.py` / `kaggle_gpu_cell.py` | One-cell Kaggle launchers |
| `src/training/trainer.py` | Legacy HF Trainer + TPU spawn — **avoid** |
| `bootstrap/` (parent dir) | Research plan + official Gemma starter notebook |

---

## 6. Known pitfalls (agent checklist)

- [ ] Do **not** start a new Gemma primary training campaign.  
- [ ] Do **not** use default HF splits as Phase-2 proxy.  
- [ ] Do **not** set `max_epochs=1000` or train without a stop condition.  
- [ ] Do **not** assume `Epoch x/999` means 999 is required (historical UI bug).  
- [ ] Do **not** use `speaker_id` as Zindi audio ID.  
- [ ] PyTorch **2.6+**: Lightning ckpt load needs `weights_only=False` (patched).  
- [ ] `Dataset.map(decode_audio=...)` is **invalid** on some datasets versions (fixed in external loader).  
- [ ] Duration decode filter at ~38 ex/s looks “hung” and can drop 80% of data — **off by default**.  
- [ ] Dual-GPU DDP on 30GB host RAM **doubles** dataset footprint — use `devices=1` on Kaggle.  
- [ ] Common Voice needs `HF_TOKEN` + license accept; FLEURS usually does not.  
- [ ] `kaggle.json` is enough for cache **and** checkpoint dataset upload/download; HF token optional for CV.  
- [ ] **Never** use Google SA for personal Drive uploads (`storageQuotaExceeded`).  
- [ ] Competition: avoid models pretrained on full WAXAL including Phase-1 test leakage.

---

## 7. Immediate next steps (priority order)

1. **Fresh Lightning VM** (studio wiped; **$15 topped up**)  
   - Run **§4.1** below: clone + download `cashgenenator/waxal-mms-checkpoints` + extract to `outputs/`.  
   - CPU prep only if HF cache missing (§4.2).

2. **Verify all three languages restored**  
   - `outputs/{lin,sna,lug}_mms-300m_fold0/checkpoints` and ideally `best_model/`.

3. **Generate Phase-1 submission** (main goal after restore)  
   ```bash
   python generate_submission.py --max-blank-frac 0.05
   ```
   Prefer GPU for speed.

4. **Only if a language is weak / incomplete**  
   - Resume with proven recipe: batch 16, lr 3e-4, unfreeze, capped epochs.  
   - Re-upload: `python scripts/upload_checkpoints_kaggle.py --langs lin,sna,lug`.

5. **Phase-2 readiness** (after Phase-1 MMS baseline submitted)  
   - Wire `ProductionASRPipeline` (VAD + MMS-LID + per-lang MMS).  
   - KenLM + pyctcdecode α/β on OOS folds.  
   - Optional Whisper ensemble.

6. **Leaderboard**  
   - Historical public score ~0.5 with Gemma-era system.  
   - Expect large jump once all three MMS `best_model` folders feed `generate_submission.py`.

---

## 8. Config cheat-sheet

| Config | When |
|--------|------|
| `config/base_mms_lightning_96gb.yaml` | Lightning high-VRAM GPU; external ON; default batch high (override with CLI) |
| `config/base_mms_gpu.yaml` | Kaggle 30GB RAM / 2×T4; low_ram; external OFF |
| `config/base_mms_tpu.yaml` | Kaggle TPU experiments |
| CLI overrides | Always preferred over editing YAML mid-session |

**Proven learning recipe (sna recovery):**  
`--batch_size 16 --lr 3e-4 --unfreeze_feature_encoder --epochs 5–10`

---

## 9. Session meta

- **Dates:** 2026-07-12 → 2026-07-14 (+ checkpoint backup day)  
- **User constraints:** Tight Lightning credits; studio can wipe entirely; **checkpoints must live off-VM**.  
- **Checkpoint home:** Kaggle dataset **`cashgenenator/waxal-mms-checkpoints`** (upload finished successfully).  
- **Agent instruction:** Prefer small reversible changes; always check real files under `waxal_asr_challenge` before advising; keep Phase-2 generalization first.  
- **Do not** regenerate long one-off command walls — extend `lightning_studio_bootstrap.py` and this HANDOVER.

### 9.1 Changelog since first HANDOVER.md commit
- Abandoned Google Drive SA upload (quota error); switched to **Kaggle datasets**.  
- Added `scripts/upload_checkpoints_kaggle.py` + `docs/KAGGLE_CHECKPOINT_UPLOAD.md`.  
- Confirmed dataset URL: https://www.kaggle.com/datasets/cashgenenator/waxal-mms-checkpoints  
- Packs: lin/sna/lug `*_mms-300m_fold0.tar.gz` (~20GB each) uploaded.  
- Fresh VM procedure rewritten around **Kaggle download → tar -xzf → outputs/**.

---

## 10. Quick verification on a new machine

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI && git pull && git log -1 --oneline
ls outputs/*/checkpoints 2>/dev/null
ls outputs/*/best_model 2>/dev/null
test -f /teamspace/studios/this_studio/hf_home/extraction_completed.txt && echo HF_CACHE_OK
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `outputs/` empty: re-run **§4.1** (download `cashgenenator/waxal-mms-checkpoints`).

---

*End of handover. Update this file at the end of every major session.*

