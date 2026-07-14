# HANDOVER — WAXAL ASR Challenge (Session 2026-07-12 → 2026-07-14)

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
3. **Checkpoints uploaded to Google Drive** so studio wipe ≠ total loss.

### 2.5 Languages trained (approximate outcomes)

| Lang | Outcome | Notes |
|------|---------|--------|
| **lin** | Strong | Long early run (~$5); train_loss got very low (~0.3 regime). Do **not** retrain unless needed. |
| **sna** | Recovered | First schedule stuck (~2.8 loss, batch 48 too few updates). Restart with **batch 16, lr 3e-4, unfreeze FE** → train~0.59 / val~1.02. Resume/export fixed for PT 2.6. |
| **lug** | In progress / resume to 10 | Same recipe as successful sna: batch 16, lr 3e-4, unfreeze. |

Exact epoch counts on disk may vary; **trust folders under `outputs/` and Drive tarballs**, not chat memory.

### 2.6 Infrastructure locations

| Asset | Location |
|-------|----------|
| GitHub | `AstralJugs69/WAXAL_ZINDI` `main` |
| Kaggle HF cache dataset | `cashgenenator/waxal-hf-cache-chunks` (~54GB split tar `hf_cache.tar.aa`…`am`) |
| Kaggle user | `cashgenenator` |
| **Checkpoints on Kaggle (preferred)** | Dataset slug: `{kaggle_user}/waxal-mms-checkpoints` — see `docs/KAGGLE_CHECKPOINT_UPLOAD.md` |
| Script | `scripts/upload_checkpoints_kaggle.py` / `lightning_studio_bootstrap.py upload` |
| Google Drive | Deprecated for checkpoints (SA has no My Drive quota); optional folder `1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru` |
| Old HF-cache Drive | `1uDx64kRRT23e7ZSfkLS9f014g2oJS3WB` (cache chunks only, not model ckpts) |
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

### 3.7 Studio wipe recovery
1. Clone repo.  
2. CPU prep **or** restore `hf_home` if still on disk.  
3. Download `*_mms-300m_fold0.tar.gz` from Drive → extract into `outputs/`.  
4. Continue train / `generate_submission.py` / re-upload.

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

## 4. Fresh Lightning VM bootstrap (after crash)

### Entry point (preferred)
```bash
cd /teamspace/studios/this_studio
git clone --depth 1 https://github.com/AstralJugs69/WAXAL_ZINDI.git   # or pull if exists
cd WAXAL_ZINDI
git pull origin main

# Put kaggle.json at /teamspace/studios/this_studio/kaggle.json for cache download
python lightning_studio_bootstrap.py prep          # CPU machine
# later on GPU:
python lightning_studio_bootstrap.py train --lang lug --epochs 10 --batch-size 16 --lr 3e-4 --resume
python lightning_studio_bootstrap.py upload --langs lin,sna,lug --keep-archives
python lightning_studio_bootstrap.py submit
```

### Restore checkpoints from Drive first
```bash
# After downloading tar.gz from Drive into studio:
cd /teamspace/studios/this_studio/WAXAL_ZINDI
mkdir -p outputs
tar -xzf lin_mms-300m_fold0.tar.gz -C outputs/
tar -xzf sna_mms-300m_fold0.tar.gz -C outputs/
tar -xzf lug_mms-300m_fold0.tar.gz -C outputs/
ls outputs/*/checkpoints outputs/*/best_model
```

### Proven GPU train recipe (lin/sna/lug)
```bash
export HF_HOME=/teamspace/studios/this_studio/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs

python -m src.training.lightning_mms \
  --config config/base_mms_lightning_96gb.yaml \
  --target_lang LANGUAGE --fold 0 --devices 1 \
  --epochs N --batch_size 16 --lr 3e-4 \
  --unfreeze_feature_encoder --no_early_stopping
# add --resume when continuing
```

### Checkpoint upload → Kaggle dataset (preferred)
- Guide: `docs/KAGGLE_CHECKPOINT_UPLOAD.md`
- Script: `scripts/upload_checkpoints_kaggle.py`
- Needs: same `kaggle.json` as cache download
- Creates/updates: `{username}/waxal-mms-checkpoints` (private)

```bash
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs
# kaggle.json at /teamspace/studios/this_studio/kaggle.json
python scripts/upload_checkpoints_kaggle.py --langs lin,sna,lug
# reuse existing tar.gz packs:
python scripts/upload_checkpoints_kaggle.py --langs lin,sna,lug --skip-pack
```

---

## 5. Key files map

| Path | Role |
|------|------|
| `lightning_studio_bootstrap.py` | **Session entry**: prep / train / upload / submit |
| `scripts/setup_data_cpu.py` | CPU: Kaggle cache + models + external + fold CSVs |
| `scripts/upload_checkpoints_gdrive.py` | Resumable Drive upload of checkpoint tarballs |
| `docs/GOOGLE_DRIVE_UPLOAD.md` | Step-by-step Drive auth + upload |
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
- [ ] `kaggle.json` is enough for cache dataset download; HF token is optional for CV.  
- [ ] Competition: avoid models pretrained on full WAXAL including Phase-1 test leakage.

---

## 7. Immediate next steps (priority order)

1. **Fresh Lightning VM**  
   - `git pull` / clone.  
   - Restore checkpoint tarballs from Drive folder `1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru`.  
   - Re-run CPU prep if `hf_home` missing (`lightning_studio_bootstrap.py prep`).

2. **Finish lug** (if not already)  
   - Same recipe as sna success: batch 16, lr 3e-4, unfreeze, epochs ~8–10, `--resume`.

3. **Export `best_model/`** for each lang if only `.ckpt` exists (train script end-of-run, or load ckpt + `save_best_model`).

4. **Generate submission**  
   ```bash
   python generate_submission.py --max-blank-frac 0.05
   ```
   Prefer GPU host for speed; needs all three `outputs/{lang}_mms-300m_fold0/best_model/`.

5. **Re-upload** new/better checkpoints to Drive after any train.

6. **Phase-2 readiness** (after Phase-1 baseline solid)  
   - Wire `ProductionASRPipeline` (VAD + MMS-LID + per-lang MMS).  
   - KenLM + pyctcdecode α/β on OOS folds.  
   - Optional Whisper ensemble.

7. **Leaderboard**  
   - Historical public score ~0.5 with Gemma-era system.  
   - Re-submit after MMS lin/sna/lug are all exported and inference is clean.

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

- **Dates:** 2026-07-12 → 2026-07-14  
- **User constraints:** Very low remaining compute budget; studio can wipe; checkpoints must live on Drive.  
- **Agent instruction:** Prefer small reversible changes; always check real files under `waxal_asr_challenge` before advising; keep Phase-2 generalization first.  
- **Do not** regenerate long one-off command walls — extend `lightning_studio_bootstrap.py` and docs instead.

---

## 10. Quick verification on a new machine

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI && git pull && git log -1 --oneline
ls outputs/*/checkpoints 2>/dev/null
ls outputs/*/best_model 2>/dev/null
test -f /teamspace/studios/this_studio/hf_home/extraction_completed.txt && echo HF_CACHE_OK
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If checkpoints only on Drive: download tarballs → extract to `outputs/` → then submit or resume.

---

*End of handover. Update this file at the end of every major session.*
