# Upload checkpoints to Kaggle (not Google Drive)

**Use this.** Kaggle already works with your `kaggle.json`. No OAuth, no service-account quota issues.

---

## What you need

| Item | Where |
|------|--------|
| `kaggle.json` | `/teamspace/studios/this_studio/kaggle.json` **or** `~/.kaggle/kaggle.json` |
| Checkpoints | `.../WAXAL_ZINDI/outputs/{lin,sna,lug}_mms-300m_fold0/` |
| Optional packs | `.../gdrive_upload_work/*.tar.gz` (reused if present) |

---

## One command (Lightning terminal)

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
git pull origin main

# ensure kaggle.json is available
ls -la /teamspace/studios/this_studio/kaggle.json
# if missing, upload it there via Lightning file UI

export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs

# Pack + create/update Kaggle dataset: YOUR_USER/waxal-mms-checkpoints
python scripts/upload_checkpoints_kaggle.py --langs lin,sna,lug

# If you already packed .tar.gz under gdrive_upload_work (skip re-pack):
python scripts/upload_checkpoints_kaggle.py --langs lin,sna,lug --skip-pack
```

Or:

```bash
python lightning_studio_bootstrap.py upload --langs lin,sna,lug
python lightning_studio_bootstrap.py upload --langs lin,sna,lug --skip-pack
```

---

## What success looks like

```text
Creating NEW private dataset cashgenenator/waxal-mms-checkpoints …
# or
Dataset exists → uploading NEW VERSION of cashgenenator/waxal-mms-checkpoints …

UPLOAD FINISHED
  Dataset: https://www.kaggle.com/datasets/YOUR_USER/waxal-mms-checkpoints
```

Open that URL while logged into Kaggle. You should see the `.tar.gz` files.

---

## Download on a **new** Lightning / Kaggle machine

**Important quirks (do not trust these blindly):**

| Command / UI | Reality |
|--------------|---------|
| `kaggle datasets list -m` **size column** | Often shows **0** even when dataset is ~59GB (web UI is correct) |
| Web **Data Explorer** | Source of truth — folders `lin/sna/lug_mms-300m_fold0` |
| Layout | May be **folders**, not `.tar.gz` |

### Recommended (handles quirks)

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
git pull origin main

mkdir -p $HOME/.kaggle
cp -f /teamspace/studios/this_studio/kaggle.json $HOME/.kaggle/kaggle.json
chmod 600 $HOME/.kaggle/kaggle.json

export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs

# lists files (may be incomplete) + downloads + moves into outputs/
python scripts/download_checkpoints_kaggle.py \
  --dataset cashgenenator/waxal-mms-checkpoints
```

### Manual CLI fallback

```bash
mkdir -p outputs ckpt_dl
python -m kaggle datasets files cashgenenator/waxal-mms-checkpoints
python -m kaggle datasets download -d cashgenenator/waxal-mms-checkpoints -p ./ckpt_dl --force
ls -lh ckpt_dl/
# unzip if needed
cd ckpt_dl && unzip -q *.zip; cd ..
# move folders
for d in lin_mms-300m_fold0 sna_mms-300m_fold0 lug_mms-300m_fold0; do
  find ckpt_dl -type d -name "$d" | head -1 | while read p; do
    rm -rf outputs/$d
    mv "$p" outputs/
  done
done
ls outputs/*/
```

---

## If upload fails

| Message | Fix |
|---------|-----|
| `kaggle.json not found` | Upload `kaggle.json` to studio root or `~/.kaggle/` |
| `401` / unauthorized | Regenerate API token on kaggle.com → Account → API |
| `404` on version | First run should **create**; ensure `datasets create` ran, not only version |
| Disk full while packing | Free space; packs are ~20GB each |
| Only some langs | Use `--langs sna,lug` for what exists under `outputs/` |

---

## Notes

- Dataset is **private** by default (only you).
- Re-running the script creates a **new dataset version** (keeps history).
- Prefer Kaggle over Google Drive for this project; Drive SA cannot use personal quota.
