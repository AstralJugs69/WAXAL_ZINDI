# Upload WAXAL checkpoints to Google Drive

**Target folder:**  
https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru  

**Folder ID:** `1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru`

**What gets uploaded:**  
For each language that has a folder like:

```text
outputs/lin_mms-300m_fold0/
outputs/sna_mms-300m_fold0/
outputs/lug_mms-300m_fold0/
```

the script packs it into one archive:

```text
lin_mms-300m_fold0.tar.gz
sna_mms-300m_fold0.tar.gz
lug_mms-300m_fold0.tar.gz
```

and uploads those with **resumable** multi-MB chunks (safe on flaky networks).

**Script:** `scripts/upload_checkpoints_gdrive.py`  
**Shortcut:** `python lightning_studio_bootstrap.py upload`

---

## Recommended path on Lightning AI: Service account

OAuth browser login is awkward on remote studios. A **service account** is the reliable way.

### Step 1 — Create a service account (once, on your laptop)

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. **APIs & Services → Library → enable “Google Drive API”.**
4. **IAM & Admin → Service Accounts → Create service account**  
   - Name e.g. `waxal-drive-uploader`
5. Open the SA → **Keys → Add key → Create new key → JSON**  
   - Download the JSON file (e.g. `waxal-sa.json`).
6. Note the SA email, looks like:  
   `waxal-drive-uploader@YOUR_PROJECT.iam.gserviceaccount.com`

### Step 2 — Share the Drive folder with the service account

1. Open: https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru  
2. **Share** → add the **service account email**  
3. Role: **Editor**  
4. Uncheck “Notify people” if you want → Share  

If you skip this step, upload will fail with 403 / not found.

### Step 3 — Put the JSON key on the Lightning studio

Upload `waxal-sa.json` into the studio, e.g.:

```text
/teamspace/studios/this_studio/waxal-sa.json
```

### Step 4 — Run the upload

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
git pull origin main

export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs
export GOOGLE_APPLICATION_CREDENTIALS=/teamspace/studios/this_studio/waxal-sa.json
export GDRIVE_FOLDER_ID=1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru

# See what will be packed
ls -la $WAXAL_OUTPUTS_DIR

# Upload all three languages (packs + resumable upload)
python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives

# Or only languages you have ready:
# python scripts/upload_checkpoints_gdrive.py --langs sna,lug --keep-archives
```

Shortcut:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/teamspace/studios/this_studio/waxal-sa.json
python lightning_studio_bootstrap.py upload --langs lin,sna,lug --keep-archives
```

### Step 5 — Verify

Open the folder in a browser. You should see files named like:

```text
lin_mms-300m_fold0.tar.gz
sna_mms-300m_fold0.tar.gz
lug_mms-300m_fold0.tar.gz
```

The script prints a link per file when done:

```text
OK: https://drive.google.com/file/d/FILE_ID/view
```

### If the upload drops mid-way

Just re-run the **same command**.  

- Local `.tar.gz` is kept if you used `--keep-archives` (so it won’t re-pack).  
- Drive update is chunked; re-run overwrites/updates the same filename.

Without `--keep-archives`, re-run will re-pack (slower but fine).

---

## Alternative A: rclone (also good)

If you already use rclone:

```bash
# One-time on a machine with browser (or follow rclone headless docs):
rclone config
# create remote name "gdrive", type Google Drive

# On Lightning (after rclone is installed + config copied):
export RCLONE_REMOTE="gdrive:WAXAL_checkpoints"   # remote:folder
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs

cd /teamspace/studios/this_studio/WAXAL_ZINDI
python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives
```

If `RCLONE_REMOTE` is set, the script **uses rclone only** (skips Google API auth).

---

## Alternative B: Manual download (no Drive API)

If auth is blocked and you only need a backup once:

1. In Lightning file browser, open:  
   `WAXAL_ZINDI/outputs/`
2. Download each `*_mms-300m_fold0/` folder (or zip in terminal first):

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI/outputs
tar -czf sna_mms-300m_fold0.tar.gz sna_mms-300m_fold0
# then download the .tar.gz via the studio UI
```

3. Drag the `.tar.gz` into the Drive folder in your browser.

This is **not** resumable API upload, but zero GCP setup.

---

## Common failures

| Error | Fix |
|-------|-----|
| `No Drive auth found` | Set `GOOGLE_APPLICATION_CREDENTIALS` to the SA JSON path |
| `403` / permission denied | Share the **folder** with the **SA email** as Editor |
| `Outputs dir missing` | `export WAXAL_OUTPUTS_DIR=.../WAXAL_ZINDI/outputs` |
| `SKIP lin: no lin_mms-...` | That language folder is missing; train it first or omit from `--langs` |
| Upload stalls | Re-run with `--keep-archives`; check studio disk free space for packing |
| Folder empty after “success” | You may have uploaded under a different Google account / project — open the exact folder ID link above |

---

## Restore later (download)

From Drive, download a `.tar.gz`, then on a new machine:

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
mkdir -p outputs
tar -xzf sna_mms-300m_fold0.tar.gz -C outputs/
ls outputs/sna_mms-300m_fold0/checkpoints/
ls outputs/sna_mms-300m_fold0/best_model/
```

That restores Lightning-compatible checkpoint trees for `--resume` / submission.

---

## Minimal cheat-sheet

```bash
# 1) SA JSON on studio + folder shared with SA email
export GOOGLE_APPLICATION_CREDENTIALS=/teamspace/studios/this_studio/waxal-sa.json
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs
export GDRIVE_FOLDER_ID=1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru

cd /teamspace/studios/this_studio/WAXAL_ZINDI
git pull origin main

# 2) Upload
python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives

# 3) Check
# https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru
```
