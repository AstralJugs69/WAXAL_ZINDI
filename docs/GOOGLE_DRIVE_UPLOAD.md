# Upload WAXAL checkpoints to Google Drive

**Target folder:**  
https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru  

**Folder ID:** `1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru`

**Script:** `scripts/upload_checkpoints_gdrive.py`

**What gets uploaded:** one resumable `.tar.gz` per language folder under `outputs/`:

```text
lin_mms-300m_fold0.tar.gz
sna_mms-300m_fold0.tar.gz
lug_mms-300m_fold0.tar.gz
```

---

## Critical: Service accounts cannot fill personal My Drive

If you see:

```text
Service Accounts do not have storage quota
storageQuotaExceeded
```

that is **expected** when:

- The folder lives in **your personal Google Drive (My Drive)**, and  
- You authenticate with a **service account** JSON.

Sharing the folder with the SA email does **not** fix this. SA has **0 bytes** quota on My Drive.

| Method | Works on personal Drive folder? |
|--------|----------------------------------|
| **OAuth as your Google user** | **Yes** (uses *your* quota) — **recommended** |
| **rclone** logged in as you | **Yes** |
| Service account + My Drive folder | **No** |
| Service account + **Shared Drive** (Workspace) | Yes, if SA is a member |

---

## Recommended: OAuth user token (personal Drive)

### Part 1 — Once on a laptop (browser required)

1. [Google Cloud Console](https://console.cloud.google.com/) → project  
2. Enable **Google Drive API**  
3. **APIs & Services → Credentials → Create credentials → OAuth client ID**  
   - Application type: **Desktop app**  
   - Download JSON → rename to `client_secret.json`  
4. If asked, configure OAuth consent screen (External is fine for personal use; add your email as test user).  
5. On the laptop, with this repo:

```bash
cd /path/to/WAXAL_ZINDI
# put client_secret.json in the repo root or set:
export GDRIVE_CLIENT_SECRET=/path/to/client_secret.json
export GDRIVE_TOKEN=/path/to/token_drive.json

python scripts/upload_checkpoints_gdrive.py --auth-only
```

6. Browser opens → log in as **the Google account that owns the Drive folder** → Allow.  
7. File created: **`token_drive.json`**

### Part 2 — On Lightning studio

1. Upload **`token_drive.json`** to:

```text
/teamspace/studios/this_studio/token_drive.json
```

2. **Do not** force the service account. Unset SA if you set it before:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
```

3. Upload (reuses packs if already created):

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
git pull origin main

export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs
export GDRIVE_TOKEN=/teamspace/studios/this_studio/token_drive.json
export GDRIVE_FOLDER_ID=1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru

# If packing already finished (from failed SA attempt):
ls -lh /teamspace/studios/this_studio/gdrive_upload_work/*.tar.gz
python scripts/upload_checkpoints_gdrive.py \
  --langs lin,sna,lug \
  --keep-archives \
  --skip-pack

# Or pack + upload:
python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives
```

### Part 3 — Verify

Open https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru  

You should see the `.tar.gz` files. Script also prints:

```text
OK: https://drive.google.com/file/d/FILE_ID/view
```

---

## Alternative: rclone (also uses *your* quota)

```bash
# Once (laptop): rclone config → Google Drive remote named "gdrive"
# Copy ~/.config/rclone/rclone.conf onto Lightning

export RCLONE_REMOTE="gdrive:WAXAL_checkpoints"
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs

cd /teamspace/studios/this_studio/WAXAL_ZINDI
python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives
```

If `RCLONE_REMOTE` is set, the script **skips** Google API auth and uses rclone.

---

## Alternative: manual (no API)

```bash
# Use already-packed archives if present:
ls /teamspace/studios/this_studio/gdrive_upload_work/*.tar.gz
# Download via Lightning file UI → drag into the Drive folder in the browser
```

---

## Service account — only for Shared Drives

Only if you use a **Google Workspace Shared Drive** (not a normal “My Drive” folder):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/teamspace/studios/this_studio/waxal-sa.json
export GDRIVE_USE_SERVICE_ACCOUNT=1
export GDRIVE_FOLDER_ID=...   # folder id INSIDE Shared Drive
python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives --use-service-account
```

---

## Common failures

| Error | Fix |
|-------|-----|
| **Service Accounts do not have storage quota** | Use **OAuth `token_drive.json`**, not SA, for My Drive |
| `No Drive auth found` | Run `--auth-only` on a laptop; copy token to studio |
| `403` permission | OAuth as the **owner** of the folder |
| `Outputs dir missing` | `export WAXAL_OUTPUTS_DIR=.../outputs` |
| Re-pack is slow | `--keep-archives --skip-pack` after first pack |

---

## Restore later

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
mkdir -p outputs
tar -xzf sna_mms-300m_fold0.tar.gz -C outputs/
```

---

## Minimal cheat-sheet (OAuth)

```bash
# Laptop once:
python scripts/upload_checkpoints_gdrive.py --auth-only
# → copy token_drive.json to Lightning studio root

# Lightning:
unset GOOGLE_APPLICATION_CREDENTIALS
export GDRIVE_TOKEN=/teamspace/studios/this_studio/token_drive.json
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs
export GDRIVE_FOLDER_ID=1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru

cd /teamspace/studios/this_studio/WAXAL_ZINDI
git pull origin main
python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives --skip-pack
```
