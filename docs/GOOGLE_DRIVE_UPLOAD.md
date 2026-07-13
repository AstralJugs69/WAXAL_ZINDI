# Google Drive checkpoint upload — step-by-step (no ambiguity)

**Goal:** Put these files into your Google Drive folder:

```text
lin_mms-300m_fold0.tar.gz
sna_mms-300m_fold0.tar.gz
lug_mms-300m_fold0.tar.gz
```

**Destination folder (open this in your browser):**  
https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru

**Why service account failed:** Google does not give service accounts storage on personal Drive. You must upload **as your Google user** (OAuth). Sharing the folder with the service account does **not** fix that.

---

# Overview (two machines)

| Where | What you do |
|-------|-------------|
| **YOUR LAPTOP** (has a web browser) | Create Google API credentials + log in once → get `token_drive.json` |
| **LIGHTNING STUDIO** (remote VM) | Put `token_drive.json` there + run the upload command |

You already packed archives on Lightning (good). We only fix **who** is allowed to upload.

---

# PART A — On your LAPTOP (one time only)

You need a normal computer with Chrome/Edge/Firefox. Not the Lightning terminal for this part.

---

## A1. Open Google Cloud and pick a project

1. Open: https://console.cloud.google.com/
2. Sign in with the **same Google account** that owns the Drive folder  
   (the account you use when you open the folder link above).
3. Top bar: click the **project name** dropdown.
4. Either:
   - Click an existing project, **or**
   - Click **NEW PROJECT** → name it `waxal-upload` → **CREATE** → select it.

You should now see that project name in the top bar.

---

## A2. Turn on Google Drive API

1. Open: https://console.cloud.google.com/apis/library/drive.googleapis.com  
   (or: left menu **APIs & Services** → **Library** → search **Google Drive API**)
2. Click **Google Drive API**.
3. Click the blue **ENABLE** button.
4. Wait until it says enabled / you see the API dashboard.

---

## A3. Configure the OAuth consent screen (required once)

1. Open: https://console.cloud.google.com/apis/credentials/consent  
   (or: **APIs & Services** → **OAuth consent screen**)
2. User type:
   - Choose **External** → **CREATE**  
   - (If you only see “Internal”, you are on a Workspace org; Internal is fine too.)
3. Fill **only required fields**:
   - App name: `WAXAL Upload`
   - User support email: pick your email
   - Developer contact email: your email
4. Click **SAVE AND CONTINUE**.
5. **Scopes** page: click **SAVE AND CONTINUE** (do not add scopes manually).
6. **Test users** page (if shown):
   - Click **+ ADD USERS**
   - Type **your Gmail address** (same as Drive owner)
   - **ADD** → **SAVE AND CONTINUE**
7. Summary → **BACK TO DASHBOARD**.

---

## A4. Create an OAuth “Desktop” client and download the JSON

1. Open: https://console.cloud.google.com/apis/credentials  
   (or: **APIs & Services** → **Credentials**)
2. Click **+ CREATE CREDENTIALS** at the top.
3. Click **OAuth client ID**.
4. If it asks to configure consent screen, go back to A3; otherwise continue.
5. Application type: choose **Desktop app**.
6. Name: `waxal-desktop`
7. Click **CREATE**.
8. A popup appears → click **DOWNLOAD JSON**  
   (or find the client in the list → download icon).
9. Save the file on your laptop.
10. **Rename** that file to exactly:

```text
client_secret.json
```

Put it in a folder you can find, e.g. Desktop.

---

## A5. Get the upload script on your laptop

### Option A — you already cloned the repo on the laptop

```bash
cd path/to/WAXAL_ZINDI
git pull origin main
```

### Option B — no clone yet

```bash
cd Desktop
git clone --depth 1 https://github.com/AstralJugs69/WAXAL_ZINDI.git
cd WAXAL_ZINDI
```

### Option C — no git

1. Open: https://github.com/AstralJugs69/WAXAL_ZINDI  
2. Code → Download ZIP → unzip  
3. Open a terminal **inside** the unzipped folder.

---

## A6. Put `client_secret.json` where the script expects it

Copy your renamed file into the **repo root** (same folder as `README` / `scripts/`):

```text
WAXAL_ZINDI/
  client_secret.json    ← here
  scripts/
    upload_checkpoints_gdrive.py
  ...
```

On Windows PowerShell example:

```powershell
copy "$env:USERPROFILE\Downloads\client_secret.json" .\client_secret.json
```

(Adjust the Downloads path if your file is elsewhere.)

---

## A7. Create `token_drive.json` (login in browser)

In a terminal **in the WAXAL_ZINDI folder**:

```bash
# install deps once
python -m pip install -q google-api-python-client google-auth-httplib2 google-auth-oauthlib

# create token (opens browser)
python scripts/upload_checkpoints_gdrive.py --auth-only
```

**What should happen:**

1. A browser window opens.  
2. Choose **your Google account** (Drive folder owner).  
3. Google may say “Google hasn’t verified this app”:
   - Click **Advanced**
   - Click **Go to WAXAL Upload (unsafe)** (or similar)
4. Click **Allow** / **Continue**.  
5. Terminal prints something like:

```text
Saved OAuth token → ...
Auth complete. Token file: ...
```

6. In the repo folder you now have:

```text
token_drive.json
```

**If the browser does not open:** the script will print a URL — copy it into the browser manually, finish login, then return to the terminal.

**Keep `token_drive.json` private** (it is a login key). Do not commit it to GitHub.

---

## A8. Copy `token_drive.json` to Lightning

You need this file on the Lightning machine at:

```text
/teamspace/studios/this_studio/token_drive.json
```

**How:**

1. Open your Lightning studio in the browser.  
2. Use the file browser / upload UI.  
3. Upload `token_drive.json` into the studio root  
   (`/teamspace/studios/this_studio/`),  
   **not** only inside `WAXAL_ZINDI` (either works if you set `GDRIVE_TOKEN`, but the path above is default).

Confirm on Lightning terminal:

```bash
ls -la /teamspace/studios/this_studio/token_drive.json
```

You must see the file listed. If “No such file”, the upload went to the wrong place.

---

# PART B — On LIGHTNING STUDIO (upload)

Do this in the Lightning **terminal**.

---

## B1. Go to the repo and update code

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
git pull origin main
```

---

## B2. Turn off the broken service-account method

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
```

(Optional check — should print empty or nothing useful:)

```bash
echo "SA=$GOOGLE_APPLICATION_CREDENTIALS"
```

You want SA **empty** so the script uses `token_drive.json` instead.

---

## B3. Point to your token and outputs

```bash
export GDRIVE_TOKEN=/teamspace/studios/this_studio/token_drive.json
export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs
export GDRIVE_FOLDER_ID=1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru
```

---

## B4. Check packed files already exist (from your earlier run)

```bash
ls -lh /teamspace/studios/this_studio/gdrive_upload_work/*.tar.gz
```

**If you see** files like `lin_mms-300m_fold0.tar.gz` (~20GB each): good — skip re-packing.

**If you see nothing**, run **without** `--skip-pack` in B5 (packing will take a while and needs free disk).

---

## B5. Upload

### If packs already exist (your case)

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI

python scripts/upload_checkpoints_gdrive.py \
  --langs lin,sna,lug \
  --keep-archives \
  --skip-pack
```

### If packs do not exist

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI

python scripts/upload_checkpoints_gdrive.py \
  --langs lin,sna,lug \
  --keep-archives
```

---

## B6. What “success” looks like in the terminal

You want lines like:

```text
Auth: OAuth user token (/teamspace/studios/this_studio/token_drive.json) — uses your Drive storage quota
Uploading lin_mms-300m_fold0.tar.gz ...
  creating new Drive file ...
  upload 10%
  upload 20%
  ...
  OK: https://drive.google.com/file/d/xxxxx/view
```

Then the same for sna and lug.

**Then open in browser:**  
https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru  

Refresh. The three `.tar.gz` files should appear.

---

## B7. If something fails

| What you see | What to do |
|--------------|------------|
| `No such file: token_drive.json` | Re-do A8; check `ls` path exactly |
| `Service Accounts do not have storage quota` | You still have SA active: run `unset GOOGLE_APPLICATION_CREDENTIALS` and ensure `GDRIVE_TOKEN` is set |
| `Auth: service account (...)` | Wrong auth — unset SA; must say `Auth: OAuth user token` |
| Browser “access blocked” on laptop | Add your email as **Test user** in OAuth consent screen (A3 step 6) |
| `invalid_client` | Wrong/missing `client_secret.json`; re-download from A4 |
| `SKIP lin: no ...` | That language folder is missing under `outputs/`; upload only langs that exist: `--langs sna,lug` |
| Upload dies at 40% | Re-run the **same** B5 command (resumable; keep `--keep-archives --skip-pack`) |
| Drive folder still empty | Logged into a different Google account in browser than the one used for OAuth |

---

# PART C — Emergency backup (no Google API)

If OAuth is too hard right now and you only need a copy off the VM:

1. Lightning left sidebar → **file browser**  
2. Go to:  
   `/teamspace/studios/this_studio/gdrive_upload_work/`  
3. Download each `.tar.gz` to your laptop (slow but works).  
4. Open https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru  
5. Drag the files from your laptop into that folder.

---

# PART D — Restore on a new Lightning VM later

1. Download a `.tar.gz` from Drive to the new studio.  
2. Run:

```bash
cd /teamspace/studios/this_studio/WAXAL_ZINDI
mkdir -p outputs
tar -xzf /path/to/sna_mms-300m_fold0.tar.gz -C outputs/
ls outputs/sna_mms-300m_fold0/checkpoints/
```

---

# One-page checklist

**Laptop**

- [ ] Cloud project selected  
- [ ] Drive API enabled  
- [ ] OAuth consent screen + **your email as test user**  
- [ ] Desktop OAuth client downloaded as `client_secret.json`  
- [ ] Repo cloned / updated  
- [ ] `python scripts/upload_checkpoints_gdrive.py --auth-only`  
- [ ] `token_drive.json` exists  

**Lightning**

- [ ] `token_drive.json` at `/teamspace/studios/this_studio/token_drive.json`  
- [ ] `unset GOOGLE_APPLICATION_CREDENTIALS`  
- [ ] `export GDRIVE_TOKEN=...` and `WAXAL_OUTPUTS_DIR=...`  
- [ ] `git pull`  
- [ ] Upload with `--keep-archives --skip-pack` if packs exist  
- [ ] Browser shows files in the Drive folder  

---

*If you get stuck, paste: (1) which letter-step you are on (A1, A7, B5…), (2) the exact last 20 lines of terminal output.*
