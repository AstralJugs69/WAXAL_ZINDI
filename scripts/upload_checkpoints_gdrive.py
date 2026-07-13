#!/usr/bin/env python3
"""
Upload WAXAL training outputs to Google Drive (resumable).

Default folder:
  https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru
  Folder ID: 1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru

IMPORTANT — Service accounts on personal My Drive:
  Google error: "Service Accounts do not have storage quota"
  SA uploads to *My Drive* folders fail even if the folder is shared with the SA.
  Use one of:
    A) OAuth as YOUR Google user (recommended for personal Drive)
    B) Shared Drive (Team Drive) + add the SA as a member
    C) rclone configured with your user account

Auth order:
  1) token_drive.json / GDRIVE_TOKEN  (OAuth user — uses YOUR storage quota)
  2) RCLONE_REMOTE
  3) GOOGLE_APPLICATION_CREDENTIALS only if GDRIVE_USE_SERVICE_ACCOUNT=1
     (and folder must be a Shared Drive, or upload will 403)

Usage:
  # After placing token_drive.json on the studio:
  python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives

  # Create token on a laptop with browser (once):
  python scripts/upload_checkpoints_gdrive.py --auth-only
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DEFAULT_FOLDER_ID = "1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru"
# drive.file = files created by this app; enough for create-in-folder as the user.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

SA_QUOTA_HELP = """
========================================================================
Service Accounts do NOT have storage quota on personal Google Drive.
Even if you Shared the folder with the SA email, uploads will 403.

FIX — pick one:

A) OAuth as YOUR user (recommended for folder in My Drive)
   1. On a laptop with a browser, enable Drive API + create OAuth Desktop client.
   2. Download client_secret JSON as client_secret.json
   3. Run:  python scripts/upload_checkpoints_gdrive.py --auth-only
   4. Copy the created token_drive.json to Lightning studio:
        /teamspace/studios/this_studio/token_drive.json
   5. Unset SA for upload:
        unset GOOGLE_APPLICATION_CREDENTIALS
        # or do not set GDRIVE_USE_SERVICE_ACCOUNT
   6. Re-run upload on Lightning.

B) Shared Drive (Team Drive) — only if you have Google Workspace
   - Create a Shared Drive, add the SA as Content manager
   - Set: export GDRIVE_USE_SERVICE_ACCOUNT=1
   - Set folder id to a folder INSIDE the Shared Drive

C) rclone with your Google user (also good)
   rclone config  → remote gdrive
   export RCLONE_REMOTE="gdrive:WAXAL_checkpoints"
   python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug --keep-archives
========================================================================
"""


def log(msg: str):
    print(msg, flush=True)


def resolve_outputs(args_outputs: str | None) -> Path:
    if args_outputs:
        return Path(args_outputs).expanduser().resolve()
    env = os.environ.get("WAXAL_OUTPUTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for p in (
        Path("/teamspace/studios/this_studio/WAXAL_ZINDI/outputs"),
        Path.cwd() / "outputs",
        Path(__file__).resolve().parents[1] / "outputs",
    ):
        if p.exists():
            return p
    return Path.cwd() / "outputs"


def pack_lang_dir(lang_dir: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        log(f"  archive exists, reusing: {archive_path.name}")
        return archive_path
    log(f"  packing {lang_dir.name} → {archive_path.name} …")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(lang_dir, arcname=lang_dir.name)
    size_gb = archive_path.stat().st_size / (1024**3)
    log(f"  packed {size_gb:.2f} GB")
    return archive_path


def try_rclone(local_file: Path, remote: str) -> bool:
    if not shutil.which("rclone"):
        return False
    dest = f"{remote.rstrip('/')}/{local_file.name}"
    log(f"  rclone copyto {local_file} → {dest}")
    r = subprocess.run(
        ["rclone", "copyto", str(local_file), dest, "--progress", "--retries", "10"],
        check=False,
    )
    return r.returncode == 0


def _install_google_libs():
    log("Installing google-api-python-client google-auth-oauthlib …")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "google-api-python-client",
            "google-auth-httplib2",
            "google-auth-oauthlib",
        ]
    )


def get_oauth_paths():
    token_path = Path(
        os.environ.get("GDRIVE_TOKEN", "/teamspace/studios/this_studio/token_drive.json")
    )
    # also allow repo-local / cwd
    for alt in (
        token_path,
        Path.cwd() / "token_drive.json",
        Path("/teamspace/studios/this_studio/token_drive.json"),
    ):
        if alt.exists():
            token_path = alt
            break
    client_path = Path(
        os.environ.get(
            "GDRIVE_CLIENT_SECRET",
            "/teamspace/studios/this_studio/client_secret.json",
        )
    )
    for alt in (
        client_path,
        Path.cwd() / "client_secret.json",
        Path("/teamspace/studios/this_studio/client_secret.json"),
    ):
        if alt.exists():
            client_path = alt
            break
    return token_path, client_path


def get_drive_service(prefer_sa: bool = False):
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        _install_google_libs()
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

    token_path, client_path = get_oauth_paths()
    sa = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    force_sa = prefer_sa or os.environ.get("GDRIVE_USE_SERVICE_ACCOUNT", "").strip() in (
        "1",
        "true",
        "yes",
    )

    # Prefer OAuth user token (uses YOUR Drive quota) unless forced SA
    if not force_sa and token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
        if creds and creds.valid:
            log(f"Auth: OAuth user token ({token_path}) — uses your Drive storage quota")
            return build("drive", "v3", credentials=creds, cache_discovery=False)

    if force_sa and sa and Path(sa).exists():
        log(f"Auth: service account ({sa})")
        log(
            "NOTE: SA only works for Shared Drives, not personal My Drive folders. "
            "If you get storageQuotaExceeded, use OAuth token instead."
        )
        creds = service_account.Credentials.from_service_account_file(sa, scopes=SCOPES)
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # Interactive OAuth (needs browser — use --auth-only on a laptop)
    if client_path.exists():
        log(f"Starting OAuth browser flow using {client_path} …")
        flow = InstalledAppFlow.from_client_secrets_file(str(client_path), SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        log(f"Saved OAuth token → {token_path}")
        log("Copy this token_drive.json to Lightning if you ran auth on a laptop.")
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # Last resort: SA without force (will likely 403 on personal Drive)
    if sa and Path(sa).exists():
        log(f"WARNING: Only service account found ({sa}).")
        log(SA_QUOTA_HELP)
        log("Proceeding with SA anyway (will fail on personal My Drive)…")
        creds = service_account.Credentials.from_service_account_file(sa, scopes=SCOPES)
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    raise FileNotFoundError(
        "No Drive auth found.\n" + SA_QUOTA_HELP +
        f"\nExpected OAuth token at: {token_path}\n"
        f"Or client_secret at: {client_path}\n"
        "Or set RCLONE_REMOTE / GOOGLE_APPLICATION_CREDENTIALS appropriately."
    )


def find_existing_file(service, folder_id: str, name: str):
    q = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    res = (
        service.files()
        .list(
            q=q,
            spaces="drive",
            fields="files(id, name, size)",
            pageSize=5,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = res.get("files", [])
    return files[0] if files else None


def upload_resumable(service, local_path: Path, folder_id: str):
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError, ResumableUploadError

    name = local_path.name
    existing = find_existing_file(service, folder_id, name)
    media = MediaFileUpload(
        str(local_path),
        mimetype="application/gzip",
        resumable=True,
        chunksize=8 * 1024 * 1024,
    )
    if existing:
        file_id = existing["id"]
        log(f"  updating existing Drive file id={file_id} ({name})")
        request = service.files().update(
            fileId=file_id,
            media_body=media,
            supportsAllDrives=True,
        )
    else:
        meta = {"name": name, "parents": [folder_id]}
        log(f"  creating new Drive file ({name})")
        request = service.files().create(
            body=meta,
            media_body=media,
            fields="id,name,size",
            supportsAllDrives=True,
        )

    response = None
    try:
        while response is None:
            status, response = request.next_chunk()
            if status:
                log(f"  upload {int(status.progress() * 100)}%")
    except (HttpError, ResumableUploadError) as exc:
        err = str(exc)
        if "storageQuotaExceeded" in err or "Service Accounts do not have storage quota" in err:
            log(SA_QUOTA_HELP)
        raise
    log(f"  OK: https://drive.google.com/file/d/{response.get('id')}/view")
    return response


def cmd_auth_only():
    """Run OAuth on a machine with a browser; write token_drive.json."""
    token_path, client_path = get_oauth_paths()
    if not client_path.exists():
        raise SystemExit(
            f"Place OAuth Desktop client JSON at:\n  {client_path}\n"
            "Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID → Desktop app"
        )
    get_drive_service(prefer_sa=False)
    log(f"Auth complete. Token file: {get_oauth_paths()[0]}")
    log("Upload token_drive.json to Lightning studio, then run upload WITHOUT service account.")


def main():
    p = argparse.ArgumentParser(description="Upload WAXAL checkpoints to Google Drive (resumable)")
    p.add_argument("--outputs", type=str, default=None)
    p.add_argument("--langs", type=str, default="lin,sna,lug")
    p.add_argument(
        "--folder-id",
        type=str,
        default=os.environ.get("GDRIVE_FOLDER_ID", DEFAULT_FOLDER_ID),
    )
    p.add_argument(
        "--work-dir",
        type=str,
        default=os.environ.get(
            "WAXAL_UPLOAD_WORK",
            "/teamspace/studios/this_studio/gdrive_upload_work",
        ),
    )
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--skip-pack", action="store_true", help="Upload existing .tar.gz only")
    p.add_argument(
        "--auth-only",
        action="store_true",
        help="Only run OAuth browser flow and save token_drive.json (use on laptop)",
    )
    p.add_argument(
        "--use-service-account",
        action="store_true",
        help="Force SA auth (only for Shared Drives)",
    )
    args = p.parse_args()

    if args.auth_only:
        cmd_auth_only()
        return

    if args.use_service_account:
        os.environ["GDRIVE_USE_SERVICE_ACCOUNT"] = "1"

    outputs = resolve_outputs(args.outputs)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    folder_id = args.folder_id

    log("=" * 64)
    log("WAXAL → Google Drive checkpoint upload (resumable)")
    log(f"  outputs   = {outputs}")
    log(f"  folder_id = {folder_id}")
    log(f"  langs     = {langs}")
    log(f"  work_dir  = {work}")
    log("=" * 64)

    if not outputs.exists():
        raise SystemExit(f"Outputs dir missing: {outputs}")

    archives = []
    for lang in langs:
        candidates = sorted(outputs.glob(f"{lang}_mms-300m_fold*"))
        if not candidates:
            log(f"SKIP {lang}: no {lang}_mms-300m_fold* under {outputs}")
            continue
        lang_dir = candidates[0]
        if len(candidates) > 1:
            log(f"  note: multiple folds for {lang}, packing first: {lang_dir.name}")
        archive = work / f"{lang_dir.name}.tar.gz"
        if not args.skip_pack:
            pack_lang_dir(lang_dir, archive)
        elif not archive.exists():
            log(f"SKIP {lang}: --skip-pack but missing {archive}")
            continue
        archives.append(archive)

    if not archives:
        raise SystemExit("Nothing to upload.")

    rclone_remote = os.environ.get("RCLONE_REMOTE")
    if rclone_remote:
        log(f"Using rclone remote: {rclone_remote}")
        for arc in archives:
            ok = try_rclone(arc, rclone_remote)
            log(f"  rclone {'OK' if ok else 'FAILED'}: {arc.name}")
        log("Done (rclone path).")
        return

    # Warn if SA is set and no OAuth token — will almost certainly fail on My Drive
    token_path, _ = get_oauth_paths()
    sa = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    force_sa = os.environ.get("GDRIVE_USE_SERVICE_ACCOUNT", "").strip() in ("1", "true", "yes")
    if sa and Path(sa).exists() and not token_path.exists() and not force_sa:
        log("WARNING: Only a service account JSON is configured.")
        log("Personal My Drive folders will REJECT SA uploads (no storage quota).")
        log(SA_QUOTA_HELP)

    service = get_drive_service(prefer_sa=force_sa)
    for arc in archives:
        log(f"Uploading {arc.name} ({arc.stat().st_size / 1e9:.2f} GB)…")
        for attempt in range(1, 6):
            try:
                upload_resumable(service, arc, folder_id)
                break
            except Exception as exc:
                log(f"  attempt {attempt} failed: {exc}")
                if "storageQuotaExceeded" in str(exc) or "Service Accounts do not have storage quota" in str(exc):
                    log(SA_QUOTA_HELP)
                    raise SystemExit(1) from exc
                if attempt == 5:
                    raise
                time.sleep(5 * attempt)

    if not args.keep_archives:
        for arc in archives:
            try:
                arc.unlink()
                log(f"Removed local archive {arc.name}")
            except Exception:
                pass

    log("All uploads finished.")
    log(f"Drive folder: https://drive.google.com/drive/folders/{folder_id}")


if __name__ == "__main__":
    main()
