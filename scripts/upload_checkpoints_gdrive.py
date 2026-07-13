#!/usr/bin/env python3
"""
Upload WAXAL training outputs to Google Drive (resumable).

Default folder (WAXAL checkpoints):
  https://drive.google.com/drive/folders/1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru
  Folder ID: 1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru

Auth (pick one):
  1) Service account JSON:
       export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
     Share the Drive folder with the SA email (Editor).
  2) OAuth client secrets (browser once on a machine with UI):
       place client_secret.json and run; creates token_drive.json
  3) rclone remote (if installed and configured):
       export RCLONE_REMOTE=gdrive:WAXAL_checkpoints

Usage:
  python scripts/upload_checkpoints_gdrive.py
  python scripts/upload_checkpoints_gdrive.py --langs lin,sna,lug
  python scripts/upload_checkpoints_gdrive.py --outputs /path/to/outputs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DEFAULT_FOLDER_ID = "1r6Vzl9MjoRzC5wKXU699eOwOF1A7A_ru"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def log(msg: str):
    print(msg, flush=True)


def resolve_outputs(args_outputs: str | None) -> Path:
    if args_outputs:
        return Path(args_outputs).expanduser().resolve()
    env = os.environ.get("WAXAL_OUTPUTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # common Lightning layout
    for p in (
        Path("/teamspace/studios/this_studio/WAXAL_ZINDI/outputs"),
        Path.cwd() / "outputs",
        Path(__file__).resolve().parents[1] / "outputs",
    ):
        if p.exists():
            return p
    return Path.cwd() / "outputs"


def pack_lang_dir(lang_dir: Path, archive_path: Path) -> Path:
    """Create a single .tar.gz of one language output (resumable upload unit)."""
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
    # rclone copyto is resumable for large files
    dest = f"{remote.rstrip('/')}/{local_file.name}"
    log(f"  rclone copyto {local_file} → {dest}")
    r = subprocess.run(
        ["rclone", "copyto", str(local_file), dest, "--progress", "--retries", "10"],
        check=False,
    )
    return r.returncode == 0


def get_drive_service():
    """Build Drive API v3 service via SA or OAuth user token."""
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
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
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

    sa = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa and Path(sa).exists():
        creds = service_account.Credentials.from_service_account_file(sa, scopes=SCOPES)
        log(f"Auth: service account ({sa})")
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # OAuth user token
    token_path = Path(
        os.environ.get("GDRIVE_TOKEN", "/teamspace/studios/this_studio/token_drive.json")
    )
    client_path = Path(
        os.environ.get(
            "GDRIVE_CLIENT_SECRET",
            "/teamspace/studios/this_studio/client_secret.json",
        )
    )
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_path.exists():
                raise FileNotFoundError(
                    "No Drive auth found.\n"
                    "  Option A: export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json\n"
                    "            (share Drive folder with the SA email)\n"
                    "  Option B: place OAuth client_secret.json at "
                    f"{client_path} and re-run on a machine that can open a browser\n"
                    "  Option C: configure rclone and export RCLONE_REMOTE=gdrive:folder"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(client_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        log(f"Saved OAuth token → {token_path}")
    log("Auth: OAuth user credentials")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_existing_file(service, folder_id: str, name: str):
    q = (
        f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    )
    res = (
        service.files()
        .list(q=q, spaces="drive", fields="files(id, name, size)", pageSize=5)
        .execute()
    )
    files = res.get("files", [])
    return files[0] if files else None


def upload_resumable(service, local_path: Path, folder_id: str):
    from googleapiclient.http import MediaFileUpload

    name = local_path.name
    existing = find_existing_file(service, folder_id, name)
    media = MediaFileUpload(
        str(local_path),
        mimetype="application/gzip",
        resumable=True,
        chunksize=8 * 1024 * 1024,  # 8MB chunks — good for flaky networks
    )
    if existing:
        file_id = existing["id"]
        log(f"  updating existing Drive file id={file_id} ({name})")
        request = service.files().update(fileId=file_id, media_body=media)
    else:
        meta = {"name": name, "parents": [folder_id]}
        log(f"  creating new Drive file ({name})")
        request = service.files().create(body=meta, media_body=media, fields="id,name,size")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log(f"  upload {int(status.progress() * 100)}%")
    log(f"  OK: https://drive.google.com/file/d/{response.get('id')}/view")
    return response


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
    args = p.parse_args()

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
        # prefer fold0 dir naming used by lightning_mms
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
            if not ok:
                log(f"  rclone failed for {arc.name}")
            else:
                log(f"  rclone OK: {arc.name}")
        log("Done (rclone path).")
        return

    service = get_drive_service()
    for arc in archives:
        log(f"Uploading {arc.name} ({arc.stat().st_size / 1e9:.2f} GB)…")
        for attempt in range(1, 6):
            try:
                upload_resumable(service, arc, folder_id)
                break
            except Exception as exc:
                log(f"  attempt {attempt} failed: {exc}")
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
