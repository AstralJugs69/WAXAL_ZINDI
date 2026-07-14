#!/usr/bin/env python3
"""
Reliable restore of WAXAL MMS checkpoints from Kaggle.

Default dataset:
  cashgenenator/waxal-mms-checkpoints

Why this exists:
  - `kaggle datasets list -m` often shows size=0 even when the dataset is full (UI shows ~59GB).
  - `download` can 404 if auth path is wrong or the CLI is confused.
  - Uploaded layout may be FOLDERS (lin_mms-300m_fold0/...) not .tar.gz.

Usage:
  python scripts/download_checkpoints_kaggle.py
  python scripts/download_checkpoints_kaggle.py --dataset cashgenenator/waxal-mms-checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def log(msg: str):
    print(msg, flush=True)


def setup_kaggle_json() -> Path:
    dest = Path.home() / ".kaggle" / "kaggle.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    for src in (
        Path("/teamspace/studios/this_studio/kaggle.json"),
        Path.cwd() / "kaggle.json",
        Path(__file__).resolve().parents[1] / "kaggle.json",
    ):
        if src.exists():
            shutil.copy2(src, dest)
            try:
                dest.chmod(0o600)
            except Exception:
                pass
            log(f"Installed kaggle.json from {src}")
            return dest
    raise FileNotFoundError(
        "kaggle.json not found. Put it at /teamspace/studios/this_studio/kaggle.json"
    )


def kaggle_cmd():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    return [sys.executable, "-m", "kaggle"]


def run_capture(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def list_dataset_files(cmd, dataset_id: str) -> list[str]:
    """Return file paths reported by the API (best-effort)."""
    # Prefer CSV for stable parsing
    r = run_capture(cmd + ["datasets", "files", dataset_id, "-v", "--csv"])
    if r.returncode != 0:
        r = run_capture(cmd + ["datasets", "files", dataset_id])
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    log("--- kaggle datasets files ---")
    log(out.strip() or "(empty)")
    names = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name"):
            continue
        # csv: name,size,creationDate...
        name = line.split(",")[0].strip().strip('"')
        if name and name not in ("name",):
            names.append(name)
    return names


def download_via_cli(cmd, dataset_id: str, dest: Path) -> Path | None:
    dest.mkdir(parents=True, exist_ok=True)
    # Clear partials
    for p in dest.glob("*"):
        if p.is_file() and p.suffix in (".zip", ".part"):
            try:
                p.unlink()
            except Exception:
                pass

    log(f"Downloading {dataset_id} → {dest} (this can take a long time ~60GB)…")
    # NOTE: do NOT always pass --unzip first; inspect zip if present
    r = subprocess.run(
        cmd + ["datasets", "download", "-d", dataset_id, "-p", str(dest), "--force"],
        check=False,
    )
    if r.returncode != 0:
        log("CLI download failed; trying Python API…")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset_id, path=str(dest), unzip=False, quiet=False, force=True)
        except Exception as exc:
            log(f"Python API download failed: {exc}")
            return None

    zips = sorted(dest.glob("*.zip"))
    if zips:
        return zips[0]
    # maybe already extracted
    if any(dest.iterdir()):
        return dest
    return None


def unzip_all(dest: Path):
    for z in dest.glob("*.zip"):
        log(f"Unzipping {z.name} …")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dest)
        log(f"  done {z.name}")


def find_lang_dirs(root: Path) -> dict[str, Path]:
    found = {}
    for lang in ("lin", "sna", "lug"):
        # exact or nested
        hits = list(root.rglob(f"{lang}_mms-300m_fold0"))
        # prefer shortest path
        hits = [h for h in hits if h.is_dir()]
        if hits:
            hits.sort(key=lambda p: len(p.parts))
            found[lang] = hits[0]
    return found


def install_to_outputs(found: dict[str, Path], outputs: Path):
    outputs.mkdir(parents=True, exist_ok=True)
    for lang, src in found.items():
        dst = outputs / src.name
        if dst.exists():
            log(f"Removing existing {dst}")
            shutil.rmtree(dst)
        log(f"Moving {src} → {dst}")
        shutil.move(str(src), str(dst))
        # sanity
        ck = dst / "checkpoints"
        bm = dst / "best_model"
        log(f"  checkpoints: {ck.exists()}  best_model: {bm.exists()}")


def main():
    ap = argparse.ArgumentParser(description="Download WAXAL checkpoints from Kaggle reliably")
    ap.add_argument(
        "--dataset",
        default=os.environ.get("KAGGLE_CHECKPOINT_DATASET", "cashgenenator/waxal-mms-checkpoints"),
    )
    ap.add_argument(
        "--download-dir",
        default=os.environ.get("WAXAL_CKPT_DL", "./ckpt_dl"),
    )
    ap.add_argument(
        "--outputs",
        default=os.environ.get(
            "WAXAL_OUTPUTS_DIR",
            "/teamspace/studios/this_studio/WAXAL_ZINDI/outputs",
        ),
    )
    ap.add_argument("--list-only", action="store_true", help="Only list remote files, do not download")
    args = ap.parse_args()

    setup_kaggle_json()
    cmd = kaggle_cmd()
    dataset_id = args.dataset
    dl = Path(args.download_dir).expanduser().resolve()
    outputs = Path(args.outputs).expanduser().resolve()

    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    log(f"Kaggle user from kaggle.json: {user}")
    log(f"Dataset: {dataset_id}")
    log("")
    log("NOTE: `kaggle datasets list -m` often shows size=0 even when the dataset")
    log("is full (~59GB in the web UI). That column is unreliable — ignore it.")
    log("Trust: website Data Explorer, or `kaggle datasets files`, or this script.")
    log("")

    # status
    st = run_capture(cmd + ["datasets", "status", dataset_id])
    log(f"status: {(st.stdout or st.stderr or '').strip()}")

    files = list_dataset_files(cmd, dataset_id)
    if files:
        log(f"API reported {len(files)} file row(s) (names may be truncated).")
    else:
        log(
            "API file list empty/failed — dataset may still be valid (web UI is source of truth). "
            "Continuing with download attempt…"
        )

    if args.list_only:
        return

    result = download_via_cli(cmd, dataset_id, dl)
    if result is None:
        raise SystemExit(
            "\nDOWNLOAD FAILED.\n"
            "Checklist:\n"
            f"  1) kaggle.json username is the owner of {dataset_id} (got: {user})\n"
            "  2) Open the dataset in browser while logged in as that user\n"
            "  3) Try: python -m kaggle datasets download -d "
            f"{dataset_id} -p ./ckpt_dl --force\n"
            "  4) Or download ZIP from the website and upload it to the studio\n"
        )

    unzip_all(dl)

    found = find_lang_dirs(dl)
    if not found:
        log("Could not find lin/sna/lug_mms-300m_fold0 directories. Tree:")
        for p in sorted(dl.rglob("*"))[:80]:
            log(f"  {p.relative_to(dl)}")
        raise SystemExit(
            "Download may have succeeded but layout unexpected. "
            "Inspect ckpt_dl/ and move folders manually into outputs/."
        )

    log(f"Found languages: {list(found.keys())}")
    install_to_outputs(found, outputs)

    log("")
    log("=" * 64)
    log("RESTORE COMPLETE")
    log(f"  outputs = {outputs}")
    for lang in ("lin", "sna", "lug"):
        d = outputs / f"{lang}_mms-300m_fold0"
        log(f"  {lang}: exists={d.exists()}")
    log("Next: python generate_submission.py --max-blank-frac 0.05")
    log("=" * 64)


if __name__ == "__main__":
    main()
