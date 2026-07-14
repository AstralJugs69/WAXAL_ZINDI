#!/usr/bin/env python3
"""
Reliable restore of WAXAL MMS checkpoints from Kaggle.

Default dataset:
  cashgenenator/waxal-mms-checkpoints

Quirks handled:
  - `list -m` size column often shows 0 (ignore it; web UI is correct)
  - Bulk `datasets download` may 404 even when files list works
    → fall back to **per-file** download (paginated)
  - Upload layout may be nested:
      lin_mms-300m_fold0/lin_mms-300m_fold0/checkpoints/...
    We flatten to:
      outputs/lin_mms-300m_fold0/...

Usage:
  python scripts/download_checkpoints_kaggle.py
  python scripts/download_checkpoints_kaggle.py --dataset cashgenenator/waxal-mms-checkpoints
  python lightning_studio_bootstrap.py download
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
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


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def list_all_files(api, dataset_id: str) -> list[str]:
    """
    Paginate through every file in the dataset.
    Returns list of remote relative paths.
    """
    names: list[str] = []
    # Newer kaggle API: dataset_list_files may not paginate; try page tokens via raw CLI first
    cmd = kaggle_cmd()
    page_token = None
    page = 0
    while True:
        page += 1
        c = cmd + ["datasets", "files", dataset_id, "-v", "--csv"]
        if page_token:
            # Some CLI versions support --page-token; if not, break after first page
            c.extend(["--page-token", page_token])
        r = run_capture(c)
        out = r.stdout or ""
        err = r.stderr or ""
        if page == 1:
            log("--- kaggle datasets files (page 1) ---")
            # Don't dump huge tokens every time
            for line in (out + err).splitlines()[:5]:
                if "Next Page Token" in line:
                    log("(has next page token… will paginate)")
                elif line.strip():
                    log(line[:200])

        # Parse next page token if present
        new_token = None
        for line in (out + err).splitlines():
            if "Next Page Token" in line or "nextPageToken" in line.lower():
                # formats: "Next Page Token = TOKEN" or CSV field
                if "=" in line:
                    new_token = line.split("=", 1)[-1].strip()
                else:
                    parts = line.split()
                    if parts:
                        new_token = parts[-1].strip()

        for line in out.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("name"):
                continue
            if "Next Page Token" in line:
                continue
            name = line.split(",")[0].strip().strip('"')
            if name and name not in names and not name.startswith("Next"):
                names.append(name)

        if not new_token or new_token == page_token:
            # CLI may not support page-token; use Python API pagination if available
            break
        page_token = new_token
        if page > 100:
            log("WARNING: stopped pagination at 100 pages")
            break
        time.sleep(0.2)

    # Python API fallback / fill
    try:
        # dataset_list_files returns first page only on many versions
        fl = api.dataset_list_files(dataset_id)
        file_list = getattr(fl, "files", fl) or []
        for f in file_list:
            n = getattr(f, "name", None) or str(f)
            if n and n not in names:
                names.append(n)
    except Exception as exc:
        log(f"Python list_files note: {exc}")

    # If still short, try walking with page-token via API raw if present
    if hasattr(api, "dataset_list_files_with_http_info"):
        pass

    log(f"Collected {len(names)} remote file path(s)")
    if names:
        log("Sample paths:")
        for n in names[:8]:
            log(f"  {n}")
        if len(names) > 8:
            log(f"  ... +{len(names) - 8} more")
    return names


def download_bulk(api, dataset_id: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    log(f"Trying bulk download of {dataset_id} …")
    try:
        api.dataset_download_files(
            dataset_id, path=str(dest), unzip=True, quiet=False, force=True
        )
        # success if we got anything substantial
        total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        log(f"Bulk download wrote ~{total / 1e9:.2f} GB under {dest}")
        return total > 1_000_000
    except Exception as exc:
        log(f"Bulk download failed: {exc}")
        return False


def download_file_cli(cmd, dataset_id: str, file_name: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    # kaggle datasets download -d ID -f path/to/file -p dest
    r = subprocess.run(
        cmd
        + [
            "datasets",
            "download",
            "-d",
            dataset_id,
            "-f",
            file_name,
            "-p",
            str(dest),
            "--force",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        # show short error once
        err = (r.stderr or r.stdout or "").strip().splitlines()
        if err:
            log(f"    CLI -f failed: {err[-1][:180]}")
        return False
    return True


def download_file_api(api, dataset_id: str, file_name: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_file(
            dataset_id, file_name, path=str(dest), force=True, quiet=True
        )
        return True
    except Exception as exc:
        log(f"    API file failed: {exc}")
        return False


def place_downloaded_file(dest: Path, remote_name: str):
    """
    Kaggle often drops the file basename into dest/, possibly zipped.
    Move into dest/remote_name hierarchy when possible.
    """
    base = Path(remote_name).name
    # unzip any zip that appeared
    for z in dest.glob("*.zip"):
        # only unzip small-ish zips that match this file
        if base in z.name or z.stat().st_size < 50_000_000:
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(dest)
                z.unlink(missing_ok=True)
            except Exception:
                pass

    target = dest / remote_name
    target.parent.mkdir(parents=True, exist_ok=True)
    # if file already at target
    if target.exists():
        return
    # find basename in dest tree (recent)
    candidates = list(dest.rglob(base))
    # prefer shallow matches not already under correct path
    for c in candidates:
        if c.is_file() and c.resolve() != target.resolve():
            if not target.exists():
                shutil.move(str(c), str(target))
            break


def download_per_file(api, dataset_id: str, names: list[str], dest: Path) -> int:
    cmd = kaggle_cmd()
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    total = len(names)
    log(f"Per-file download of {total} files (fallback because bulk 404)…")
    for i, name in enumerate(names, 1):
        # Skip huge lightning logs noise optional? keep all for completeness
        log(f"[{i}/{total}] {name}")
        success = download_file_api(api, dataset_id, name, dest)
        if not success:
            success = download_file_cli(cmd, dataset_id, name, dest)
        if success:
            place_downloaded_file(dest, name)
            ok += 1
        else:
            log(f"  FAILED: {name}")
        if i % 10 == 0:
            time.sleep(0.5)
    log(f"Per-file download: {ok}/{total} succeeded")
    return ok


def find_lang_dirs(root: Path) -> dict[str, Path]:
    """
    Find language output roots. Handles double nesting:
      lin_mms-300m_fold0/lin_mms-300m_fold0/checkpoints
    Prefer the innermost dir that contains checkpoints/ or best_model/.
    """
    found = {}
    for lang in ("lin", "sna", "lug"):
        pattern = f"{lang}_mms-300m_fold0"
        hits = [p for p in root.rglob(pattern) if p.is_dir()]
        if not hits:
            continue
        # Prefer dirs that look like real training outputs
        scored = []
        for h in hits:
            score = 0
            if (h / "checkpoints").is_dir():
                score += 10
            if (h / "best_model").is_dir():
                score += 10
            # prefer deeper (inner) nest when both exist
            score += len(h.parts) * 0.01
            scored.append((score, h))
        scored.sort(key=lambda x: x[0], reverse=True)
        found[lang] = scored[0][1]
    return found


def install_to_outputs(found: dict[str, Path], outputs: Path):
    outputs.mkdir(parents=True, exist_ok=True)
    for lang, src in found.items():
        dst = outputs / f"{lang}_mms-300m_fold0"
        if dst.exists():
            log(f"Removing existing {dst}")
            shutil.rmtree(dst)
        log(f"Copying {src} → {dst}")
        shutil.copytree(src, dst)
        log(
            f"  checkpoints={ (dst / 'checkpoints').exists() } "
            f"best_model={ (dst / 'best_model').exists() }"
        )


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
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument(
        "--force-per-file",
        action="store_true",
        help="Skip bulk download; always use per-file mode",
    )
    args = ap.parse_args()

    setup_kaggle_json()
    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    dataset_id = args.dataset
    dl = Path(args.download_dir).expanduser().resolve()
    outputs = Path(args.outputs).expanduser().resolve()

    log(f"Kaggle user from kaggle.json: {user}")
    log(f"Dataset: {dataset_id}")
    log("")
    log("NOTE: `kaggle datasets list -m` size=0 is often a LIE. Web UI / files list are truth.")
    log("")

    api = get_api()
    names = list_all_files(api, dataset_id)
    if args.list_only:
        return

    if not names:
        log("WARNING: file list empty from API — will still try bulk download")

    dl.mkdir(parents=True, exist_ok=True)
    bulk_ok = False
    if not args.force_per_file:
        bulk_ok = download_bulk(api, dataset_id, dl)

    if not bulk_ok:
        if not names:
            raise SystemExit(
                "Bulk download failed and no file list available.\n"
                "Open the dataset in the browser and use Download, or fix kaggle.json."
            )
        n_ok = download_per_file(api, dataset_id, names, dl)
        if n_ok == 0:
            raise SystemExit("All per-file downloads failed.")

    found = find_lang_dirs(dl)
    if not found:
        log("Could not locate language folders. First 60 paths under download dir:")
        for p in sorted(dl.rglob("*"))[:60]:
            if p.is_file():
                log(f"  {p.relative_to(dl)}")
        raise SystemExit("Layout unexpected — inspect ckpt_dl/ manually.")

    log(f"Found languages: {sorted(found.keys())}")
    install_to_outputs(found, outputs)

    log("")
    log("=" * 64)
    log("RESTORE COMPLETE")
    log(f"  outputs = {outputs}")
    for lang in ("lin", "sna", "lug"):
        d = outputs / f"{lang}_mms-300m_fold0"
        log(f"  {lang}: {d.exists()}")
    log("Next: python generate_submission.py --max-blank-frac 0.05 --hf_token '…'")
    log("=" * 64)


if __name__ == "__main__":
    main()
