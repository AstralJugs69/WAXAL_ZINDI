#!/usr/bin/env python3
"""
Upload WAXAL checkpoints to Kaggle as a dataset (resumable via re-run / versioning).

Uses your existing kaggle.json (same as HF cache download).

Default dataset slug:
  {kaggle_username}/waxal-mms-checkpoints

What is uploaded (one .tar.gz per language that exists):
  lin_mms-300m_fold0.tar.gz
  sna_mms-300m_fold0.tar.gz
  lug_mms-300m_fold0.tar.gz

Usage (Lightning):
  export WAXAL_OUTPUTS_DIR=/teamspace/studios/this_studio/WAXAL_ZINDI/outputs
  # kaggle.json at ~/.kaggle/kaggle.json or /teamspace/studios/this_studio/kaggle.json

  python scripts/upload_checkpoints_kaggle.py --langs lin,sna,lug

  # Reuse packs from a previous attempt:
  python scripts/upload_checkpoints_kaggle.py --langs lin,sna,lug --skip-pack
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DEFAULT_DATASET_SLUG_SUFFIX = "waxal-mms-checkpoints"
DEFAULT_TITLE = "WAXAL MMS Checkpoints (lin/sna/lug)"


def log(msg: str):
    print(msg, flush=True)


def run(cmd, check=True):
    log(">>> " + " ".join(map(str, cmd)))
    r = subprocess.run(list(map(str, cmd)))
    if check and r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode}): {cmd}")
    return r.returncode


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
        "kaggle.json not found.\n"
        "Put it at /teamspace/studios/this_studio/kaggle.json\n"
        "or ~/.kaggle/kaggle.json"
    )


def read_kaggle_username(kaggle_json: Path) -> str:
    data = json.loads(kaggle_json.read_text(encoding="utf-8"))
    user = data.get("username")
    if not user:
        raise ValueError(f"No 'username' in {kaggle_json}")
    return user


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
    if archive_path.exists() and archive_path.stat().st_size > 0:
        log(f"  reusing pack: {archive_path.name} ({archive_path.stat().st_size / 1e9:.2f} GB)")
        return archive_path
    log(f"  packing {lang_dir.name} → {archive_path.name} …")
    # Prefer gzip for smaller upload; can take a while
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(lang_dir, arcname=lang_dir.name)
    log(f"  packed {archive_path.stat().st_size / 1e9:.2f} GB")
    return archive_path


def ensure_kaggle_cli():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=False)
    # prefer module entrypoint for path reliability
    return [sys.executable, "-m", "kaggle"]


def write_metadata(staging: Path, owner: str, slug_suffix: str, title: str, is_private: bool):
    meta = {
        "title": title,
        "id": f"{owner}/{slug_suffix}",
        "licenses": [{"name": "other"}],
        "keywords": ["waxal", "asr", "mms", "checkpoints"],
        "subtitle": "Per-language MMS-300M Lightning checkpoints for WAXAL",
        "description": (
            "Resumable training artifacts for WAXAL ASR (lin/sna/lug).\n\n"
            "Each `*_mms-300m_fold0.tar.gz` unpacks to "
            "`outputs/{lang}_mms-300m_fold0/` with `checkpoints/` and `best_model/`.\n\n"
            "Restore:\n"
            "```\n"
            "mkdir -p outputs && tar -xzf sna_mms-300m_fold0.tar.gz -C outputs/\n"
            "```\n"
        ),
        "isPrivate": is_private,
        "resources": [],
    }
    path = staging / "dataset-metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Wrote {path}")
    log(f"  dataset id: {meta['id']}  private={is_private}")
    return meta["id"]


def dataset_exists(kaggle_cmd: list, dataset_id: str) -> bool:
    # `kaggle datasets status OWNER/SLUG` returns 0 if exists
    r = subprocess.run(
        kaggle_cmd + ["datasets", "status", dataset_id],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        return True
    # fallback list
    r2 = subprocess.run(
        kaggle_cmd + ["datasets", "list", "-m", "--csv"],
        capture_output=True,
        text=True,
    )
    if r2.returncode == 0 and dataset_id.split("/")[-1] in (r2.stdout or ""):
        return True
    return False


def main():
    p = argparse.ArgumentParser(description="Upload WAXAL checkpoints to a Kaggle dataset")
    p.add_argument("--outputs", type=str, default=None)
    p.add_argument("--langs", type=str, default="lin,sna,lug")
    p.add_argument(
        "--slug",
        type=str,
        default=os.environ.get("KAGGLE_CHECKPOINT_SLUG", DEFAULT_DATASET_SLUG_SUFFIX),
        help="Dataset slug suffix (default: waxal-mms-checkpoints)",
    )
    p.add_argument("--title", type=str, default=DEFAULT_TITLE)
    p.add_argument(
        "--work-dir",
        type=str,
        default=os.environ.get(
            "WAXAL_KAGGLE_UPLOAD_WORK",
            "/teamspace/studios/this_studio/kaggle_checkpoint_upload",
        ),
    )
    p.add_argument(
        "--pack-dir",
        type=str,
        default=os.environ.get(
            "WAXAL_UPLOAD_WORK",
            "/teamspace/studios/this_studio/gdrive_upload_work",
        ),
        help="Where .tar.gz packs live (reuses packs from earlier Drive attempt)",
    )
    p.add_argument("--skip-pack", action="store_true")
    p.add_argument("--public", action="store_true", help="Make dataset public (default private)")
    p.add_argument(
        "--message",
        type=str,
        default="Update WAXAL MMS checkpoints",
        help="Version message when dataset already exists",
    )
    args = p.parse_args()

    kaggle_json = setup_kaggle_json()
    owner = read_kaggle_username(kaggle_json)
    kaggle_cmd = ensure_kaggle_cli()

    outputs = resolve_outputs(args.outputs)
    pack_dir = Path(args.pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(args.work_dir)
    if staging.exists():
        # clean only metadata/data copies, not packs
        for f in staging.glob("*"):
            if f.name == "dataset-metadata.json" or f.suffix in (".gz", ".zip", ".tar"):
                try:
                    if f.is_file() and f.parent == staging:
                        f.unlink()
                except Exception:
                    pass
    staging.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    log("=" * 64)
    log("WAXAL → Kaggle dataset upload")
    log(f"  kaggle user = {owner}")
    log(f"  outputs     = {outputs}")
    log(f"  pack_dir    = {pack_dir}")
    log(f"  staging     = {staging}")
    log(f"  langs       = {langs}")
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
        archive = pack_dir / f"{lang_dir.name}.tar.gz"
        if not args.skip_pack:
            pack_lang_dir(lang_dir, archive)
        elif not archive.exists():
            log(f"SKIP {lang}: --skip-pack but missing {archive}")
            continue
        # copy/link into staging (Kaggle uploads the staging folder)
        dest = staging / archive.name
        if dest.exists():
            dest.unlink()
        try:
            os.link(archive, dest)  # hardlink saves disk if same filesystem
            log(f"  linked {archive.name} → staging")
        except OSError:
            log(f"  copying {archive.name} → staging (no hardlink)…")
            shutil.copy2(archive, dest)
        archives.append(dest)

    if not archives:
        raise SystemExit("Nothing to upload. Check outputs/ and --langs.")

    dataset_id = write_metadata(
        staging,
        owner=owner,
        slug_suffix=args.slug,
        title=args.title,
        is_private=not args.public,
    )

    # Kaggle CLI: create vs version
    exists = dataset_exists(kaggle_cmd, dataset_id)
    if not exists:
        log(f"Creating NEW private dataset {dataset_id} …")
        # --dir-mode tar keeps large files simpler than zip on some CLIs
        cmd = kaggle_cmd + [
            "datasets",
            "create",
            "-p",
            str(staging),
            "--dir-mode",
            "tar",
        ]
        rc = run(cmd, check=False)
        if rc != 0:
            log("create with --dir-mode tar failed; retrying without dir-mode…")
            run(kaggle_cmd + ["datasets", "create", "-p", str(staging)], check=True)
    else:
        log(f"Dataset exists → uploading NEW VERSION of {dataset_id} …")
        cmd = kaggle_cmd + [
            "datasets",
            "version",
            "-p",
            str(staging),
            "-m",
            args.message,
            "--dir-mode",
            "tar",
        ]
        rc = run(cmd, check=False)
        if rc != 0:
            log("version with --dir-mode tar failed; retrying without dir-mode…")
            run(
                kaggle_cmd
                + ["datasets", "version", "-p", str(staging), "-m", args.message],
                check=True,
            )

    log("")
    log("=" * 64)
    log("UPLOAD FINISHED")
    log(f"  Dataset: https://www.kaggle.com/datasets/{dataset_id}")
    log("  On a new machine:")
    log(f"    kaggle datasets download -d {dataset_id} -p ./ckpt_dl --unzip")
    log("    mkdir -p outputs && tar -xzf ckpt_dl/sna_mms-300m_fold0.tar.gz -C outputs/")
    log("=" * 64)


if __name__ == "__main__":
    main()
