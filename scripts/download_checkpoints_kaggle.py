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


def _extract_file_name(f) -> str | None:
    if f is None:
        return None
    if isinstance(f, str):
        return f.strip() or None
    if isinstance(f, dict):
        n = f.get("name") or f.get("ref") or f.get("path")
        return str(n).strip() if n else None
    n = getattr(f, "name", None) or getattr(f, "ref", None)
    return str(n).strip() if n else None


def _extract_next_page_token(resp) -> str | None:
    if resp is None:
        return None
    if isinstance(resp, dict):
        tok = (
            resp.get("nextPageToken")
            or resp.get("next_page_token")
            or resp.get("nextPage")
        )
        return str(tok).strip() if tok else None
    tok = (
        getattr(resp, "next_page_token", None)
        or getattr(resp, "nextPageToken", None)
        or getattr(resp, "next_page", None)
    )
    return str(tok).strip() if tok else None


def list_all_files_via_api(api, dataset_id: str) -> list[str]:
    """
    Try Python KaggleApi methods. Signatures differ across kaggle-api versions.
    """
    names: list[str] = []
    owner, slug = dataset_id.split("/", 1)

    # Candidate callables + kwargs styles across kaggle-api versions
    candidates = []
    for method_name in (
        "dataset_list_files",
        "datasets_list_files",
        "dataset_list_files_with_http_info",
    ):
        meth = getattr(api, method_name, None)
        if callable(meth):
            candidates.append((method_name, meth))

    # Also try nested client used by newer kaggle packages
    for attr in ("datasets_list_files", "dataset_list_files"):
        for obj_name in ("dataset_api_client", "datasets_api", "api_client"):
            obj = getattr(api, obj_name, None)
            if obj is None:
                continue
            meth = getattr(obj, attr, None)
            if callable(meth):
                candidates.append((f"{obj_name}.{attr}", meth))

    if not candidates:
        log("API pagination note: no dataset list-files method on this KaggleApi build")
        return names

    for method_name, meth in candidates:
        try:
            page_token = None
            page = 0
            local: list[str] = []
            while True:
                page += 1
                # Try common call signatures
                resp = None
                last_err = None
                call_styles = [
                    lambda pt=page_token: meth(
                        owner, slug, page_size=200, page_token=pt
                    ),
                    lambda pt=page_token: meth(
                        dataset_id, page_size=200, page_token=pt
                    ),
                    lambda pt=page_token: meth(
                        owner_slug=owner,
                        dataset_slug=slug,
                        page_size=200,
                        page_token=pt,
                    ),
                    lambda pt=page_token: meth(dataset_id) if pt is None else None,
                    lambda pt=page_token: meth(owner, slug) if pt is None else None,
                ]
                for style in call_styles:
                    try:
                        maybe = style()
                        if maybe is not None:
                            resp = maybe
                            break
                    except TypeError as e:
                        last_err = e
                        continue
                    except Exception as e:
                        last_err = e
                        continue

                if resp is None:
                    if last_err:
                        log(f"  API {method_name} failed: {last_err}")
                    break

                # Unwrap (data, status, headers) http_info tuples
                if isinstance(resp, tuple) and resp:
                    resp = resp[0]

                file_list = (
                    getattr(resp, "files", None)
                    or getattr(resp, "dataset_files", None)
                    or (resp if isinstance(resp, list) else None)
                    or (resp.get("files") if isinstance(resp, dict) else None)
                    or []
                )
                before = len(local)
                for f in file_list:
                    n = _extract_file_name(f)
                    if n and n not in local:
                        local.append(n)
                added = len(local) - before
                page_token = _extract_next_page_token(resp)
                log(
                    f"  API {method_name} page {page}: +{added} "
                    f"(total {len(local)}) token={'yes' if page_token else 'no'}"
                )
                if not page_token or added == 0:
                    break
                if page > 200:
                    log("WARNING: stopped API pagination at 200 pages")
                    break
                time.sleep(0.15)

            if local:
                names = local
                log(f"API list via {method_name}: {len(names)} files")
                return names
        except Exception as exc:
            log(f"API pagination note ({method_name}): {exc}")
            continue

    return names


def list_all_files_via_cli(dataset_id: str) -> list[str]:
    """
    Paginate with:
      python -m kaggle datasets files DATASET -v --csv --page-size 200
      [--page-token TOKEN]

    Default page size is 20 — without pagination sna (and later files) are missed.
    """
    cmd = kaggle_cmd()
    names: list[str] = []
    page_token: str | None = None
    page = 0
    seen_tokens: set[str] = set()

    while True:
        page += 1
        c = cmd + [
            "datasets",
            "files",
            dataset_id,
            "-v",
            "--csv",
            "--page-size",
            "200",
        ]
        if page_token:
            c += ["--page-token", page_token]

        r = run_capture(c)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        combined = out + "\n" + err

        if r.returncode != 0 and not out:
            log(f"CLI files list failed (rc={r.returncode}): {err[:300]}")
            break

        new_token = None
        for line in combined.splitlines():
            # Formats seen:
            #   Next Page Token = AbCdEf...
            #   Next Page Token=AbCdEf...
            #   "nextPageToken": "..."
            low = line.lower()
            if "next page token" in low:
                if "=" in line:
                    new_token = line.split("=", 1)[-1].strip().strip('"').strip("'")
                elif ":" in line:
                    new_token = line.split(":", 1)[-1].strip().strip('"').strip("'")
            elif "nextpagetoken" in low.replace(" ", ""):
                # json-ish
                for sep in (":", "="):
                    if sep in line:
                        cand = line.split(sep, 1)[-1].strip().strip(",").strip('"').strip("'")
                        if cand and cand.lower() not in ("null", "none", ""):
                            new_token = cand
                            break

        before = len(names)
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("name") and "," in line:
                # CSV header
                continue
            if "next page token" in line.lower():
                continue
            # CSV: name,size,creationDate,...
            name = line.split(",")[0].strip().strip('"')
            if not name or name.lower() in ("name", "ref"):
                continue
            if name not in names:
                names.append(name)

        added = len(names) - before
        log(
            f"  CLI page {page}: +{added} files (total {len(names)}) "
            f"token={'yes' if new_token else 'no'}"
        )

        if not new_token or new_token in seen_tokens:
            break
        if added == 0 and page > 1:
            # token present but nothing new — stop
            break
        seen_tokens.add(new_token)
        page_token = new_token
        if page > 200:
            log("WARNING: stopped CLI pagination at 200 pages")
            break
        time.sleep(0.2)

    return names


def list_all_files(api, dataset_id: str) -> list[str]:
    """
    Paginate through every file in the dataset.
    Returns list of remote relative paths.

    IMPORTANT: default CLI page size is 20. Without --page-size 200 + page tokens,
    only the first page is seen (lin/lug only; sna missing).
    """
    names = list_all_files_via_api(api, dataset_id)

    # Always prefer CLI if API list looks incomplete (e.g. missing sna)
    has_all_langs = all(
        any(n.startswith(f"{lang}_") or f"/{lang}_" in n for n in names)
        for lang in ("lin", "sna", "lug")
    )
    if len(names) < 5 or not has_all_langs:
        if names:
            log(
                f"API listed {len(names)} files but languages incomplete "
                f"(need lin/sna/lug) — using CLI pagination"
            )
        else:
            log("Using CLI pagination for full file list…")
        cli_names = list_all_files_via_cli(dataset_id)
        # Merge, prefer CLI order if longer
        if len(cli_names) >= len(names):
            names = cli_names
        else:
            for n in cli_names:
                if n not in names:
                    names.append(n)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    names = uniq

    log(f"Collected {len(names)} remote file path(s)")
    langs_present = {
        lang
        for lang in ("lin", "sna", "lug")
        if any(n.startswith(f"{lang}_") or f"/{lang}_" in n or n.startswith(f"{lang}/") for n in names)
    }
    log(f"Languages in file list: {sorted(langs_present) or '(none detected)'}")
    if "sna" not in langs_present:
        log("WARNING: sna paths not in list — pagination may still be incomplete")

    if names:
        log("Sample paths:")
        for n in names[:8]:
            log(f"  {n}")
        if len(names) > 8:
            log(f"  ... +{len(names) - 8} more")
        # Show a sna sample if present
        sna_samples = [n for n in names if "sna" in n.lower()][:3]
        if sna_samples:
            log("Sna sample paths:")
            for n in sna_samples:
                log(f"  {n}")
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
    skipped = 0
    total = len(names)
    log(f"Per-file download of {total} files (fallback because bulk 404)…")
    for i, name in enumerate(names, 1):
        target = dest / name
        # Skip if already present with non-trivial size (re-run friendly)
        if target.is_file() and target.stat().st_size > 64:
            log(f"[{i}/{total}] SKIP (exists): {name}")
            ok += 1
            skipped += 1
            continue
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
    log(f"Per-file download: {ok}/{total} succeeded ({skipped} already present)")
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
    ap.add_argument(
        "--langs",
        default="lin,sna,lug",
        help="Comma-separated languages to restore (default: lin,sna,lug). "
        "Use --langs sna to only fetch Shona after a partial restore.",
    )
    args = ap.parse_args()

    want_langs = {x.strip().lower() for x in args.langs.split(",") if x.strip()}
    want_langs &= {"lin", "sna", "lug"}
    if not want_langs:
        raise SystemExit("--langs must include at least one of: lin, sna, lug")

    setup_kaggle_json()
    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    dataset_id = args.dataset
    dl = Path(args.download_dir).expanduser().resolve()
    outputs = Path(args.outputs).expanduser().resolve()

    log(f"Kaggle user from kaggle.json: {user}")
    log(f"Dataset: {dataset_id}")
    log(f"Languages requested: {sorted(want_langs)}")
    log("")
    log("NOTE: `kaggle datasets list -m` size=0 is often a LIE. Web UI / files list are truth.")
    log("")

    api = get_api()
    names = list_all_files(api, dataset_id)
    if args.list_only:
        return

    # Filter to requested languages (paths like sna_mms-300m_fold0/...)
    if names:
        filtered = []
        for n in names:
            for lang in want_langs:
                if (
                    n.startswith(f"{lang}_")
                    or f"/{lang}_" in n
                    or n.startswith(f"{lang}/")
                    or f"/{lang}/" in n
                ):
                    filtered.append(n)
                    break
        if filtered:
            log(f"Filtered to {len(filtered)}/{len(names)} files for langs={sorted(want_langs)}")
            names = filtered
        else:
            log(
                f"WARNING: no paths matched langs={sorted(want_langs)}; "
                "downloading full list instead"
            )

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
    found = {k: v for k, v in found.items() if k in want_langs}
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
        mark = d.exists()
        if lang not in want_langs:
            log(f"  {lang}: {mark} (not requested this run)")
        else:
            log(f"  {lang}: {mark}")
    missing = [L for L in want_langs if not (outputs / f"{L}_mms-300m_fold0").exists()]
    if missing:
        log(f"WARNING: still missing after restore: {missing}")
        log("  Re-run with --list-only to inspect file list, or re-upload from train VM.")
    else:
        log("Next: python generate_submission.py --max-blank-frac 0.05 --hf_token '…'")
    log("=" * 64)


if __name__ == "__main__":
    main()
