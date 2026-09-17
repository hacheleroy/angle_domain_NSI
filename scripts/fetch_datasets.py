#!/usr/bin/env python3
"""Download and safely extract the public PICMUS and MBTrace inputs.

The files remain subject to their source licenses and are excluded from Git.
Interrupted HTTP downloads resume through a ``.part`` file when the server
supports byte ranges.  Known extracted-file SHA-256 values are verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PICMUS_IN_VIVO_URL = (
    "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/sites/"
    "www.creatis.insa-lyon.fr.Challenge.IEEE_IUS_2016/files/in_vivo.zip"
)
PICMUS_ARCHIVE_URL = (
    "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/sites/"
    "www.creatis.insa-lyon.fr.Challenge.IEEE_IUS_2016/files/archive_to_download.zip"
)
MBTRACE_SHARE_URL = "https://uofi.box.com/s/8w3igtnilb7uilr2jxh07jln6j6f3opx"

KNOWN_SHA256 = {
    "PICMUS/in_vivo/carotid_cross/carotid_cross_expe_dataset_rf.hdf5": (
        "c1c0cc8b3c2efb4d3374b53c63a69d1251dab19616e100f0c73b644e798c0e6d"
    ),
    "PICMUS/in_vivo/carotid_long/carotid_long_expe_dataset_rf.hdf5": (
        "789a86d88ff28e2d4f7c8cd5c764cfe9c1071919f0fb0734316ce1d0b20c647e"
    ),
    "PICMUS/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5": (
        "c3964c54cc9e3f99f89e3942b7797d0f8d03d835a9d8dd9c1fe59f745685a83b"
    ),
    "PICMUS/resolution_distorsion/resolution_distorsion_expe_phantom.hdf5": (
        "6633ad03bcbdcf441f0a7bf71d93839420c7bee338b74aaf194b6e1a38b63f67"
    ),
    "PICMUS/resolution_distorsion/resolution_distorsion_expe_scan.hdf5": (
        "335e781316d6df4c941e011528f97348a39466ceb8435da8b1a270393ba690f5"
    ),
    "Open-NSI/Basic/data/MBTrace.mat": (
        "c70e976607879cf85d7e62448c2f9169e09fcd757603cac01a9b89da95bef929"
    ),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fetch public NSI datasets")
    parser.add_argument(
        "--data-root", type=Path, default=root / "data",
        help="Destination root (default: repository data directory).",
    )
    parser.add_argument(
        "--download-dir", type=Path, default=None,
        help="Archive cache (default: DATA_ROOT/downloads).",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "picmus-in-vivo", "picmus-resolution", "mbtrace"),
        default="all",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        print(f"Using existing archive: {destination}")
        return destination
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() and not force else 0
    if force and partial.exists():
        partial.unlink()
    headers = {"User-Agent": "angle-domain-NSI-reproducibility/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        total_header = response.headers.get("Content-Length")
        total = offset + int(total_header) if total_header else None
        downloaded = offset
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\rDownloading {destination.name}: "
                        f"{downloaded / 2**20:.1f}/{total / 2**20:.1f} MiB",
                        end="",
                        flush=True,
                    )
    print()
    partial.replace(destination)
    return destination


def box_mbtrace_download_url() -> str:
    request = urllib.request.Request(
        MBTRACE_SHARE_URL,
        headers={"User-Agent": "angle-domain-NSI-reproducibility/1.0"},
    )
    with urllib.request.urlopen(request) as response:
        page = response.read().decode("utf-8", errors="replace")
    match = re.search(
        r'"typedID":"f_(\d+)".{0,2000}?"name":"MBTrace\.mat"',
        page,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            "The Open-NSI Box page no longer exposes MBTrace.mat as expected. "
            f"Open {MBTRACE_SHARE_URL} and download the file manually."
        )
    file_id = match.group(1)
    return (
        "https://uofi.app.box.com/index.php?rm=box_download_shared_file"
        "&shared_name=8w3igtnilb7uilr2jxh07jln6j6f3opx"
        f"&file_id=f_{file_id}"
    )


def safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member path: {name}")
    return path


def extract_selected(
    archive: Path,
    destination: Path,
    suffix_to_relative_destination: dict[str, str],
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(archive) as bundle:
        members = {member.filename: member for member in bundle.infolist()}
        for suffix, relative in suffix_to_relative_destination.items():
            matches = [
                member for name, member in members.items()
                if safe_member_name(name).as_posix().endswith(suffix)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one ZIP member ending {suffix!r}, found {len(matches)}."
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(matches[0]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            written.append(target)
    return written


def verify_known(data_root: Path, relative_paths: list[str]) -> list[dict[str, str]]:
    rows = []
    for relative in relative_paths:
        path = data_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        expected = KNOWN_SHA256[relative]
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )
        rows.append({"path": relative, "sha256": actual})
        print(f"Verified {relative}: {actual}")
    return rows


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    download_dir = (
        args.download_dir.expanduser().resolve()
        if args.download_dir is not None
        else data_root / "downloads"
    )
    download_dir.mkdir(parents=True, exist_ok=True)
    requested = (
        {"picmus-in-vivo", "picmus-resolution", "mbtrace"}
        if args.dataset == "all"
        else {args.dataset}
    )
    verified: list[dict[str, str]] = []
    archives: list[Path] = []

    if "picmus-in-vivo" in requested:
        archive = download(
            PICMUS_IN_VIVO_URL, download_dir / "in_vivo.zip", force=args.force
        )
        archives.append(archive)
        paths = {
            "in_vivo/carotid_cross/carotid_cross_expe_dataset_rf.hdf5": (
                "carotid_cross/carotid_cross_expe_dataset_rf.hdf5"
            ),
            "in_vivo/carotid_long/carotid_long_expe_dataset_rf.hdf5": (
                "carotid_long/carotid_long_expe_dataset_rf.hdf5"
            ),
        }
        extract_selected(archive, data_root / "PICMUS" / "in_vivo", paths)
        verified.extend(verify_known(data_root, [
            "PICMUS/in_vivo/carotid_cross/carotid_cross_expe_dataset_rf.hdf5",
            "PICMUS/in_vivo/carotid_long/carotid_long_expe_dataset_rf.hdf5",
        ]))

    if "picmus-resolution" in requested:
        archive = download(
            PICMUS_ARCHIVE_URL,
            download_dir / "archive_to_download.zip",
            force=args.force,
        )
        archives.append(archive)
        names = [
            "resolution_distorsion_expe_dataset_rf.hdf5",
            "resolution_distorsion_expe_phantom.hdf5",
            "resolution_distorsion_expe_scan.hdf5",
        ]
        extract_selected(
            archive,
            data_root / "PICMUS" / "resolution_distorsion",
            {
                f"database/experiments/resolution_distorsion/{name}": name
                for name in names
            },
        )
        verified.extend(verify_known(data_root, [
            f"PICMUS/resolution_distorsion/{name}" for name in names
        ]))

    if "mbtrace" in requested:
        destination = data_root / "Open-NSI" / "Basic" / "data" / "MBTrace.mat"
        if not destination.is_file() or args.force:
            download(box_mbtrace_download_url(), destination, force=args.force)
        verified.extend(verify_known(data_root, ["Open-NSI/Basic/data/MBTrace.mat"]))

    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": {
            "picmus_in_vivo": PICMUS_IN_VIVO_URL,
            "picmus_archive": PICMUS_ARCHIVE_URL,
            "mbtrace_share": MBTRACE_SHARE_URL,
        },
        "verified_files": verified,
    }
    manifest_path = data_root / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    if not args.keep_archives:
        for archive in archives:
            archive.unlink(missing_ok=True)
    print(f"Dataset acquisition complete: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; the partial download can be resumed.", file=sys.stderr)
        raise SystemExit(130)
