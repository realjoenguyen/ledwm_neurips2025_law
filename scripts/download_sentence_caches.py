#!/usr/bin/env python3
"""Download and verify the sentence caches required by the environments."""

import argparse
import hashlib
from pathlib import Path


REPO_ID = "realjoenguyen/ledwm-sentence-caches"
REVISION = "c4d62f5a46cad498795dec8d422a0f2ae49b4269"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPOSITORY_ROOT / "ledwm" / "embodied" / "envs" / "data"
CHECKSUM_FILE = Path(__file__).with_name("sentence_cache_checksums.sha256")


def load_checksums():
    checksums = {}
    for line in CHECKSUM_FILE.read_text().splitlines():
        digest, relative_path = line.split(maxsplit=1)
        checksums[relative_path] = digest
    return checksums


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(destination, checksums):
    missing = []
    mismatched = []
    for relative_path, expected in checksums.items():
        path = destination / relative_path
        if not path.is_file():
            missing.append(relative_path)
        elif sha256(path) != expected:
            mismatched.append(relative_path)

    if missing or mismatched:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if mismatched:
            details.append("checksum mismatch: " + ", ".join(mismatched))
        raise SystemExit("Cache verification failed\n  " + "\n  ".join(details))

    total_bytes = sum((destination / path).stat().st_size for path in checksums)
    print(
        f"Verified {len(checksums)} cache files ({total_bytes} bytes) in "
        f"{destination}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned public LEDWM sentence-cache dataset and verify "
            "every file."
        )
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"cache directory (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing files without downloading",
    )
    args = parser.parse_args()

    destination = args.destination.expanduser().resolve()
    checksums = load_checksums()

    if not args.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is required; create environment.yml first or run "
                "`python -m pip install huggingface_hub`."
            ) from exc

        destination.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading {REPO_ID}@{REVISION} to {destination} "
            "(approximately 870 MB)..."
        )
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            token=False,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=sorted(checksums),
        )

    verify(destination, checksums)


if __name__ == "__main__":
    main()
