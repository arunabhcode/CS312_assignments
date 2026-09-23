import argparse
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from data import (
    DEFAULT_DATA_REPO_ID,
    DEFAULT_DATA_REVISION,
    MANIFEST_FILE,
    default_dataset_root,
)


CHUNK_BYTES = 8 * 1024 * 1024


def hf_file_url(repo_id, revision, filename):
    quoted_file = urllib.parse.quote(filename)
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted_file}"


def _format_bytes(num_bytes):
    num_bytes = int(num_bytes)
    for scale, suffix in [
        (1024**3, "GiB"),
        (1024**2, "MiB"),
        (1024, "KiB"),
    ]:
        if num_bytes >= scale:
            return f"{num_bytes / scale:.2f} {suffix}"
    return f"{num_bytes} bytes"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_file_is_valid(path, expected_size=None, expected_sha256=None):
    path = Path(path)
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    if expected_sha256 is not None and _file_sha256(path) != expected_sha256:
        return False
    return True


def _download_file(url, path, *, expected_size=None, expected_sha256=None, force=False):
    path = Path(path)
    if not force and _existing_file_is_valid(path, expected_size, expected_sha256):
        print(f"Already present: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Downloading {url} -> {path}")
    digest = hashlib.sha256()
    bytes_written = 0
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
                if expected_size and bytes_written % (512 * 1024 * 1024) < CHUNK_BYTES:
                    print(
                        f"  {_format_bytes(bytes_written)} / "
                        f"{_format_bytes(expected_size)}"
                    )
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc

    if expected_size is not None and bytes_written != expected_size:
        raise RuntimeError(
            f"Downloaded {bytes_written} bytes for {path}, expected {expected_size}."
        )
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"sha256 mismatch for {path}: got {actual_sha256}, "
            f"expected {expected_sha256}."
        )

    os.replace(tmp_path, path)


def download_default_data(
    output_dir=None,
    *,
    repo_id=DEFAULT_DATA_REPO_ID,
    revision=DEFAULT_DATA_REVISION,
    force=False,
):
    output_dir = Path(output_dir) if output_dir is not None else default_dataset_root()
    manifest_path = output_dir / MANIFEST_FILE
    manifest_url = hf_file_url(repo_id, revision, MANIFEST_FILE)
    _download_file(manifest_url, manifest_path, force=force)

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", [])
    if not files:
        raise ValueError(f"{manifest_path} does not list any files.")

    for file_info in files:
        filename = file_info["path"]
        _download_file(
            hf_file_url(repo_id, revision, filename),
            output_dir / filename,
            expected_size=file_info.get("size_bytes"),
            expected_sha256=file_info.get("sha256"),
            force=force,
        )

    print(f"Data ready at {output_dir}")
    return {
        "data_dir": str(output_dir),
        "repo_id": repo_id,
        "revision": revision,
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description="Download Deep Learning Alchemy token data.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repo-id", default=DEFAULT_DATA_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_default_data(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        force=args.force,
    )


if __name__ == "__main__":
    main()
