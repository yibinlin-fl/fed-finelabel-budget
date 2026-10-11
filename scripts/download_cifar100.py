"""Download and verify official CIFAR-100; stdlib only.

Run explicitly; P0 tests/preflight never download data.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
OFFICIAL_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
ACCEPTED_FILES = {
    "cifar-100-python/train", "cifar-100-python/test", "cifar-100-python/meta",
    "cifar-100-python/file.txt",
}


def download_archive(archive):
    """Stage incomplete downloads separately; do not poison the final archive."""
    temporary = archive.with_suffix(".download")
    checksum = hashlib.md5()
    with urllib.request.urlopen(URL, timeout=30) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            checksum.update(block)
    if checksum.hexdigest() != OFFICIAL_MD5:
        raise RuntimeError("Official dataset MD5 mismatch; staged file retained for diagnosis")
    os.replace(temporary, archive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/cifar100")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-100-python.tar.gz"
    if not archive.is_file():
        print(f"Downloading official CIFAR-100 into {archive} ...")
        download_archive(archive)
    checksum = hashlib.md5()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != OFFICIAL_MD5:
        raise RuntimeError("Official dataset MD5 mismatch. Remove damaged archive and retry.")
    dest = root / "cifar-100-python"
    dest.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = str(PurePosixPath(member.name))
            if member.isdir():
                continue
            if name not in ACCEPTED_FILES:
                continue  # harmless auxiliary files are not extracted
            if not member.isfile() or member.size > 200_000_000:
                raise RuntimeError(f"unsafe archive entry: {member.name}")
            handle = tar.extractfile(member)
            if handle is None:
                raise RuntimeError(f"unable to extract {member.name}")
            target = dest / PurePosixPath(name).name
            with handle, target.open("wb") as out:
                shutil.copyfileobj(handle, out)
    print("CIFAR-100 verified and extracted. Next: fed-finelabel data-check --root", root)

if __name__ == "__main__":
    main()
