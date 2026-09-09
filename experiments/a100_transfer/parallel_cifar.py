"""Retrieve the exact official CIFAR100 binary archive using bounded byte ranges.

Run in an already reserved CPU container. This script creates no Modal function,
app, or paid job, and never touches another downloader's temporary file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.request
import uuid

OFFICIAL_URL = "https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz"
MIRROR_URL = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-binary.tar.gz"
ARCHIVE_BYTES = 168_513_733
OFFICIAL_MD5 = "03b5dce01913d631647c71ecec9e9cb8"
ENVIRONMENT = "gradient-dissent-a100"
VOLUME = "gradient-dissent-a100-20260909"


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(directory, *, workers=8, deadline_seconds=600):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "cifar-100-binary.tar.gz"
    if archive.exists():
        if archive.stat().st_size != ARCHIVE_BYTES or file_hash(archive, "md5") != OFFICIAL_MD5:
            raise ValueError("An existing canonical archive failed the official MD5 gate")
        return {"cached": True, "archive_md5": OFFICIAL_MD5,
                "archive_sha256": file_hash(archive), "bytes": ARCHIVE_BYTES}
    if workers != 8 or not 1 <= deadline_seconds <= 900:
        raise ValueError("This bounded helper uses exactly 8 workers and at most a 900s deadline")
    started = time.monotonic()
    deadline = started + deadline_seconds
    workdir = directory / (".parallel-cifar-" + uuid.uuid4().hex)
    workdir.mkdir()
    progress = [0] * workers
    progress_lock = threading.Lock()
    ranges = [(i * ARCHIVE_BYTES // workers, (i + 1) * ARCHIVE_BYTES // workers - 1)
              for i in range(workers)]

    def fetch(index):
        begin, end = ranges[index]
        expected = end - begin + 1
        path = workdir / f"range-{index:02d}.part"
        attempts = []
        # Two tries from the faster mirror, then an official-host fallback.
        # Exactly three maximum attempts per range, not three per source.
        for attempt in range(3):
            source = MIRROR_URL if attempt < 2 else OFFICIAL_URL
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Overall CIFAR transfer deadline reached")
            with progress_lock:
                progress[index] = 0
            attempt_started = time.monotonic()
            req = urllib.request.Request(source, headers={
                "Range": f"bytes={begin}-{end}", "Accept-Encoding": "identity",
            })
            try:
                with urllib.request.urlopen(req, timeout=min(90, remaining)) as response:
                    expected_header = f"bytes {begin}-{end}/{ARCHIVE_BYTES}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_header:
                        raise ValueError("Server failed exact 206 Content-Range validation")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) != expected:
                        raise ValueError("Server returned an incorrect range Content-Length")
                    received = 0
                    with path.open("wb") as target:
                        while True:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Overall CIFAR transfer deadline reached")
                            block = response.read(min(1 << 20, expected - received + 1))
                            if not block:
                                break
                            target.write(block)
                            received += len(block)
                            if received > expected:
                                raise ValueError("Response body exceeded the requested byte range")
                            with progress_lock:
                                progress[index] = received
                    if received != expected:
                        raise ValueError(f"Short range: got {received}, expected {expected}")
                attempts.append({"url": source, "success": True,
                                 "seconds": time.monotonic() - attempt_started})
                return {"index": index, "begin": begin, "end": end, "bytes": received,
                        "sha256": file_hash(path), "attempts": attempts}
            except Exception as error:
                attempts.append({"url": source, "success": False,
                                 "error": f"{type(error).__name__}: {error}",
                                 "seconds": time.monotonic() - attempt_started})
                print(f"Range {index} attempt {attempt+1}/3 failed: {error}", flush=True)
                if attempt == 2 or time.monotonic() >= deadline:
                    raise RuntimeError(f"Range {index} exhausted its bounded attempts: {attempts}") from error
                time.sleep(attempt + 1)

    print(f"Downloading {ARCHIVE_BYTES:,} bytes in 8 verified ranges; official MD5 is mandatory", flush=True)
    # Successful reads check the overall deadline; any individual blocked read is
    # bounded by <=90s. Thus shutdown can exceed the overall deadline by <=90s.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(fetch, i) for i in range(workers)}
        completed = []
        while pending:
            done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                completed.append(future.result())
            with progress_lock:
                received = sum(progress)
            print(f"CIFAR ranges: {received:,}/{ARCHIVE_BYTES:,} bytes; "
                  f"{len(completed)}/8 complete; {time.monotonic()-started:.1f}s", flush=True)
    assembled = workdir / "assembled.tar.gz"
    with assembled.open("wb") as target:
        for index in range(workers):
            with (workdir / f"range-{index:02d}.part").open("rb") as source:
                shutil.copyfileobj(source, target, length=1 << 20)
        target.flush()
        os.fsync(target.fileno())
    if assembled.stat().st_size != ARCHIVE_BYTES or file_hash(assembled, "md5") != OFFICIAL_MD5:
        raise ValueError("Assembled archive FAILED official byte-count/MD5 gate; canonical path untouched")
    digest = file_hash(assembled)
    assembled.replace(archive)
    result = {"cached": False, "official_source": OFFICIAL_URL,
              "mirror_source": MIRROR_URL, "bytes": ARCHIVE_BYTES,
              "archive_md5": OFFICIAL_MD5, "archive_sha256": digest,
              "download_seconds": time.monotonic()-started,
              "range_records": sorted(completed, key=lambda item: item["index"])}
    (directory / "parallel-download-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.rmtree(workdir)  # Only this invocation's private, now-verified fragments.
    print(f"Official MD5 passed. Atomic canonical archive ready in {result['download_seconds']:.1f}s", flush=True)
    return result


def prepare_and_commit(root, *, commit=False):
    started = time.monotonic()
    root = Path(root)
    retrieval = download_archive(root / "data/vision")
    # The calling process should put the mounted experiment source on sys.path.
    from language import sha256_file
    from vision import load_cifar100
    language_root = root / "data/language"
    language = json.loads((language_root / "manifest.json").read_text())
    for split in language["splits"].values():
        for name in ("tokens", "document_offsets"):
            artifact = split[name]
            if sha256_file(language_root / artifact["file"]) != artifact["sha256"]:
                raise ValueError("Completed language artifact failed checksum verification")
    vision = load_cifar100(root / "data/vision", download=False)["metadata"]
    vision["retrieval"] = retrieval
    seconds = time.monotonic() - started
    manifest = {
        "language": language, "vision": vision, "seconds": seconds,
        "seconds_scope": "This accelerated CIFAR retrieval, verification, and merged metadata generation only; excludes earlier language preparation, redundant original download, and subsequent volume commit.",
        "preparation_method": "Existing reserved CPU container, 8 byte ranges, exact official archive MD5 gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = root / ("dataset-manifest.json.parallel-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(root / "dataset-manifest.json")
    if commit:
        import modal
        print(f"Committing already-mounted volume {VOLUME} in {ENVIRONMENT}", flush=True)
        volume = modal.Volume.from_name(VOLUME, environment_name=ENVIRONMENT, create_if_missing=False)
        volume.commit()
        print("COMMITTED: exact CIFAR archive and merged dataset-manifest.json are visible to new containers", flush=True)
    return {"committed": commit, "seconds_before_commit": seconds,
            "archive_md5": vision["archive_md5"], "archive_sha256": vision["archive_sha256"],
            "manifest": str(root / "dataset-manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/work"))
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_and_commit(args.root, commit=args.commit), indent=2))
