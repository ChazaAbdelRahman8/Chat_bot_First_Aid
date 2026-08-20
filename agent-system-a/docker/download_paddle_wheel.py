"""Download the pinned PaddlePaddle Linux CPU wheel with resume support."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 50


def wheel_metadata(version: str) -> tuple[str, str, int]:
    request = Request(
        f"https://pypi.org/pypi/paddlepaddle/{version}/json",
        headers={"Accept-Encoding": "identity", "User-Agent": "agent-a-build/1.0"},
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    suffix = "cp312-cp312-manylinux1_x86_64.whl"
    matches = [row for row in payload["urls"] if row["filename"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one PaddlePaddle wheel ending in {suffix!r}")
    row = matches[0]
    return row["url"], row["digests"]["sha256"], int(row["size"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        current = destination.stat().st_size if destination.exists() else 0
        if current == expected_size:
            return
        if current > expected_size:
            destination.unlink()
            current = 0
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "agent-a-build/1.0",
        }
        if current:
            headers["Range"] = f"bytes={current}-"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                if current and status != 206:
                    destination.unlink(missing_ok=True)
                    continue
                mode = "ab" if current else "wb"
                with destination.open(mode) as stream:
                    while chunk := response.read(CHUNK_SIZE):
                        stream.write(chunk)
            downloaded = destination.stat().st_size
            print(f"PaddlePaddle wheel: {downloaded}/{expected_size} bytes", flush=True)
        except (TimeoutError, HTTPError, URLError, OSError) as exc:
            downloaded = destination.stat().st_size if destination.exists() else 0
            print(
                f"Download attempt {attempt} paused at {downloaded}/{expected_size}: "
                f"{type(exc).__name__}",
                flush=True,
            )
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError("PaddlePaddle wheel download did not complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url, expected_hash, expected_size = wheel_metadata(args.version)
    if args.output.exists() and args.output.stat().st_size == expected_size:
        if sha256(args.output) == expected_hash:
            print("Using cached, verified PaddlePaddle wheel", flush=True)
            return
        args.output.unlink()
    download(url, args.output, expected_size)
    actual_hash = sha256(args.output)
    if actual_hash != expected_hash:
        args.output.unlink(missing_ok=True)
        raise RuntimeError("PaddlePaddle wheel SHA256 mismatch")
    print("PaddlePaddle wheel download and SHA256 verification complete", flush=True)


if __name__ == "__main__":
    main()
