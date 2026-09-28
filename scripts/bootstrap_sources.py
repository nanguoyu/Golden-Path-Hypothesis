#!/usr/bin/env python3
"""Fetch pinned public source code. This does not download model weights."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "hunyuan_video": ("https://github.com/Tencent-Hunyuan/HunyuanVideo.git", "e748c73ac064728bf6bd15b1cdb8161e55a4f331"),
    "taylorseer": ("https://github.com/Shenyi-Z/TaylorSeer.git", "704ee98c74f7f04da443daa3c0aa2cc7803d86e3"),
    "vbench": ("https://github.com/Vchitect/VBench.git", "45e79ec14e69a2187202c675d2dbce1a71843d53"),
}


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def fetch(name):
    url, commit = SOURCES[name]
    destination = ROOT / "reference" / name / "code"
    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(f"Refusing to overwrite {destination}")
        if run("git", "status", "--porcelain", cwd=destination):
            raise RuntimeError(f"Existing checkout has changes: {destination}")
        actual = run("git", "rev-parse", "HEAD", cwd=destination)
        if actual != commit:
            raise RuntimeError(f"Existing checkout is at {actual}, expected {commit}: {destination}")
        print(f"Already present: {name} at {commit}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run("git", "init", str(destination))
    run("git", "remote", "add", "origin", url, cwd=destination)
    run("git", "fetch", "--depth", "1", "origin", commit, cwd=destination)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=destination)
    if run("git", "rev-parse", "HEAD", cwd=destination) != commit:
        raise RuntimeError(f"Unexpected revision for {name}")
    print(f"Fetched {name} at {commit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", choices=[*SOURCES, "all"])
    args = parser.parse_args()
    names = list(SOURCES) if "all" in args.sources else list(dict.fromkeys(args.sources))
    for name in names:
        fetch(name)


if __name__ == "__main__":
    main()
