#!/usr/bin/env python3
"""Prepare the benchmark prompts without downloading generated images or videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
HICACHE_COMMIT = "94e11b1a5e7d2b42813ec3fc479339da6fc49207"
GENEVAL_COMMIT = "af4902f24d3ca90ebbb446dd9891a59e0f82725f"
HUNYUAN_COMMIT = "e748c73ac064728bf6bd15b1cdb8161e55a4f331"
VBENCH_COMMIT = "45e79ec14e69a2187202c675d2dbce1a71843d53"


@dataclass(frozen=True)
class Source:
    filename: str
    url: str
    sha256: str


SOURCES = {
    "drawbench_flux": Source(
        "resources/prompts/prompt.txt",
        f"https://raw.githubusercontent.com/fenglang918/HiCache/{HICACHE_COMMIT}/resources/prompts/prompt.txt",
        "4056f6f1125417b3964a1623200158d7de26ee5869d2be8661737297ddca0dd1",
    ),
    "drawbench_qwen": Source(
        "reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt",
        f"https://raw.githubusercontent.com/fenglang918/HiCache/{HICACHE_COMMIT}/models/qwen_image/prompts/DrawBench200.txt",
        "3bf3fce1c56d5fedf1af832cfb855bd4e16bacd2e4a36c123d50d04d4fd0afb7",
    ),
    "parti": Source(
        "resources/prompts/datasets/PartiPrompts.tsv",
        "https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv",
        "fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb",
    ),
    "geneval": Source(
        "resources/prompts/datasets/geneval_object_names.txt",
        f"https://raw.githubusercontent.com/djghosh13/geneval/{GENEVAL_COMMIT}/prompts/object_names.txt",
        "608f6a0e5c8ca1c7a92b818430141fe268c85b7930b50f4160f1d65893b5bacd",
    ),
    "penguin": Source(
        "reference/hunyuan_video/code/assets/PenguinVideoBenchmark.csv",
        f"https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo/{HUNYUAN_COMMIT}/assets/PenguinVideoBenchmark.csv",
        "5139a15a41498ab7f6c0844072bd92ddd09c02eebdd96192e90d97fa87f7eea0",
    ),
    "vbench": Source(
        "reference/vbench/code/vbench/VBench_full_info.json",
        f"https://raw.githubusercontent.com/Vchitect/VBench/{VBENCH_COMMIT}/vbench/VBench_full_info.json",
        "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd",
    ),
    "diffusiondb": Source(
        "resources/prompts/datasets/diffusiondb/metadata.parquet",
        "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet",
        "eecd341187bc91c07f5994ad0660d40228ea025616fd57a509bef8323677c68f",
    ),
}
GROUPS = {
    "drawbench": ("drawbench",),
    "parti": ("parti",),
    "geneval": ("geneval",),
    "video": ("video",),
    "image": ("drawbench", "parti", "geneval"),
    "light": ("drawbench", "parti", "geneval", "video"),
    "diffusiondb": ("diffusiondb",),
    "all": ("drawbench", "parti", "geneval", "video", "diffusiondb"),
}
PROMPT_OUTPUTS = {
    "parti": (
        "resources/prompts/partiprompts_full_eval1632_seed42.txt", 1632,
        "22d761025cc7be5f5eaca5988f54fe4ad1b2d66a76f55a8eea59984e810b9dec",
    ),
    "geneval": (
        "resources/prompts/geneval_seed43_n100.txt", 553,
        "55b6e3650e980c13daf30569eb666e000faf19bfc2b7b4cbe999fcdb7aa9d680",
    ),
    "diffusiondb": (
        "resources/prompts/diffusiondb_2m_clean_10000_seed42.txt", 10000,
        "0d74a9787a2aff1f8982ccec1988373774cb26cce3572193a44018bc0e9cdf28",
    ),
    "diffusiondb_calibration": (
        "resources/prompts/diffusiondb_2m_clean_calib512_seed20260729.txt", 512,
        "162541807108e9743ffe5d4cc2e84f8aa697d6714fff1c594f038347d4e012eb",
    ),
}


def sha256(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(filename: Path, expected: str) -> None:
    if not filename.is_file():
        raise FileNotFoundError(filename)
    actual = sha256(filename)
    if actual != expected:
        raise ValueError(
            f"Source or prompt order differs from the experiment: {filename}\n"
            f"Expected SHA256 {expected}, found {actual}. Existing files were not replaced."
        )


def source_file(name: str, args: argparse.Namespace) -> Path:
    spec = SOURCES[name]
    destination = ROOT / spec.filename
    supplied = args.sources.get(name)
    if supplied is not None:
        verify(supplied, spec.sha256)
        # A local multi-gigabyte parquet need not be copied into the checkout.
        if name == "diffusiondb":
            return supplied
    if destination.exists():
        verify(destination, spec.sha256)
        return destination
    if supplied is None and args.offline:
        raise FileNotFoundError(f"Missing {name}: supply --source {name}=FILE or allow downloads.")
    if supplied is None and name == "diffusiondb" and not args.allow_large_download:
        raise ValueError("DiffusionDB download requires --allow-large-download or --source diffusiondb=FILE.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".partial") as temporary:
        pending = Path(temporary.name)
        try:
            if supplied is not None:
                with supplied.open("rb") as handle:
                    shutil.copyfileobj(handle, temporary)
            else:
                print(f"Downloading {name}: {spec.url}", flush=True)
                with urlopen(spec.url, timeout=120) as response:
                    shutil.copyfileobj(response, temporary, length=1024 * 1024)
            temporary.flush()
            verify(pending, spec.sha256)
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    pending.replace(destination)
    return destination


def run_builder(filename: str, *arguments: object) -> None:
    script = ROOT / filename
    if not script.is_file():
        raise FileNotFoundError(f"Required data builder is missing: {script}")
    command = [sys.executable, str(script), *(str(arg) for arg in arguments)]
    subprocess.run(command, cwd=ROOT, check=True)


def check_prompt_output(name: str, *, existing_only: bool = False) -> None:
    relative, count, expected = PROMPT_OUTPUTS[name]
    filename = ROOT / relative
    if existing_only and not filename.exists():
        return
    verify(filename, expected)
    lines = filename.read_text(encoding="utf-8").splitlines()
    if len(lines) != count or any(not line.strip() for line in lines):
        raise ValueError(f"Expected {count} nonempty ordered prompts: {filename}")
    print(f"Verified {relative}: {count} prompts", flush=True)


def prepare_images(dataset: str, args: argparse.Namespace) -> None:
    out = ROOT / "resources/prompts"
    out.mkdir(parents=True, exist_ok=True)
    if dataset == "drawbench":
        for name in ("drawbench_flux", "drawbench_qwen"):
            filename = source_file(name, args)
            if len(filename.read_text(encoding="utf-8").splitlines()) != 200:
                raise ValueError(f"Expected 200 DrawBench prompts: {filename}")
            print(f"Verified {name}: 200 prompts", flush=True)
        return
    check_prompt_output(dataset, existing_only=True)
    source = source_file(dataset, args)
    if dataset == "parti":
        run_builder("analysis/build_parti_prompt_splits.py", "--tsv", source,
                    "--out_dir", out, "--seed", 42, "--calib_n", 0,
                    "--eval_n", 1632, "--prefix", "partiprompts_full")
    else:
        run_builder("analysis/build_geneval_prompt_file.py", "--object_names", source,
                    "--out_dir", out, "--seed", 43, "--num_prompts_per_task", 100)
    check_prompt_output(dataset)


def prepare_video(args: argparse.Namespace) -> None:
    expected_outputs = (
        ("penguin599", 599, "cad0f745795ade1f5f3641768a8cac14d9730eef826bbe58412249d358cf4ea1"),
        ("vbench944", 944, "9441dc5cfc668ecbcf3612caff1aa662bb62c3d49b482b64fa718446c452f3f4"),
    )
    for name, _, expected in expected_outputs:
        existing = ROOT / "resources/hunyuan_video/evaluation" / f"{name}.json"
        if existing.exists():
            verify(existing, expected)
    source_file("penguin", args)
    source_file("vbench", args)
    resources = ROOT / "resources/hunyuan_video"
    resources.mkdir(parents=True, exist_ok=True)
    if not (resources / "generation_protocols.v1.json").is_file():
        run_builder("analysis/hunyuan_video/build_stage_a_protocols.py")
    run_builder("analysis/hunyuan_video/build_stage_a_vbench.py")
    run_builder("analysis/hunyuan_video/build_stage_a_manifests.py")
    run_builder("analysis/hunyuan_video/build_evaluation_prompts.py")
    run_builder("analysis/hunyuan_video/build_calibration_prompts.py")
    for name, count, expected in expected_outputs:
        filename = resources / "evaluation" / f"{name}.json"
        verify(filename, expected)
        items = json.loads(filename.read_text(encoding="utf-8"))["items"]
        if len(items) != count:
            raise ValueError(f"Expected {count} entries in {filename}")
        multiline = sum("\n" in row["prompt"] for row in items)
        if name == "penguin599" and multiline != 2:
            raise ValueError("Penguin's two embedded-newline prompts were not preserved.")
        print(f"Verified {name}: {count} entries, {multiline} multiline prompts", flush=True)


def prepare_diffusiondb(args: argparse.Namespace) -> None:
    for name in ("diffusiondb", "diffusiondb_calibration"):
        check_prompt_output(name, existing_only=True)
    metadata = source_file("diffusiondb", args)
    out = ROOT / "resources/prompts"
    run_builder("analysis/build_diffusiondb_prompt_file.py", "--metadata_parquet", metadata,
                "--out_dir", out, "--subset", "2m", "--target_n", 10000, "--seed", 42)
    check_prompt_output("diffusiondb")
    evaluation = ROOT / PROMPT_OUTPUTS["diffusiondb"][0]
    run_builder("analysis/build_diffusiondb_prompt_file.py", "--metadata_parquet", metadata,
                "--out_dir", out, "--subset", "2m", "--target_n", 512,
                "--seed", 20260729, "--prefix", "diffusiondb_2m_clean_calib512_seed20260729",
                "--exclude_prompt_file", evaluation)
    check_prompt_output("diffusiondb_calibration")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(GROUPS), required=True,
                        help="image excludes DiffusionDB; light adds video metadata; all includes DiffusionDB")
    parser.add_argument("--source", action="append", default=[], metavar="NAME=FILE",
                        help=f"use a local source instead of downloading: {', '.join(SOURCES)}")
    parser.add_argument("--offline", action="store_true", help="never download missing sources")
    parser.add_argument("--allow-large-download", action="store_true",
                        help="allow downloading the multi-gigabyte DiffusionDB metadata parquet")
    args = parser.parse_args(argv)
    args.sources = {}
    for item in args.source:
        name, separator, value = item.partition("=")
        if not separator or name not in SOURCES or not value or name in args.sources:
            parser.error(f"invalid or duplicate --source: {item}")
        args.sources[name] = Path(value).expanduser().resolve()
    if "diffusiondb" in GROUPS[args.dataset]:
        available = args.sources.get("diffusiondb", ROOT / SOURCES["diffusiondb"].filename).is_file()
        if not available and (args.offline or not args.allow_large_download):
            parser.error("DiffusionDB requires a local --source diffusiondb=FILE or --allow-large-download.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for dataset in GROUPS[args.dataset]:
        if dataset == "video":
            prepare_video(args)
        elif dataset == "diffusiondb":
            prepare_diffusiondb(args)
        else:
            prepare_images(dataset, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
