#!/usr/bin/env python3
"""Serve the SS schedule evaluation of one (FLUX, K, objective) over HTTP.

This is `flux/schedule_search_runner.py`'s `search` mode with the search
algorithm taken out: the model stays resident on the node's GPUs through the
runner's own `PairPool` / `worker_setup` / `worker_evaluate`, the eight frozen
calibration pairs and their full-compute references are built exactly as a
search job builds them, and every schedule that arrives is scored by the same
`ObjectiveEvaluator` and written as one row of `<output_dir>/evals.jsonl` under
the algorithm name `--algorithm` gives (`agent_claude` by default,
`agent_deepseek` for the DeepSeek runner).  What proposes the schedules is a client on
the other end of the socket instead of a loop in this process.

Endpoints (JSON in, JSON out, standard library only):

    GET  /status     {"k", "objective", "n_pairs", "evals", "ready"}
    GET  /describe   the problem statement: the space, the calibration slot
                     names, the warm-start schedules and the objective scales
                     -- no prompts and no images
    POST /evaluate   {"full_steps": [...]} -> the schedule's per-pair objective
                     values and all five per-pair metrics; an already-scored
                     schedule comes back from the store with "repeat": true and
                     is not generated again
    POST /shutdown   stop the server and release the GPUs

`--fake` replaces the model with a deterministic formula over the schedule, so
the client side can be exercised without a GPU.  Nothing else changes: the same
evaluator, the same JSONL rows, the same endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import sys
import threading
import zlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flux.schedule_search_runner import (  # noqa: E402
    MODEL,
    worker_evaluate,
    worker_setup,
)
from lib.schedule_search import (  # noqa: E402
    METRIC_NAMES,
    OBJECTIVES,
    EvalSink,
    ObjectiveEvaluator,
    PairPool,
    SearchSpace,
    load_space,
    lookup_objective_scales,
    objective_units,
)

DEFAULT_ALGORITHM = "agent_claude"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_config",
        type=Path,
        default=Path("resources/schedule_search/config.v1.json"),
    )
    parser.add_argument("--k", type=int, choices=(29, 37, 41), required=True)
    parser.add_argument(
        "--algorithm",
        default=DEFAULT_ALGORITHM,
        help="the name this run's rows carry in evals.jsonl and in the "
        "client's summary.json, e.g. agent_deepseek "
        f"[default: {DEFAULT_ALGORITHM}]",
    )
    parser.add_argument("--objective", choices=OBJECTIVES, default="psnr")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="GPUs of this node to spread one evaluation's pairs over",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="score schedules with a deterministic formula instead of the "
        "model, so the client side runs without a GPU",
    )
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--revision", default="3de623fc")
    parser.add_argument(
        "--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev"
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# the GPU-free stand-in
# --------------------------------------------------------------------------


def fake_score(space: SearchSpace, n_pairs: int):
    """A deterministic scorer with the shape of the real one, for local tests.

    Front-loaded full steps and short cached runs score higher, which is enough
    structure for a client to climb; the per-pair spread and the small jitter
    make the returned vectors look like measured ones.  Same schedule in, same
    numbers out.
    """

    def score(cache_steps: Sequence[int]) -> list[dict[str, float]]:
        cached = sorted(int(step) for step in cache_steps)
        held = set(cached)
        early = sum(math.exp(-step / 8.0) for step in cached)
        longest = run = 0
        for step in range(space.num_steps):
            run = run + 1 if step in held else 0
            longest = max(longest, run)
        base = 30.0 - 2.4 * early - 0.09 * longest
        key = ",".join(str(step) for step in cached).encode()
        rows: list[dict[str, float]] = []
        for index in range(n_pairs):
            jitter = (zlib.crc32(key + str(index).encode()) % 1000) / 1000.0 - 0.5
            psnr = base + 0.9 * math.sin(index + 1.0) + 0.04 * jitter
            rows.append(
                {
                    "psnr": float(psnr),
                    "ssim": float(min(0.999, max(0.0, psnr / 40.0))),
                    "lpips": float(min(1.0, max(0.0, (34.0 - psnr) / 60.0))),
                    "image_reward": float(0.02 * (psnr - 25.0)),
                    "clip": float(0.30 + 0.002 * (psnr - 25.0)),
                }
            )
        return rows

    return score


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------


class Service:
    """One resident evaluator behind a lock, with a store of what it scored."""

    def __init__(
        self,
        *,
        space: SearchSpace,
        evaluate: ObjectiveEvaluator,
        k: int,
        objective: str,
        pair_labels: Sequence[str],
        warm_starts: Sequence[dict[str, Any]],
        scales: dict[str, Any] | None,
        algorithm: str = DEFAULT_ALGORITHM,
    ) -> None:
        self.space = space
        self.evaluate = evaluate
        self.k = int(k)
        self.objective = str(objective)
        self.algorithm = str(algorithm)
        self.pair_labels = list(pair_labels)
        self.warm_starts = list(warm_starts)
        self.scales = scales
        self.lock = threading.Lock()
        self.store: dict[tuple[int, ...], dict[str, Any]] = {}

    # -- read-only views --------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "objective": self.objective,
            "n_pairs": len(self.pair_labels),
            "evals": len(self.store),
            "ready": True,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "model": MODEL,
            "algorithm": self.algorithm,
            "space": self.space.identity_payload,
            "objective": self.objective,
            "objective_units": objective_units(self.objective),
            "objective_scales": self.scales,
            "metrics": list(METRIC_NAMES),
            "n_pairs": len(self.pair_labels),
            "calibration_slots": list(self.pair_labels),
            "warm_starts": [
                {
                    "name": row["name"],
                    "bits": row["bits"],
                    "full_steps": list(
                        self.space.full_steps(self.space.combo_of_bits(row["bits"]))
                    ),
                }
                for row in self.warm_starts
            ],
            "evals": len(self.store),
        }

    # -- the one write ----------------------------------------------------

    def score(self, full_steps: Sequence[int]) -> dict[str, Any]:
        """Validate, score if new, and answer with the stored row either way."""

        combo = self.space.combo_of_full_steps(full_steps)  # raises ValueError
        with self.lock:
            hit = self.store.get(combo)
            if hit is not None:
                return {**hit, "repeat": True}
            values = [float(v) for v in self.evaluate(combo)]
            metrics = self.evaluate.metrics_of(combo)
            row = {
                "full_steps": list(self.space.full_steps(combo)),
                "bits": self.space.bits(combo),
                "objective_values": [round(v, 6) for v in values],
                "mean_objective": float(sum(values) / len(values)),
                "metrics": {
                    name: [round(float(v), 6) for v in column]
                    for name, column in metrics.items()
                },
            }
            self.store[combo] = row
            return {**row, "repeat": False}


def make_handler(service: Service, stop: threading.Event):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter job log
            print(f"[ss-server] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            if self.path.rstrip("/") == "/status":
                self._send(200, service.status())
            elif self.path.rstrip("/") == "/describe":
                self._send(200, service.describe())
            else:
                self._send(404, {"error": f"no such endpoint: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            if path == "/shutdown":
                self._send(200, {"stopping": True, "evals": len(service.store)})
                stop.set()
                return
            if path != "/evaluate":
                self._send(404, {"error": f"no such endpoint: {self.path}"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                full_steps = body["full_steps"]
            except Exception as error:  # noqa: BLE001 - a bad request, not a crash
                self._send(400, {"error": f"body must be JSON with full_steps: {error}"})
                return
            try:
                self._send(200, service.score(full_steps))
            except ValueError as error:
                self._send(400, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 - the model side failed
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

    return Handler


def main() -> int:
    args = parse_args()
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    space = load_space(config, MODEL, args.k)
    if args.num_steps != space.num_steps:
        raise SystemExit(f"this space requires --num_steps {space.num_steps}")
    search_cfg = config["search"]
    slots = config["calibration"]["pairs"]
    pairs = [(row["prompt"], int(row["seed"])) for row in slots]
    pair_labels = [row["slot"] for row in slots]
    warm_starts = config.get("warm_starts", {}).get(MODEL, {}).get(str(args.k), [])
    scales = lookup_objective_scales(search_cfg, args.objective, MODEL, args.k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "mode": "search",
        "model": MODEL,
        "k": int(args.k),
        "algorithm": str(args.algorithm),
        "objective": args.objective,
    }
    sink = EvalSink(args.output_dir / "evals.jsonl", header=header)

    close = lambda: None  # noqa: E731 - replaced by the pool's own closer
    if args.fake:
        score = fake_score(space, len(pairs))
        print("[ss-server] --fake: no model is loaded", flush=True)
    else:
        if int(args.gpus) < 1:
            raise SystemExit("--gpus must be at least 1")
        pool = PairPool(
            world_size=int(args.gpus),
            n_pairs=len(pairs),
            setup=worker_setup,
            evaluate=worker_evaluate,
            blob={
                "args": args,
                "pairs": pairs,
                "num_steps": space.num_steps,
                "conditioning_file": None,
                "prompt_indices": (),
                "model_commit": None,
            },
            tag="ss-server",
        )
        close = pool.close
        score = pool.evaluate
        print(
            f"[ss-server] {len(pairs)} full-compute references ready across "
            f"{pool.world_size} workers",
            flush=True,
        )

    service = Service(
        space=space,
        evaluate=ObjectiveEvaluator(
            space=space,
            score=score,
            sink=sink,
            objective=args.objective,
            scales=scales,
        ),
        k=int(args.k),
        objective=args.objective,
        pair_labels=pair_labels,
        warm_starts=warm_starts,
        scales=scales,
        algorithm=str(args.algorithm),
    )

    stop = threading.Event()
    httpd = ThreadingHTTPServer((args.host, int(args.port)), make_handler(service, stop))
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    server_file = args.output_dir / "server.json"
    server_file.write_text(
        json.dumps(
            {
                "host": socket.gethostname(),
                "port": int(port),
                "pid": os.getpid(),
                "started": _now(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def on_signal(signum: int, _frame: Any) -> None:
        print(f"[ss-server] signal {signum}: shutting down", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2})
    thread.start()
    print(
        f"[ss-server] {MODEL} K{args.k} objective={args.objective} "
        f"listening on {args.host}:{port} of {socket.gethostname()} "
        f"(pid {os.getpid()}); state in {args.output_dir}",
        flush=True,
    )
    try:
        while not stop.wait(1.0):
            pass
    finally:
        httpd.shutdown()
        thread.join(timeout=30.0)
        httpd.server_close()
        close()
        sink.close()
        server_file.unlink(missing_ok=True)
    print(f"[ss-server] stopped after {len(service.store)} evaluations", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
