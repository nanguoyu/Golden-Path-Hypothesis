#!/usr/bin/env python3
"""Turn the `trajectory_bf16_floor.py --json` dumps of one video backbone into
the plan section 3.0 verdict table (docs/video_full_trajectory_plan_zh.md, P1).

    python analysis/video_trajectory/p1_floor_report.py --backbone hunyuan_video \
        --bf16 p1_floor_bfloat16.json --f32 p1_floor_float32.json --out p1_floor.md

Readability rule (results doc section 2 convention): a reading is usable where
measured / floor >= SNR_MIN; the minimum readable turn window is the smallest
w whose every window centre clears it. Nothing here is computed from latents,
only from the medians the floor probe already printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SNR_MIN = 3.0
SPACING_STEPS = (0, 1, 10, 20, 30, 40, 49)
DEV_STATES = (1, 2, 3, 5, 10, 20, 30, 40, 49)


def _win_rows(dump: dict, dtype: str) -> tuple[list[str], int | None]:
    rows, min_w = [], None
    for w, d in sorted(dump["windows"].items(), key=lambda kv: int(kv[0])):
        r = np.asarray(d["ratio"], dtype=float)
        c = d["centers"]
        ok = int((r >= SNR_MIN).sum())
        first = next((c[i] for i in range(len(r)) if r[i] >= SNR_MIN), None)
        last_bad = max((c[i] for i in range(len(r)) if r[i] < SNR_MIN), default=None)
        if ok == len(r) and min_w is None and int(w) > 1:
            min_w = int(w)
        rows.append(f"| {dtype} | {w} | {len(r)} | {np.nanmin(r):.2f} (c={c[int(np.nanargmin(r))]}) | "
                    f"{np.nanmedian(r):.1f} | {np.nanmax(r):.1f} | {ok}/{len(r)} | "
                    f"{first if first is not None else '—'} | {last_bad if last_bad is not None else '—'} | "
                    f"{np.median(d['floor_med_deg']):.3g} |")
    return rows, min_w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--bf16", type=Path, required=True)
    ap.add_argument("--f32", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t1_straightness_median", type=float, default=None,
                    help="median T1 straightness of the same paths (for the 'usable?' row)")
    args = ap.parse_args()
    dumps = {"bf16": json.loads(args.bf16.read_text()), "float32": json.loads(args.f32.read_text())}
    n = dumps["bf16"]["n_trajectories"]
    L = [f"# P1 quantization floor — {args.backbone}", "",
         f"{n} stored T3 paths (`{dumps['bf16']['latents']}`), floors measured for two "
         f"quantization dtypes; SNR = median measured / median floor over paths; readable = SNR ≥ {SNR_MIN:g}.",
         "Measured profiles come from the bf16 T3 store in both rows (the float32 rows answer "
         "'what would the instrument floor be on the float32 in-flight rows the T1 profiles were computed on').", ""]

    # turn windows
    L += ["## 1. Turn angle: single step (w=1) and multi-step windows", "",
          "| dtype | w | centres | min SNR (centre) | median SNR | max SNR | centres ≥ 3 | first readable centre | last unreadable centre | median floor (deg) |",
          "|---|---|---:|---|---:|---:|---:|---|---|---:|"]
    verdict = {}
    for dtype, dump in dumps.items():
        rows, min_w = _win_rows(dump, dtype)
        L += rows
        verdict[dtype] = min_w
    L += ["", f"Minimum readable multi-step window (all centres SNR ≥ 3): bf16 **w = {verdict['bf16'] or '> 11'}**, "
          f"float32 **w = {verdict['float32'] or '> 11'}**.", ""]
    w1 = dumps["float32"]["windows"].get("1")
    if w1:
        r = np.asarray(w1["ratio"])
        L += [f"Single-step turn (w=1) on float32 rows: {int((r >= SNR_MIN).sum())}/{len(r)} junctions readable, "
              f"min SNR {r.min():.1f} at junction {int(r.argmin())}, median {np.median(r):.0f}; "
              f"junction SNR at 0/1/2/5/10/25/40/48: " + "/".join(f"{r[j]:.0f}" for j in (0, 1, 2, 5, 10, 25, 40, 48)),
              f"→ single-step turn readable on float32 rows: **{'yes' if (r >= SNR_MIN).all() else 'partly'}**.", ""]
        r16 = np.asarray(dumps["bf16"]["windows"]["1"]["ratio"])
        L += [f"Same on bf16: {int((r16 >= SNR_MIN).sum())}/{len(r16)} junctions readable, min {r16.min():.2f}, "
              f"first readable junction {next((i for i in range(len(r16)) if r16[i] >= SNR_MIN), '—')}.", ""]

    # deviation
    L += ["## 2. Deviation from the chord (d_perp / chord), states 1..49", "",
          "| dtype | earliest state with SNR ≥ 3 | states < 3 | " + " | ".join(f"s{s}" for s in DEV_STATES) + " |",
          "|---|---|---|" + "---:|" * len(DEV_STATES)]
    for dtype, dump in dumps.items():
        m = np.asarray(dump["deviation"]["measured_med"]); f = np.asarray(dump["deviation"]["floor_med"])
        snr = m[1:50] / f[1:50]
        first = next((i + 1 for i in range(49) if snr[i] >= SNR_MIN), None)
        bad = [i + 1 for i in range(49) if snr[i] < SNR_MIN]
        L.append(f"| {dtype} | {first} | {bad or 'none'} | " + " | ".join(f"{m[s] / f[s]:.3g}" for s in DEV_STATES) + " |")
    L.append("")

    # scalars
    L += ["## 3. State norm, chord, path length, straightness, plane share", "",
          "| dtype | magnitude one-rounding rel. shift (median) | chord rel. floor | path_len rel. bias | straightness rel. bias | off-plane energy measured | off-plane floor | ratio |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dtype, dump in dumps.items():
        mg = float(np.median(np.abs(dump["magnitude_rel_med"])))
        ps = dump["plane_share"]
        mg_ = 1 - ps["measured_med"]; fl = 1 - ps["planar_floor_med"]
        L.append(f"| {dtype} | {mg:.2e} | {dump['chord_rel_med']:+.2e} | {dump['path_len_rel_med']:+.2e} | "
                 f"{dump['straightness_rel_med']:+.2e} | {mg_:.4f} | {fl:.2e} | {mg_ / fl:,.0f}× |")
    L.append("")
    if args.t1_straightness_median is not None:
        s1 = args.t1_straightness_median - 1.0
        for dtype, dump in dumps.items():
            b = dump["straightness_rel_med"]
            L.append(f"- {dtype}: T1 median (straightness − 1) = {s1:.4e} vs floor bias {b:+.2e} → "
                     f"{s1 / abs(b):.0f}× the floor → straightness {'usable' if s1 / abs(b) >= SNR_MIN else 'NOT usable'}.")
        L.append("")

    # per-step spacing bias
    L += ["## 4. Per-step displacement over-estimation (spacing rel. bias, median over paths)", "",
          "| dtype | " + " | ".join(f"step {s}" for s in SPACING_STEPS) + " | all-step median |",
          "|---|" + "---:|" * (len(SPACING_STEPS) + 1)]
    for dtype, dump in dumps.items():
        sp = np.asarray(dump["spacing_rel_med"])
        L.append(f"| {dtype} | " + " | ".join(f"{sp[s]:+.2e}" for s in SPACING_STEPS) + f" | {np.median(sp):+.2e} |")
    L += ["", "The bf16 row is the correction to subtract from any spacing / velocity / path_len read from the "
          "T3 store; the float32 row is the instrument bias on the T1 profiles.", ""]

    # verdict summary
    L += ["## 5. Verdict summary", "",
          f"- minimum readable turn window: bf16 w={verdict['bf16'] or '>11'}, float32 w={verdict['float32'] or '>11'}",
          ]
    if w1:
        L.append(f"- single-step turn on float32 rows: {'readable everywhere' if (np.asarray(w1['ratio']) >= SNR_MIN).all() else 'not readable everywhere'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
