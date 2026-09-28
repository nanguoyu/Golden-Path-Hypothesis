"""P1 of `docs/schedule_search_plan_zh.md`: write the annealing temperatures --
and, for the objectives of section 4b, the objective scales -- into the frozen
SS config from the probe summaries.

Per (model, K):  t_max = 10 x median |swap delta|,  t_min = median / 100,
both rounded to two significant figures.  The 10x ratio is the one the
table benchmark validated at K41 (t_max 0.5 over a ~0.05 dB median swap);
t_min sits two orders below the typical proposal so the geometric schedule
actually cools -- the search-internal comparison is deterministic paired
evaluation, so the calibration standard error (a between-prompt spread) does
not bound t_min.

The swap delta is measured on the *objective's* value, so each objective has
its own temperature in its own units: dB for `psnr`, LPIPS for `lpips`, and
dimensionless for `psnr_lpips_z`.  The PSNR entries keep the place P1 gave
them, `search.temperatures[<model>][<k>]`; another objective is
`search.temperatures[<objective>][<model>][<k>]`.  `search.objective_scales[
<model>][<k>]` is the mean and standard deviation each metric's probe means
had, which is what `psnr_lpips_z` standardises with.

A probe summary written before the runners recorded five metrics carries only
PSNR; the objectives that need the rest are reported as missing and their
setting has to be probed again.  Only `search.temperatures`,
`search.objective_scales` and their notes change.
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.schedule_search import METRIC_NAMES, objective_value  # noqa: E402

CONFIG = Path("resources/schedule_search/config.v1.json")
PROBES = Path("resources/schedule_search/probes")

#: The objectives backfilled beside the PSNR one, and whether they need scales.
EXTRA_OBJECTIVES = ("lpips", "psnr_lpips_z")


def two_sig(value: float) -> float:
    if value <= 0:
        raise SystemExit(f"non-positive temperature {value}")
    return float(f"{value:.2g}")


def probe_scales(probe: dict) -> dict[str, dict[str, float]] | None:
    """Mean and standard deviation of each metric's probe means."""

    stats = probe.get("metric_stats")
    if not stats:
        return None
    return {
        name: {"mean": float(stats[name]["mean"]), "std": float(stats[name]["std"])}
        for name in METRIC_NAMES
        if name in stats
    }


def objective_swap_median(
    probe: dict, objective: str, scales: dict | None
) -> float | None:
    """Median |swap delta| of one objective, from the probe's per-eval means.

    Evaluations `2t` and `2t + 1` are the two halves of swap pair `t`, and every
    objective is affine in the metrics, so the objective of an evaluation is
    read straight off that evaluation's metric means.
    """

    means = probe.get("metric_means")
    if not means:
        return None
    n_evals = len(means["psnr"])
    values = [
        objective_value(
            objective,
            {name: float(column[index]) for name, column in means.items()},
            scales=scales,
        )
        for index in range(n_evals)
    ]
    deltas = [
        abs(values[2 * t + 1] - values[2 * t]) for t in range(n_evals // 2)
    ]
    return float(np.median(np.asarray(deltas, dtype=np.float64))) if deltas else None


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    temperatures: dict[str, dict[str, dict[str, float]]] = {}
    extra: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        objective: {} for objective in EXTRA_OBJECTIVES
    }
    scales_out: dict[str, dict[str, dict]] = {}
    missing: list[str] = []
    for path in sorted(PROBES.glob("*_k*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        probe = run["probe"]
        model, k = str(run["model"]), str(run["k"])
        median = float(probe["swap_delta_db"]["median_abs"])
        temperatures.setdefault(model, {})[k] = {
            "t_max": two_sig(10.0 * median),
            "t_min": two_sig(median / 100.0),
            "probe_median_abs_swap_db": median,
            "probe_calibration_se_db": float(probe["calibration_se_db"]),
            "probe_mean_psnr_db": probe["mean_psnr_db"],
        }
        scales = probe_scales(probe)
        if scales is None:
            missing.append(f"{model} K{k}")
            continue
        scales_out.setdefault(model, {})[k] = scales
        for objective in EXTRA_OBJECTIVES:
            swap = objective_swap_median(probe, objective, scales)
            if swap is None or swap <= 0:
                missing.append(f"{model} K{k} {objective}")
                continue
            extra[objective].setdefault(model, {})[k] = {
                "t_max": two_sig(10.0 * swap),
                "t_min": two_sig(swap / 100.0),
                "probe_median_abs_swap": swap,
            }

    expected = {(m, k) for m in ("flux", "qwen") for k in ("29", "37", "41")}
    have = {(m, k) for m, ks in temperatures.items() for k in ks}
    if have != expected:
        raise SystemExit(f"probe summaries incomplete: have {sorted(have)}")
    # `temperatures` itself stays the {model: {K: row}} map; the config's copy
    # additionally holds one {model: {K: row}} map per further objective.
    written = dict(temperatures)
    for objective in EXTRA_OBJECTIVES:
        if extra[objective]:
            written[objective] = extra[objective]
    config["search"]["temperatures"] = written
    config["search"]["temperatures_note"] = (
        "P1 backfill from resources/schedule_search/probes/: t_max = 10 x median "
        "|swap delta| of the objective's value, t_min = median / 100, two "
        "significant figures; the PSNR objective is [model][K], another "
        "objective is [objective][model][K]"
    )
    if scales_out:
        config["search"]["objective_scales"] = scales_out
        config["search"]["objective_scales_note"] = (
            "mean and standard deviation of each metric's probe evaluation means, "
            "from resources/schedule_search/probes/; psnr_lpips_z standardises "
            "PSNR and negative LPIPS with them"
        )
    CONFIG.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for model, ks in temperatures.items():
        for k, row in sorted(ks.items()):
            print(
                f"{model} K{k} psnr: t_max {row['t_max']}  t_min {row['t_min']}  "
                f"(median swap {row['probe_median_abs_swap_db']:.4f} dB)"
            )
    for objective in EXTRA_OBJECTIVES:
        for model, ks in extra[objective].items():
            for k, row in sorted(ks.items()):
                print(
                    f"{model} K{k} {objective}: t_max {row['t_max']}  "
                    f"t_min {row['t_min']}  "
                    f"(median swap {row['probe_median_abs_swap']:.6f})"
                )
    if missing:
        print(
            "\nprobes without five-metric statistics, so no scales and no "
            "objective temperature: " + ", ".join(missing)
        )
        print("re-probe those settings to search them on a metric other than PSNR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
