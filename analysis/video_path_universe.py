#!/usr/bin/env python3
"""Path universe of the video baseline matrices, all nine methods on equal footing.

Video counterpart of section 5 of the image-side full-results report. Every
method is one *entity* in path space: the four dynamic gates are distributions
over observed 50-bit paths (rebuilt from every generation's decisions), the
three fixed families are point masses (BudCache table, MeanCache table, and the
shared uniform table that TaylorSeer / HiCache / L2P all run on). Per partition
(backbone × K), and per dataset where the statistic is per-dataset:

* path-distribution statistics per gate: generations, unique paths, top-1/2/3/10
  share, entropy (bits), cloud diameter (max pairwise Hamming among observed
  paths), n90 (smallest number of paths covering >= 90 % of generations);
* universe size: distinct paths over the four gates, pairwise Hamming median /
  max, the exact-K Hamming cap min(2K, 2(50-K));
* every entity's top-1 path (bitstring, share, cached steps, warm-up length);
* three distance layers between all 7 entities: top-1 Hamming, top-3 nearest
  member Hamming, and the transport distance W between full distributions with
  Hamming ground cost (exact, via linear programming);
* exact overlaps: for every entity pair, the number of shared paths and the
  shared probability mass;
* stability: top-1 across seed streams, across datasets, across backbones,
  across budgets (nesting of cached-step sets);
* same-schedule-different-payload and same-payload-different-schedule quality
  contrasts from the 162-cell tables.

The second half reproduces, on the video entities, the geometry the image-side
report builds in its section 5.3 / 5.4 / 5.6, using the same definitions as
`analysis/full_results_local/distances.py` and `path_universe.py`:

* alternative ground metrics for the transport distance -- full-anchor Jaccard
  (1 - |F(u) & F(v)| / |F(u) | F(v)|, F = the set of full-compute steps) and
  anchor transport (1-D Wasserstein-1 between the uniform distributions over
  the normalised full-step positions t / 49) -- plus exact-support total
  variation, the step-marginal L1, and the Spearman rank correlation of each
  against the Hamming reading;
* the budget / relocation split of the expected cross Hamming distance
  (d_H = |K_u - K_v| + 2 * relocation);
* the exact-overlap census per dataset and per pair class;
* basins: the pair-distance summary, the connectivity sweep over the pooled
  path universe (merge radius r* against the direct minimum distance), the
  mass-weighted components at radius 4 / 6 / 8 with the popular-versus-best
  quality read, and the per-dataset stability of every cross-entity co-basin;
* the path-distance / quality-gap linkage over all 36 method pairs and the
  same-path-different-payload table over the three shared-schedule pairs.

    python analysis/video_path_universe.py --out resources/video_full_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
T_ALL = ("hunyuan_video", "wan21")
DS = ("penguin599", "vbench944")
NPROMPT = {"penguin599": 599, "vbench944": 944}
KS = ("K29", "K37", "K41")
GATES = ("seacache", "teacache", "sencache", "dicache")
FIXED = ("shared", "budcache", "meancache")
ENT = GATES + FIXED
CONFIG = {"hunyuan_video": "resources/hunyuan_video/baseline_matrix_config.v2.json",
          "wan21": "resources/wan21/baseline_matrix_config.v1.json"}


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def cache_steps(p: str) -> list[int]:
    return [i for i, c in enumerate(p) if c == "1"]


def warmup(p: str) -> int:
    """Number of leading full steps before the first cached step."""
    return p.index("1") if "1" in p else 50


def entropy_bits(mass: list[float]) -> float:
    return -sum(m * math.log2(m) for m in mass if m > 0)


def n90(mass_desc: list[float]) -> int:
    acc = 0.0
    for i, m in enumerate(mass_desc, 1):
        acc += m
        if acc >= 0.9 - 1e-12:
            return i
    return len(mass_desc)


def wasserstein_hamming(pa: dict[str, float], pb: dict[str, float]) -> float:
    """Exact earth mover's distance with Hamming ground cost (LP)."""
    A, Bp = list(pa), list(pb)
    if len(A) == 1 and len(Bp) == 1:
        return float(hamming(A[0], Bp[0]))
    C = np.array([[hamming(a, b) for b in Bp] for a in A], dtype=float)
    n, m = C.shape
    c = C.reshape(-1)
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1
    for j in range(m):
        A_eq[n + j, j::m] = 1
    b_eq = np.concatenate([np.array([pa[a] for a in A]), np.array([pb[b] for b in Bp])])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return float(res.fun)


def full_steps(p: str) -> list[int]:
    return [i for i, c in enumerate(p) if c == "0"]


def d_jaccard(a: str, b: str) -> float:
    """Full-anchor Jaccard: 1 - |F(a) & F(b)| / |F(a) | F(b)|."""
    fa, fb = set(full_steps(a)), set(full_steps(b))
    union = fa | fb
    return 0.0 if not union else 1.0 - len(fa & fb) / len(union)


def w1_1d(xs, px, ys, py) -> float:
    """1-D Wasserstein-1 between two weighted point clouds on the line."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    px, py = np.asarray(px, float), np.asarray(py, float)
    if xs.size == 0 or ys.size == 0:
        return 0.0
    pts = np.unique(np.concatenate([xs, ys]))
    if pts.size < 2:
        return 0.0
    fx = np.array([px[xs <= p].sum() for p in pts])
    fy = np.array([py[ys <= p].sum() for p in pts])
    return float(np.abs(fx[:-1] - fy[:-1]) @ np.diff(pts))


def d_anchor(a: str, b: str) -> float:
    """Anchor transport: W1 between the uniform laws on the normalised
    full-step positions t / 49 of the two paths."""
    fa = np.array(full_steps(a), float) / 49.0
    fb = np.array(full_steps(b), float) / 49.0
    if fa.size == 0 or fb.size == 0:
        return 0.0
    return w1_1d(fa, np.full(fa.size, 1 / fa.size), fb, np.full(fb.size, 1 / fb.size))


GROUND = {"hamming": lambda a, b: float(hamming(a, b)), "jaccard": d_jaccard, "anchor": d_anchor}


def wasserstein(pa: dict[str, float], pb: dict[str, float], ground="hamming") -> float:
    """Exact optimal transport cost between two finite path distributions."""
    g = GROUND[ground]
    A, Bp = list(pa), list(pb)
    if len(A) == 1 and len(Bp) == 1:
        return float(g(A[0], Bp[0]))
    C = np.array([[g(a, b) for b in Bp] for a in A], dtype=float)
    n, m = C.shape
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1
    for j in range(m):
        A_eq[n + j, j::m] = 1
    b_eq = np.concatenate([np.array([pa[a] for a in A]), np.array([pb[b] for b in Bp])])
    res = linprog(C.reshape(-1), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return float(res.fun)


def total_variation(pa: dict[str, float], pb: dict[str, float]) -> float:
    keys = set(pa) | set(pb)
    return 0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys)


def step_marginal(p: dict[str, float]) -> np.ndarray:
    m = np.zeros(50)
    for path, mass in p.items():
        m += mass * np.array([int(c) for c in path], dtype=float)
    return m


def marginal_l1(pa, pb) -> float:
    return float(np.abs(step_marginal(pa) - step_marginal(pb)).sum())


def expected_budget_relocate(pa, pb) -> tuple[float, float]:
    """Expected |K_a - K_b| and expected relocation under the product measure,
    so that E[d_H] = E[budget] + 2 E[relocation]."""
    eb = er = 0.0
    for u, mu in pa.items():
        for v, mv in pb.items():
            db = abs(u.count("1") - v.count("1"))
            eb += mu * mv * db
            er += mu * mv * (hamming(u, v) - db) / 2
    return eb, er


#: step 0 and the last step are full in every observed top-1 path, so only the
#: middle 48 positions are free; the image-side report uses the same window.
N_FREE_POSITIONS = 48
#: a step whose nine-method mean cache probability is at least this is a
#: consensus-cache step; at most 1 - it, a consensus-full step (image side §5.1)
CONSENSUS_CACHE_MIN = 0.9
CONSENSUS_FULL_MAX = 0.1
N_PEAKS = 5


def random_overlap_baseline(ka: int, kb: int, n_free: int = N_FREE_POSITIONS) -> float:
    """Expected number of jointly cached steps for two independent uniformly
    random schedules caching ka and kb of the n_free free positions."""
    return ka * kb / n_free


def longest_run(p: str) -> int:
    """Longest run of consecutive cached steps in one path."""
    best = cur = 0
    for c in p:
        cur = cur + 1 if c == "1" else 0
        best = max(best, cur)
    return best


def expected_longest_run(dist: dict[str, float]) -> float:
    return sum(m * longest_run(p) for p, m in sorted(dist.items()))


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the 50 steps on which two step-marginal curves sit on the
    same side of their own mean cache rate.  Shape only, absolute rates never
    meet (image-side definition)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.sign(a - a.mean()) == np.sign(b - b.mean())))


def percentile_of_score(values, x: float) -> float:
    """100 * (#{v < x} + 0.5 #{v == x}) / n, ties at half weight."""
    v = np.asarray(list(values), dtype=float)
    return float(100.0 * (float((v < x).sum()) + 0.5 * float((v == x).sum())) / v.size)


def classical_mds(D: np.ndarray) -> np.ndarray:
    """Two-dimensional classical scaling, the layout the path-universe figures
    draw (deterministic; no random start)."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    return v[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def layout_fidelity(paths: list[str]) -> float:
    """Spearman correlation between the true pairwise Hamming distances of a
    path set and the distances of its two-dimensional layout."""
    A = np.array([[int(c) for c in p] for p in paths])
    D = (A[:, None, :] != A[None, :, :]).sum(-1).astype(float)
    X = classical_mds(D)
    planar = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(paths), 1)
    return float(spearmanr(D[iu], planar[iu]).statistic)


def components(paths: list[str], r: int) -> list[list[int]]:
    """Connected components of the graph "edge iff Hamming <= r"."""
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(len(paths)), 2):
        if hamming(paths[i], paths[j]) <= r:
            parent[find(i)] = find(j)
    out: dict[int, list[int]] = {}
    for i in range(len(paths)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def spearman(xs, ys, tol: float = 1e-9) -> float | None:
    """Spearman rank correlation; None when either side is constant to within
    `tol` (a reading with no spread cannot rank anything)."""
    for v in (xs, ys):
        if max(v) - min(v) <= tol:
            return None
    return float(spearmanr(xs, ys).statistic)


def fixed_paths(T):
    cfg = json.loads((REPO / CONFIG[T]).read_text())["schedule_tables"]
    out = {}
    for K in KS:
        out[K] = {}
        for fam, keys in (("shared", (f"triplet_{K}", f"shared_{K}")), ("budcache", (f"budcache_{K}",)), ("meancache", (f"meancache_{K}",))):
            for k in keys:
                if k in cfg:
                    steps = set(int(s) for s in cfg[k]["cache_steps"])
                    out[K][fam] = "".join("1" if i in steps else "0" for i in range(50))
    return out


def gate_dists(T):
    """{(gate, ds, K): {path: count}} and per-seed {(gate, ds, K, seed): {path: count}}"""
    pooled, per_seed = defaultdict(Counter), defaultdict(Counter)
    for r in csv.DictReader(open(REPO / f"resources/video_native_gate_paths/{T}/per_seed_path_counts.tsv"), delimiter="\t"):
        per_seed[(r["method"], r["dataset"], r["budget"], int(r["seed"]))][r["schedule"]] += int(r["count"])
        pooled[(r["method"], r["dataset"], r["budget"])][r["schedule"]] += int(r["count"])
    return pooled, per_seed


def quality(T):
    rows = list(csv.DictReader(open(REPO / f"docs/figures/baseline_matrix/{T}/cells.tsv"), delimiter="\t"))
    q = {}
    for ds in DS:
        for K in KS:
            for m in set(r["method"] for r in rows):
                sel = [float(r["psnr_mean"]) for r in rows if r["dataset"] == ds and r["budget"] == K and r["method"] == m]
                if sel:
                    q[(ds, K, m)] = st.mean(sel)
    return q


METRICS = ("psnr_mean", "ssim_mean", "lpips_mean", "temporal_lpips_delta_mean")
METHODS = ("seacache", "teacache", "sencache", "dicache", "budcache", "meancache",
           "taylorseer_o1", "hicache_o2", "l2p")
METHOD_ENTITY = {"seacache": "seacache", "teacache": "teacache", "sencache": "sencache",
                 "dicache": "dicache", "budcache": "budcache", "meancache": "meancache",
                 "taylorseer_o1": "shared", "hicache_o2": "shared", "l2p": "shared"}
SHARED_METHODS = ("taylorseer_o1", "hicache_o2", "l2p")
BASIN_RADII = (4, 6, 8)
MERGE_RADIUS = 6
MAX_MERGE_RADIUS = 25
MAIN_BASIN_MASS = 0.40
BASIN_MEMBER_SHARE = 0.10
#: the two normalised thresholds the image-side basin table uses, in bits of 50
NEAR_BITS = (5.0, 7.5)


def quality_all(T):
    """{(dataset, K, method, metric): seed mean} over the 162-cell table."""
    rows = list(csv.DictReader(open(REPO / f"docs/figures/baseline_matrix/{T}/cells.tsv"), delimiter="\t"))
    q = {}
    for ds in DS:
        for K in KS:
            for m in METHODS:
                for metric in METRICS:
                    sel = [float(r[metric]) for r in rows
                           if r["dataset"] == ds and r["budget"] == K and r["method"] == m]
                    if sel:
                        q[(ds, K, m, metric)] = st.mean(sel)
    return q


def alt_distance_block(ent: dict[str, dict[str, float]]) -> dict:
    """Entity-entity distances under the alternative ground metrics, plus the
    rank agreement of each reading with the Hamming reading."""
    out: dict = {g: {a: {} for a in ENT} for g in ("w_jaccard", "w_anchor")}
    out["tv"] = {a: {} for a in ENT}
    out["marginal_l1"] = {a: {} for a in ENT}
    out["e_budget"] = {a: {} for a in ENT}
    out["e_relocate"] = {a: {} for a in ENT}
    out["e_hamming"] = {a: {} for a in ENT}
    for a, b in combinations(ENT, 2):
        wj = wasserstein(ent[a], ent[b], "jaccard")
        wa = wasserstein(ent[a], ent[b], "anchor")
        tv = total_variation(ent[a], ent[b])
        ml = marginal_l1(ent[a], ent[b])
        eb, er = expected_budget_relocate(ent[a], ent[b])
        for key, val in (("w_jaccard", wj), ("w_anchor", wa), ("tv", tv), ("marginal_l1", ml),
                         ("e_budget", eb), ("e_relocate", er), ("e_hamming", eb + 2 * er)):
            out[key][a][b] = out[key][b][a] = val
    return out


def basin_block(ent, ent_by_ds, psnr_entity) -> dict:
    """Pair-distance summary, connectivity, mass-weighted components and their
    per-dataset stability, on the pooled path universe of one partition."""
    paths = sorted(set().union(*[set(ent[e]) for e in ENT]))
    idx = {p: i for i, p in enumerate(paths)}
    member = {e: [idx[p] for p in ent[e]] for e in ENT}
    out: dict = {}

    # connectivity: direct minimum distance against the merge radius r*
    d_min, r_star = {}, {}
    for a, b in combinations(ENT, 2):
        d_min[(a, b)] = min(hamming(paths[i], paths[j]) for i in member[a] for j in member[b])
    for r in range(MAX_MERGE_RADIUS + 1):
        comps = components(paths, r)
        root = {}
        for c, mem in enumerate(comps):
            for i in mem:
                root[i] = c
        for a, b in combinations(ENT, 2):
            if (a, b) not in r_star and {root[i] for i in member[a]} & {root[i] for i in member[b]}:
                r_star[(a, b)] = r
    out["bridging"] = {f"{a}|{b}": {"direct_min_hamming": d_min[(a, b)],
                                    "merge_radius": r_star[(a, b)],
                                    "bridging_gain": d_min[(a, b)] - r_star[(a, b)]}
                       for a, b in combinations(ENT, 2)}
    out["bridging_summary"] = {
        "pairs": len(d_min),
        "pairs_with_chain": sum(1 for k in d_min if r_star[k] < d_min[k]),
        "max_gain": max(d_min[k] - r_star[k] for k in d_min),
        "max_gain_pair": max(d_min, key=lambda k: d_min[k] - r_star[k]),
    }

    # mass-weighted components: one unit of decision mass per entity
    best_entity = max(ENT, key=lambda e: psnr_entity[e][1])
    out["components"] = {}
    for r in BASIN_RADII:
        comps = components(paths, r)
        rows = []
        for mem in comps:
            mass = {e: sum(ent[e].get(paths[i], 0.0) for i in mem) for e in ENT}
            rows.append({"n_paths": len(mem), "total_mass_of_7": sum(mass.values()),
                         "members_ge_10pct": [e for e in ENT if mass[e] >= BASIN_MEMBER_SHARE],
                         "mass": mass})
        rows.sort(key=lambda x: -x["total_mass_of_7"])
        best_i = max(range(len(rows)), key=lambda i: rows[i]["mass"][best_entity])
        out["components"][f"r{r}"] = {
            "n_components": len(rows),
            "n_main_components": sum(1 for x in rows if x["total_mass_of_7"] >= MAIN_BASIN_MASS),
            "n_multi_entity_components": sum(1 for x in rows if len(x["members_ge_10pct"]) > 1),
            "largest_mass_of_7": rows[0]["total_mass_of_7"],
            "cross_entity_pairs": sorted("|".join(sorted(pr)) for x in rows
                                         for pr in combinations(x["members_ge_10pct"], 2)),
            "best_quality_entity": best_entity,
            "best_quality_component_rank": best_i + 1,
            "popular_equals_best_quality": best_i == 0,
            "rows": [{k: v for k, v in x.items() if k != "mass"} for x in rows],
        }

    # per-dataset stability of the cross-entity co-basins at the merge radius
    per_ds = {}
    for ds in DS:
        p_ds = sorted(set().union(*[set(ent_by_ds[ds][e]) for e in ENT]))
        i_ds = {p: i for i, p in enumerate(p_ds)}
        mem_ds = {e: {i_ds[p] for p in ent_by_ds[ds][e]} for e in ENT}
        pairs = set()
        for mem in components(p_ds, MERGE_RADIUS):
            inside = [e for e in ENT if mem_ds[e] & set(mem)]
            pairs |= {"|".join(sorted(pr)) for pr in combinations(inside, 2)}
        per_ds[ds] = pairs
    allpairs = sorted(set().union(*per_ds.values()))
    out["merge_stability"] = {"radius": MERGE_RADIUS,
                              "pairs": {p: sum(1 for ds in DS if p in per_ds[ds]) for p in allpairs},
                              "n_datasets": len(DS)}
    return out


def linkage_block(entW, entH, Qall, K) -> dict:
    """Path distance against quality gap over all 36 method pairs, and the
    same-path-different-payload table over the three shared-schedule pairs."""
    dq = {m: {metric: st.mean(Qall[(ds, K, m, metric)] for ds in DS) for metric in METRICS}
          for m in METHODS}
    rng = {metric: max(dq[m][metric] for m in METHODS) - min(dq[m][metric] for m in METHODS)
           for metric in METRICS}
    pairs = []
    for a, b in combinations(METHODS, 2):
        ea, eb = METHOD_ENTITY[a], METHOD_ENTITY[b]
        w = 0.0 if ea == eb else entW[ea][eb]
        h = 0 if ea == eb else entH[ea][eb]
        pairs.append({"a": a, "b": b, "w_hamming": w, "top1_hamming": h, "same_path": ea == eb,
                      **{f"abs_delta_{metric}": abs(dq[a][metric] - dq[b][metric]) for metric in METRICS},
                      **{f"share_{metric}": abs(dq[a][metric] - dq[b][metric]) / rng[metric]
                         for metric in METRICS}})
    rho = {metric: spearman([p["w_hamming"] for p in pairs],
                            [p[f"abs_delta_{metric}"] for p in pairs]) for metric in METRICS}
    ws = sorted(p["w_hamming"] for p in pairs)
    ds_psnr = sorted(p["abs_delta_psnr_mean"] for p in pairs)
    mw, mq = st.median(ws), st.median(ds_psnr)
    quad = Counter(("near" if p["w_hamming"] <= mw else "far") + "-" +
                   ("near" if p["abs_delta_psnr_mean"] <= mq else "far") for p in pairs)
    same = []
    for a, b in combinations(SHARED_METHODS, 2):
        for ds in DS:
            r = {metric: max(Qall[(ds, K, m, metric)] for m in METHODS)
                 - min(Qall[(ds, K, m, metric)] for m in METHODS) for metric in METRICS}
            same.append({"a": a, "b": b, "dataset": ds,
                         **{f"abs_delta_{metric}": abs(Qall[(ds, K, a, metric)] - Qall[(ds, K, b, metric)])
                            for metric in METRICS},
                         **{f"share_{metric}": abs(Qall[(ds, K, a, metric)] - Qall[(ds, K, b, metric)]) / r[metric]
                            for metric in METRICS}})
    return {"pairs": pairs, "metric_range": rng, "spearman_w_vs_gap": rho,
            "quadrants_psnr": dict(quad), "median_w": mw, "median_abs_delta_psnr": mq,
            "same_path_different_payload": same}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    OUT: dict = {}

    for T in T_ALL:
        fixed = fixed_paths(T)
        pooled, per_seed = gate_dists(T)
        Q = quality(T)
        Qall = quality_all(T)
        OUT[T] = {"partitions": {}}
        for K in KS:
            P: dict = {"stats": {}, "top1": {}, "universe": {}, "dist": {}, "overlap": {}, "stability": {}}
            # ---- per-dataset distribution statistics for gates --------------
            entity_dist_by_ds: dict[str, dict[str, dict[str, float]]] = {ds: {} for ds in DS}
            for ds in DS:
                for g in GATES:
                    cnt = pooled[(g, ds, K)]
                    tot = sum(cnt.values())
                    mass = {p: c / tot for p, c in cnt.items()}
                    desc = sorted(mass.values(), reverse=True)
                    paths = list(mass)
                    diam = max((hamming(a, b) for a, b in combinations(paths, 2)), default=0)
                    P["stats"][f"{ds}/{g}"] = {
                        "generations": tot, "prompts": NPROMPT[ds], "seeds": 3,
                        "unique": len(mass), "top1": desc[0], "top2": sum(desc[:2]), "top3": sum(desc[:3]),
                        "top10": sum(desc[:10]), "entropy_bits": entropy_bits(desc), "cloud_diameter": diam,
                        "n90": n90(desc),
                        "top1_path": max(mass, key=mass.get),
                    }
                    entity_dist_by_ds[ds][g] = mass
                for f in FIXED:
                    entity_dist_by_ds[ds][f] = {fixed[K][f]: 1.0}
            # ---- equal-weight pooled distributions over the two datasets ------
            ent: dict[str, dict[str, float]] = {}
            for e in ENT:
                acc: dict[str, float] = defaultdict(float)
                for ds in DS:
                    for p, m in entity_dist_by_ds[ds][e].items():
                        acc[p] += m / len(DS)
                ent[e] = dict(acc)
            # ---- top-1 per entity ----------------------------------------------
            for e in ENT:
                p1 = max(ent[e], key=ent[e].get)
                P["top1"][e] = {"path": p1, "share": ent[e][p1], "n_cached": p1.count("1"), "warmup": warmup(p1),
                                "last_step_full": p1[49] == "0"}
            # ---- universe: distinct dynamic paths, pairwise Hamming --------------
            dyn_paths = sorted(set(p for g in GATES for p in ent[g]))
            pd = [hamming(a, b) for a, b in combinations(dyn_paths, 2)]
            Kint = int(K[1:])
            P["universe"] = {
                "n_dynamic_unique": len(dyn_paths), "n_fixed": len(FIXED),
                "n_generations_dynamic": sum(P["stats"][f"{ds}/{g}"]["generations"] for ds in DS for g in GATES),
                "pairwise_hamming_median": st.median(pd) if pd else None, "pairwise_hamming_max": max(pd) if pd else None,
                "exact_k_cap": min(2 * Kint, 2 * (50 - Kint)),
                "consensus": {  # fraction of the 7 entities' top-1 paths caching each step
                    "always_full_steps": [i for i in range(50) if all(P["top1"][e]["path"][i] == "0" for e in ENT)],
                    "always_cached_steps": [i for i in range(50) if all(P["top1"][e]["path"][i] == "1" for e in ENT)],
                },
            }
            # ---- three distance layers among the 7 entities ----------------------
            top3 = {e: sorted(ent[e], key=lambda p: -ent[e][p])[:3] for e in ENT}
            L1, L2, L3 = {}, {}, {}
            for a in ENT:
                L1[a], L2[a], L3[a] = {}, {}, {}
                for b in ENT:
                    if a == b:
                        continue
                    L1[a][b] = hamming(P["top1"][a]["path"], P["top1"][b]["path"])
                    L2[a][b] = min(hamming(x, y) for x in top3[a] for y in top3[b])
                    L3[a][b] = wasserstein_hamming(ent[a], ent[b])
            P["dist"] = {"top1_hamming": L1, "top3_min_hamming": L2, "w_hamming": L3}
            # nearest neighbour per entity by W
            P["nearest_by_w"] = {a: min(L3[a], key=L3[a].get) for a in ENT}
            # ---- exact overlaps -----------------------------------------------
            for a, b in combinations(ENT, 2):
                shared = set(ent[a]) & set(ent[b])
                P["overlap"][f"{a}|{b}"] = {"n_shared_paths": len(shared),
                                           "shared_mass_a": sum(ent[a][p] for p in shared),
                                           "shared_mass_b": sum(ent[b][p] for p in shared)}
            P["overlap_summary"] = {
                "pairs": len(list(combinations(ENT, 2))),
                "pairs_with_zero_overlap": sum(1 for v in P["overlap"].values() if v["n_shared_paths"] == 0),
            }
            # ---- stability ---------------------------------------------------------
            stab = {}
            for g in GATES:
                # across seed streams: is the top-1 path the same in all three streams (per dataset)?
                same_seed = {}
                for ds in DS:
                    tops = []
                    for seed in sorted(set(s for (gg, d, k, s) in per_seed if gg == g and d == ds and k == K)):
                        c = per_seed[(g, ds, K, seed)]
                        tops.append(max(c, key=c.get))
                    same_seed[ds] = len(set(tops)) == 1
                # across datasets: top-1 Hamming penguin vs vbench
                xds = hamming(P["stats"][f"penguin599/{g}"]["top1_path"], P["stats"][f"vbench944/{g}"]["top1_path"])
                stab[g] = {"top1_same_across_3_seeds": same_seed, "top1_hamming_penguin_vs_vbench": xds}
            P["stability"] = stab
            # ---- quality contrasts ----------------------------------------------------
            P["quality_psnr_penguin"] = {m: Q[("penguin599", K, m)] for m in ("seacache", "teacache", "sencache", "dicache", "budcache", "meancache", "taylorseer_o1", "hicache_o2", "l2p")}
            # ---- alternative ground metrics for the same entity pairs -----------
            P["dist_alt"] = alt_distance_block(ent)
            near = {"hamming": L3}
            near["jaccard"] = P["dist_alt"]["w_jaccard"]
            near["anchor"] = P["dist_alt"]["w_anchor"]
            P["nearest_by_ground"] = {g: {a: min(near[g][a], key=near[g][a].get) for a in ENT}
                                      for g in near}
            base = [L3[a][b] for a, b in combinations(ENT, 2)]
            P["ground_spearman"] = {
                name: spearman(base, [P["dist_alt"][key][a][b] for a, b in combinations(ENT, 2)])
                for name, key in (("jaccard", "w_jaccard"), ("anchor", "w_anchor"),
                                  ("total_variation", "tv"), ("step_marginal_l1", "marginal_l1"))}
            tvs = [P["dist_alt"]["tv"][a][b] for a, b in combinations(ENT, 2)]
            P["tv_summary"] = {"min": min(tvs), "max": max(tvs)}
            eh = sum(P["dist_alt"]["e_hamming"][a][b] for a, b in combinations(ENT, 2))
            P["decomposition"] = {
                "mean_expected_budget": st.mean(P["dist_alt"]["e_budget"][a][b] for a, b in combinations(ENT, 2)),
                "mean_expected_hamming": eh / len(list(combinations(ENT, 2))),
                "relocation_share_of_expected_hamming":
                    2 * sum(P["dist_alt"]["e_relocate"][a][b] for a, b in combinations(ENT, 2)) / eh,
            }
            # ---- overlap census per dataset and per pair class -------------------
            def kind(a, b):
                fa, fb = a in FIXED, b in FIXED
                return "fixed-fixed" if fa and fb else ("gate-fixed" if fa or fb else "gate-gate")
            census = {}
            for ds in DS:
                E = entity_dist_by_ds[ds]
                for kk in ("all", "gate-gate", "gate-fixed", "fixed-fixed"):
                    sel = [(a, b) for a, b in combinations(ENT, 2) if kk == "all" or kind(a, b) == kk]
                    shared = {(a, b): set(E[a]) & set(E[b]) for a, b in sel}
                    census[f"{ds}/{kk}"] = {
                        "pairs": len(sel),
                        "zero_overlap": sum(1 for p in sel if not shared[p]),
                        "max_shared_mass": max((sum(min(E[a][p], E[b][p]) for p in shared[(a, b)])
                                                for a, b in sel), default=0.0)}
            P["census_by_dataset"] = census
            # ---- basins and the path-distance / quality linkage -------------------
            psnr_entity = {}
            for e in ENT:
                ms = SHARED_METHODS if e == "shared" else (e,)
                vals = [st.mean(Qall[(ds, K, m, "psnr_mean")] for ds in DS) for m in ms]
                psnr_entity[e] = (min(vals), max(vals))
            P["psnr_entity_dataset_equal"] = {e: list(v) for e, v in psnr_entity.items()}
            P["basin"] = basin_block(ent, entity_dist_by_ds, psnr_entity)
            P["basin"]["pair_summary"] = {
                "pairs": len(base), "min": min(base), "mean": st.mean(base), "max": max(base),
                "cap": P["universe"]["exact_k_cap"],
                "mean_over_cap": st.mean(base) / P["universe"]["exact_k_cap"],
                "max_over_cap": max(base) / P["universe"]["exact_k_cap"],
                **{f"n_below_{b:g}_bits": sum(1 for v in base if v < b) for b in NEAR_BITS},
            }
            P["linkage"] = linkage_block(L3, L1, Qall, K)
            # ---- per-step marginals, nine-method consensus and disagreement ----
            marg = {e: step_marginal(ent[e]) for e in ENT}
            P["step_marginal"] = {e: [float(v) for v in marg[e]] for e in ENT}
            nine = np.array([marg[METHOD_ENTITY[m]] for m in METHODS])
            consensus, disagree = nine.mean(axis=0), nine.std(axis=0, ddof=0)
            zone = ["consensus-cache" if v >= CONSENSUS_CACHE_MIN else
                    ("consensus-full" if v <= CONSENSUS_FULL_MAX else "mixed") for v in consensus]
            order = sorted(range(50), key=lambda t: (-disagree[t], t))
            P["consensus"] = {
                "n_methods": len(METHODS),
                "consensus": [float(v) for v in consensus],
                "disagree": [float(v) for v in disagree],
                "zone": zone,
                "n_consensus_cache": sum(1 for z in zone if z == "consensus-cache"),
                "n_consensus_full": sum(1 for z in zone if z == "consensus-full"),
                "n_mixed": sum(1 for z in zone if z == "mixed"),
                "consensus_cache_steps": [t for t in range(50) if zone[t] == "consensus-cache"],
                "consensus_full_steps": [t for t in range(50) if zone[t] == "consensus-full"],
                "peaks": [{"rank": r, "step": t, "disagree": float(disagree[t]),
                           "consensus": float(consensus[t]), "zone": zone[t]}
                          for r, t in enumerate(order[:N_PEAKS], 1)],
            }
            # ---- shared cached steps of two top-1 paths against chance ---------
            ro = {}
            for a, b in combinations(ENT, 2):
                pa, pb = P["top1"][a]["path"], P["top1"][b]["path"]
                ka, kb = pa.count("1"), pb.count("1")
                shared = sum(1 for i in range(50) if pa[i] == "1" and pb[i] == "1")
                base = random_overlap_baseline(ka, kb)
                ro[f"{a}|{b}"] = {"k_a": ka, "k_b": kb, "shared_cached_steps": shared,
                                  "chance": base, "excess": shared - base}
            exc = sorted(v["excess"] for v in ro.values())
            P["random_overlap"] = {
                "n_free_positions": N_FREE_POSITIONS, "pairs": ro,
                "summary": {"pairs": len(ro), "min": exc[0], "median": st.median(exc), "max": exc[-1],
                            "n_within_2": sum(1 for v in exc if abs(v) <= 2),
                            "chance_min": min(v["chance"] for v in ro.values()),
                            "chance_max": max(v["chance"] for v in ro.values())},
            }
            # ---- the two offline searchers against each other -------------------
            allw = [L3[a][b] for a, b in combinations(ENT, 2)]
            P["searcher_pair"] = {
                "top1_hamming": L1["budcache"]["meancache"],
                "top3_min_hamming": L2["budcache"]["meancache"],
                "w_hamming": L3["budcache"]["meancache"],
                "percentile_in_21_pairs": percentile_of_score(allw, L3["budcache"]["meancache"]),
                "e_budget": P["dist_alt"]["e_budget"]["budcache"]["meancache"],
                "e_relocate": P["dist_alt"]["e_relocate"]["budcache"]["meancache"],
            }
            # ---- the transport distance grouped by pair family -------------------
            def pair_kind(a, b):
                fa, fb = a in FIXED, b in FIXED
                return "fixed-fixed" if fa and fb else ("gate-fixed" if fa or fb else "gate-gate")
            P["w_by_family"] = {}
            for kk in ("gate-gate", "gate-fixed", "fixed-fixed"):
                sel = [L3[a][b] for a, b in combinations(ENT, 2) if pair_kind(a, b) == kk]
                P["w_by_family"][kk] = {"pairs": len(sel), "mean": st.mean(sel),
                                        "min": min(sel), "max": max(sel)}
            # ---- structure: expected longest run of consecutive cached steps -----
            P["structure"] = {e: {"longest_run_exp": expected_longest_run(ent[e]),
                                  "longest_run_top1": longest_run(P["top1"][e]["path"])} for e in ENT}
            # ---- two datasets: distribution sharing, top-3 overlap ---------------
            share = {}
            for g in GATES:
                a, b = entity_dist_by_ds["penguin599"][g], entity_dist_by_ds["vbench944"][g]
                common = sorted(set(a) & set(b))
                t3a = sorted(a, key=lambda p: (-a[p], p))[:3]
                t3b = sorted(b, key=lambda p: (-b[p], p))[:3]
                share[g] = {"n_paths_penguin": len(a), "n_paths_vbench": len(b),
                            "n_shared_paths": len(common),
                            "shared_mass": sum(min(a[p], b[p]) for p in common),
                            "n_shared_top3": len(set(t3a) & set(t3b))}
            P["dataset_sharing"] = share
            # ---- per-dataset path universe panels --------------------------------
            P["universe_by_dataset"] = {}
            for ds in DS:
                pts = sorted(set(p for g in GATES for p in entity_dist_by_ds[ds][g]))
                P["universe_by_dataset"][ds] = {
                    "n_dynamic_unique": len(pts), "n_fixed": len(FIXED),
                    "n_generations_dynamic": sum(P["stats"][f"{ds}/{g}"]["generations"] for g in GATES),
                    "prompts": NPROMPT[ds],
                    "mds_2d_fidelity": layout_fidelity(pts + [fixed[K][f] for f in FIXED]),
                }
            P["universe"]["mds_2d_fidelity"] = layout_fidelity(
                dyn_paths + [fixed[K][f] for f in FIXED])
            # ---- concentration against quality over the four gates ----------------
            cq = {}
            for ds in DS:
                psnr = [Qall[(ds, K, g, "psnr_mean")] for g in GATES]
                cq[ds] = {feat: spearman([P["stats"][f"{ds}/{g}"][key] for g in GATES], psnr)
                          for feat, key in (("top1_share", "top1"), ("entropy_bits", "entropy_bits"))}
                cq[ds]["n_gates"] = len(GATES)
            P["concentration_vs_quality"] = cq
            # ---- the closest fixed-gate pair: nearly one path, two payloads -------
            emeth = {"shared": SHARED_METHODS, "budcache": ("budcache",), "meancache": ("meancache",)}
            dmin = min(L1[f][g] for f in FIXED for g in GATES)
            tied = []
            for f in FIXED:
                for g in GATES:
                    if L1[f][g] != dmin:
                        continue
                    fq = [st.mean(Qall[(ds, K, m, "psnr_mean")] for ds in DS) for m in emeth[f]]
                    gq = st.mean(Qall[(ds, K, g, "psnr_mean")] for ds in DS)
                    tied.append({
                        "fixed": f, "gate": g, "top1_hamming": dmin,
                        "relocations": (dmin - abs(P["top1"][f]["path"].count("1")
                                                   - P["top1"][g]["path"].count("1"))) / 2,
                        "psnr_fixed_min": min(fq), "psnr_fixed_max": max(fq), "psnr_gate": gq,
                        "psnr_gap_min": min(abs(v - gq) for v in fq),
                        "psnr_gap_max": max(abs(v - gq) for v in fq)})
            P["closest_fixed_gate"] = {"top1_hamming": dmin, "n_tied": len(tied), "pairs": tied}
            OUT[T]["partitions"][K] = P
        # ---- across budgets: nesting of top-1 cached-step sets ------------------
        nest = {}
        for e in ENT:
            s29 = set(cache_steps(OUT[T]["partitions"]["K29"]["top1"][e]["path"]))
            s37 = set(cache_steps(OUT[T]["partitions"]["K37"]["top1"][e]["path"]))
            s41 = set(cache_steps(OUT[T]["partitions"]["K41"]["top1"][e]["path"]))
            nest[e] = {"K29_subset_of_K37": s29 <= s37, "K37_subset_of_K41": s37 <= s41,
                       "K29_steps_not_in_K37": len(s29 - s37), "K37_steps_not_in_K41": len(s37 - s41),
                       # retention: share of the smaller budget's cached steps that survive
                       "retention_K29_in_K37": len(s29 & s37) / len(s29),
                       "retention_K37_in_K41": len(s37 & s41) / len(s37)}
        OUT[T]["budget_nesting"] = nest
    # ---- across backbones: same entity top-1 HYV vs Wan --------------------------
    OUT["cross_backbone_top1_hamming"] = {
        f"{e}/{K}": hamming(OUT["hunyuan_video"]["partitions"][K]["top1"][e]["path"], OUT["wan21"]["partitions"][K]["top1"][e]["path"])
        for e in ENT for K in KS}
    # ---- across backbones: shape agreement of the two step-marginal curves -------
    OUT["cross_backbone_shape_agreement"] = {}
    for e in ENT:
        for K in KS:
            a = np.array(OUT["hunyuan_video"]["partitions"][K]["step_marginal"][e])
            b = np.array(OUT["wan21"]["partitions"][K]["step_marginal"][e])
            OUT["cross_backbone_shape_agreement"][f"{e}/{K}"] = {
                "sign_agree": sign_agreement(a, b),
                "n_steps_agree": int((np.sign(a - a.mean()) == np.sign(b - b.mean())).sum()),
                "expected_k_hunyuan_video": float(a.sum()), "expected_k_wan21": float(b.sum()),
                "same_top1": OUT["cross_backbone_top1_hamming"][f"{e}/{K}"] == 0,
            }
    (args.out / "paths.json").write_text(json.dumps(OUT, indent=1, default=str))
    print("wrote", args.out / "paths.json")
    for T in T_ALL:
        for K in KS:
            U = OUT[T]["partitions"][K]["universe"]; o = OUT[T]["partitions"][K]["overlap_summary"]
            print(f"  {T} {K}: dyn unique {U['n_dynamic_unique']} from {U['n_generations_dynamic']} gens; pair H med/max {U['pairwise_hamming_median']}/{U['pairwise_hamming_max']} cap {U['exact_k_cap']}; zero-overlap pairs {o['pairs_with_zero_overlap']}/{o['pairs']}; always-full {U['consensus']['always_full_steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
