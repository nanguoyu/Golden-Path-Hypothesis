"""Wan2.1 method objects for the nine-method baseline matrix.

Six of the nine methods are the HunyuanVideo state machines imported unchanged
(`docs/wan21_baseline_matrix_plan_zh.md` section 2.3): hicache, taylorseer,
reuse/budcache, l2p, meancache and sencache are pure math plus a gate state
machine over `hunyuan_video.actions` dataclasses, so the only Wan-specific part
is the constants they are constructed with.  Two of the nine cannot be reused
and are built here:

* **TeaCache** -- `hunyuan_video/methods/teacache.py:27-30` hardcodes the
  Hunyuan `img_mod`/`img_norm1` modulated input, while the official
  TeaCache4Wan2.1 `use_ret_steps=True` t2v branch scores the *timestep
  embedding* e0
  (`reference/teacache/code/TeaCache4Wan2.1/teacache_generate.py:520`).  The
  indicator and the `("wan21", "1.3b")` coefficients are one unit; the old
  `wan21/teacache.py` shim put those coefficients on the modulated fingerprint
  and is a recorded semantic mismatch (plan section 2.6).
* **SeaCache** -- the Wan fingerprint is the block-0 modulated input built from
  `(blocks[0].modulation + e0).chunk(6)` (`wan21/seacache.py:326-328`), not the
  Hunyuan one.  The SEA filter itself is bound to the in-repo Wan
  implementation rather than re-derived, so the two lanes cannot drift.

The ninth method, DiCache, is a fused gate + adapter and lives in
`wan21/dicache.py` (plan section 2.10).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
import torch.cuda.amp as amp

from hunyuan_video.actions import FixedScheduleAction, MethodDecision
from hunyuan_video.methods.hicache import HiCacheMethod
from hunyuan_video.methods.l2p import L2POutputMethod
from hunyuan_video.methods.meancache import MeanCacheMethod
from hunyuan_video.methods.reuse import ReuseMethod
from hunyuan_video.methods.sencache import SenCacheMethod
from hunyuan_video.methods.taylorseer import TaylorSeerMethod
from lib.gates import rel_l1
from lib.teacache_coeffs import get_coeffs
from wan21.seacache import CacheForwardConfig, _prepare_gate_fingerprint


# ---------------------------------------------------------------------------
# Wan2.1 t2v-1.3B protocol constants (plan sections 1.1 / 2.0)
# ---------------------------------------------------------------------------

#: `reference/taylorseer/code/TaylorSeer-Wan2.1/wan/configs/wan_t2v_1_3B.py:25`.
WAN_NUM_LAYERS = 30

#: Fine payload slots: 30 uniform blocks x {self_attn, cross_attn, ffn}.  Not
#: the Hunyuan 120 (20 double x 4 + 40 single x 1), and not the stale "fine_96"
#: label at `wan21/fine_payload.py:463` -- 30 x 3 = 90.
WAN_FINE_SLOT_COUNT = WAN_NUM_LAYERS * 3

#: Original block calls on a fully-computed solver step: 30 blocks x 2 CFG
#: forwards.  Numerically equal to the Hunyuan 60 (20 double + 40 single, single
#: forward) by coincidence only -- the Wan derivation is 30 x 2.
WAN_ORIGINAL_BLOCK_CALLS_PER_STEP = WAN_NUM_LAYERS * 2

#: Wan-VAE latent element count at 832x480x65 frames: 16 channels x 17 latent
#: frames x 60 x 104 (8x spatial / 4x temporal stride).  sqrt(d) ~= 1302.80.
#: The upstream Wan SenCache hardcodes SCALING_FACTOR=1447.98 for its 81-frame
#: shape (16 x 21 x 60 x 104); same semantics, different number -- the constant
#: must not be copied (plan section 2.5).
WAN_LATENT_NUMEL = 16 * 17 * 60 * 104

# Per-method warmup / boundary constants, each from its own authority.
TAYLORSEER_FIRST_ENHANCE = 3   # plan section 2.2 item 1
TAYLORSEER_MAX_ORDER = 1       # O1, SeaCache-paper FLUX convention
HICACHE_FIRST_ENHANCE = 3      # `reference/hicache/code/models/hicache_fast_impl.py:130`
HICACHE_MAX_ORDER = 2          # O2
HICACHE_SIGMA = 0.5
TEACACHE_RET_STEPS = 5         # official `ret_steps = 5*2` calls = 5 solver steps
SENCACHE_FIRST_ENHANCE = 3     # plan section 2.2 (upstream Wan has no warmup)
SENCACHE_MAX_SKIP = 10         # upstream `sencache_K` default
SENCACHE_SWITCH_RATIO = 0.2    # upstream `total_calls * 0.2`
SENCACHE_RET_STEPS = 0         # upstream `retention_steps = 0`
SENCACHE_CUTOFF_STEPS = -1     # -> terminal step forced full
MEANCACHE_FIRST_FULL_STEPS = 5
MEANCACHE_LAST_FULL_STEPS = 1
MEANCACHE_JVP_SPAN = 4
DICACHE_RET_RATIO = 0.2
DICACHE_PROBE_DEPTH = 1

#: The nine matrix methods, in the plan's family order (section 1).
MATRIX_METHODS = (
    "seacache",
    "teacache",
    "sencache",
    "dicache",
    "taylorseer_o1",
    "hicache_o2",
    "l2p",
    "budcache",
    "meancache",
)

#: Methods whose decision is a dynamic gate or a whole-transformer reuse
#: schedule: one method object per run, driven on the cond branch only.
COARSE_METHODS = frozenset({"seacache", "teacache", "sencache", "budcache"})

#: Methods whose *payload* state (slot histories, L2P history, velocity
#: history) differs between the CFG branches: one method object per branch.
PER_BRANCH_METHODS = frozenset({"taylorseer_o1", "hicache_o2", "l2p", "meancache"})

#: The three methods that share one fixed schedule table per cache tier
#: (plan section 3.3).
SHARED_SCHEDULE_METHODS = ("taylorseer_o1", "hicache_o2", "l2p")


#: What `relax_warmup=True` lowers each method's head-of-trajectory warmup to.
#: Used by the video SPX experiment only (`--spx_relax_warmup`), where foreign
#: schedules are transplanted onto each payload and a warmup that is a search
#: convention rather than a structural requirement would delete whole cells:
#:
#:   * `meancache` 5 -> 2: the JVP span is already clamped to the available
#:     history and degrades to velocity reuse below 2
#:     (`hunyuan_video/methods/meancache.py:98-102`);
#:   * `budcache` 3 -> 1: `ReuseMethod` needs one previous residual and nothing
#:     else, which is exactly what the Hunyuan `reuse_exact` twin enforces
#:     (`hunyuan_video/backend.py:235`). The 3 comes from the BudCache search's
#:     own space, not from the payload.
#:
#: The frozen matrix never passes the flag, so its 162 cells are unaffected.
RELAXED_FIRST_FULL_STEPS = {"meancache": 2, "budcache": 1}


def forbidden_cache_steps(method: str, num_steps: int, *,
                          relax_warmup: bool = False) -> frozenset[int]:
    """Steps a fixed-schedule method may never cache (plan section 3.3).

    Derived from each method's own warmup constants rather than tabulated, so a
    change to `first_enhance` cannot leave the schedule validator behind.
    `wan21/matrix_config.py::FORBIDDEN_CACHE_STEPS` is expected to be built from
    this function, not to restate it.

    `relax_warmup` is the video SPX opt-in described at
    `RELAXED_FIRST_FULL_STEPS`; it touches `meancache` and `budcache` only.
    """

    total = int(num_steps)
    terminal = {total - 1}
    if method == "taylorseer_o1":
        return frozenset({0} | terminal)
    if method == "hicache_o2":
        return frozenset(set(range(HICACHE_FIRST_ENHANCE)) | terminal)
    if method == "l2p":
        # L2P degrades to full when its history is not ready, so only step 0 is
        # structurally impossible (`hunyuan_video/methods/l2p.py:50-52`).
        return frozenset({0})
    if method == "budcache":
        head = RELAXED_FIRST_FULL_STEPS["budcache"] if relax_warmup else 3
        return frozenset(set(range(head)) | terminal)
    if method == "meancache":
        head = (RELAXED_FIRST_FULL_STEPS["meancache"] if relax_warmup
                else MEANCACHE_FIRST_FULL_STEPS)
        return frozenset(set(range(head)) | terminal)
    if method == "dicache":
        # DiCache has no fixed schedule in the matrix at all; the video SPX
        # `di_two_anchor` column gives it one, and its two-anchor payload needs
        # two full steps in front of the first cached one.
        return frozenset({0, 1} | terminal)
    raise KeyError(f"{method} has no fixed schedule on the Wan2.1 matrix lane")


# ---------------------------------------------------------------------------
# The CFG dual-forward ruling
# ---------------------------------------------------------------------------


class CondDecidesArbiter:
    """The plan's section 2.1 ruling, in one object.

    Wan2.1 runs two model forwards per solver step -- cond then uncond
    (`wan21/runner.py:407-408`).  If both branches gated independently, "was
    this step cached" would have no single answer and the K29/K37/K41 tier
    definition, the threshold search's step counting and the comparison against
    fixed-schedule methods would all lose their meaning.  So:

    * the cache/full decision is taken **once, on the cond branch**, and the
      uncond branch executes the same action;
    * payload state is **not** shared -- each branch keeps its own residual,
      slot history, L2P history and velocity history;
    * gate features are read on the cond branch.

    This object arbitrates the action only; it never touches payload state.
    Precedent for the cond-defines / uncond-consumes convention is already in
    the lane at `wan21/taylorseer_fine.py:318-323`, and the official SenCache
    Wan2.1 variant (`knowledge/SenCache/code/Wan2.1/sencache.py:70-104`) is
    itself cond-decides.  For TeaCache and DiCache, whose official Wan variants
    gate per branch on `cnt % 2`, this is a deliberate recorded deviation.
    """

    BRANCH_NAMES = {0: "cond", 1: "uncond"}
    POLICY = "cond_decides"

    def __init__(self, num_steps: int) -> None:
        self.num_steps = int(num_steps)
        self.reset()

    def reset(self) -> None:
        self.forward_index = 0
        self.step = 0
        self.branch = "cond"
        self.decision: MethodDecision | None = None

    @property
    def is_cond(self) -> bool:
        return self.branch == "cond"

    def begin_forward(self) -> tuple[int, str]:
        """Advance the clock at the model pre-hook; returns (step, branch)."""

        if self.forward_index >= 2 * self.num_steps:
            raise RuntimeError(
                f"Wan2.1 model forward {self.forward_index} exceeds the "
                f"{self.num_steps}-step x 2-branch protocol"
            )
        self.step = self.forward_index // 2
        self.branch = self.BRANCH_NAMES[self.forward_index % 2]
        if self.is_cond:
            self.decision = None
        elif self.decision is None:
            raise RuntimeError(f"uncond branch reached step {self.step} before cond")
        return self.step, self.branch

    def settle(self, decision: MethodDecision) -> MethodDecision:
        """Adopt the cond branch's decision as this step's action."""

        if not self.is_cond:
            raise RuntimeError("only the cond branch may settle a step decision")
        self.decision = decision
        return decision

    def follow(self, decision: MethodDecision | None = None) -> MethodDecision:
        """Return the settled action for the uncond branch.

        Per-branch methods run their own fixed-schedule state machine so their
        warmup clamp advances in lockstep; passing that decision in asserts the
        two schedules agree, which they must, since both branches read the same
        frozen table.
        """

        if self.decision is None:
            raise RuntimeError(f"uncond branch reached step {self.step} before cond")
        if decision is not None and bool(decision.full) != bool(self.decision.full):
            raise RuntimeError(
                f"uncond schedule disagrees with cond at step {self.step}: "
                f"{decision.reason!r} vs {self.decision.reason!r}"
            )
        return self.decision

    def end_forward(self) -> bool:
        """Advance the clock at the model post-hook.

        Returns True once both branches of the current step have run, which is
        when the step's single decision record is complete.
        """

        closed = not self.is_cond
        self.forward_index += 1
        return closed


# ---------------------------------------------------------------------------
# The two Wan-native gates
# ---------------------------------------------------------------------------


def polynomial(value: float, coefficients: Sequence[float]) -> float:
    """Horner evaluation, highest-degree coefficient first (numpy.poly1d order)."""

    result = 0.0
    for coefficient in coefficients:
        result = result * value + float(coefficient)
    return result


def modulated_input(x: torch.Tensor, e: torch.Tensor, first_block: Any) -> torch.Tensor:
    """Wan's block-0 modulated input, the SeaCache fingerprint source.

    Byte-for-byte the computation at `wan21/seacache.py:326-328`, which is in
    turn `WanAttentionBlock.forward`'s pre-self-attention normalisation
    (`.../wan/modules/model.py:293-301`) evaluated with block 0's modulation.
    It is inline upstream, so it is written out here rather than imported.
    """

    with amp.autocast(dtype=torch.float32):
        chunks = (first_block.modulation + e).chunk(6, dim=1)
        return first_block.norm1(x).float() * (1 + chunks[1]) + chunks[0]


class WanSeaCacheGate:
    """SeaCache gate on Wan's native fingerprint.

    The accumulator and the forced-boundary rule match
    `hunyuan_video/methods/seacache.py`; the fingerprint and its SEA filtering
    are bound to the in-repo Wan implementation
    (`wan21/seacache.py:157-178`, dims=(-2,-3,-4), mode="flow", power_exp 3.0),
    which was validated against
    `reference/seacache/code/Wan2.1/{seacache_generate,util_seacache}.py` and
    ran the n=200 golden-path batch.
    """

    name = "seacache"

    def __init__(
        self,
        *,
        num_steps: int,
        threshold: float,
        scheduler_provider: Callable[[], Any],
        first_enhance: int = 1,
        power_exp: float = 3.0,
        norm_mode: str = "mean",
    ) -> None:
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.scheduler_provider = scheduler_provider
        self.first_enhance = int(first_enhance)
        self.power_exp = float(power_exp)
        self.norm_mode = str(norm_mode)
        # `_prepare_gate_fingerprint` reads its knobs off a CacheForwardConfig;
        # feeding it one keeps the matrix lane on the same code path as the
        # golden-path lane instead of a second copy of the SEA call.
        self._sea_cfg = CacheForwardConfig(
            mode="SeaCache",
            num_steps=self.num_steps,
            first_enhance=self.first_enhance,
            seacache_thresh=self.threshold,
            seacache_power_exp=self.power_exp,
            seacache_norm_mode=self.norm_mode,
        )
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.accumulated = 0.0
        self.previous_fingerprint: torch.Tensor | None = None

    def decide(
        self,
        *,
        x: torch.Tensor,
        e: torch.Tensor,
        first_block: Any,
        grid_sizes: torch.Tensor,
        **_kwargs: Any,
    ) -> MethodDecision:
        current = _prepare_gate_fingerprint(
            modulated=modulated_input(x, e, first_block),
            grid_sizes=grid_sizes,
            scheduler=self.scheduler_provider(),
            step=self.step,
            cfg=self._sea_cfg,
            gate="seacache",
        )
        force = (
            self.step < self.first_enhance
            or self.step == self.num_steps - 1
            or self.previous_fingerprint is None
        )
        scalar: float | None = None
        if force:
            full = True
            reason = "forced_boundary"
            self.accumulated = 0.0
        else:
            scalar = rel_l1(current, self.previous_fingerprint)
            self.accumulated += scalar
            full = self.accumulated >= self.threshold
            reason = "threshold_full" if full else "threshold_cache"
            if full:
                self.accumulated = 0.0
        self.previous_fingerprint = current.detach()
        self.step += 1
        return MethodDecision(
            full=full,
            reason=reason,
            gate_scalar=scalar,
            accumulated=self.accumulated,
            threshold=self.threshold,
        )


class WanTeaCacheGate:
    """Official TeaCache4Wan2.1 `use_ret_steps=True` t2v state machine.

    `reference/teacache/code/TeaCache4Wan2.1/teacache_generate.py:520-535,886`:
    the indicator is the timestep embedding **e0** (`self.time_projection(e)`),
    not the TeaCache paper's first-block modulated noisy input; the relative L1
    change of e0 is rescaled by the quartic fitted for that branch and
    accumulated; a step is cached while the accumulator stays below the
    threshold, and the accumulator is zeroed on every full step.  Warmup is
    `ret_steps = 5*2` calls, i.e. 5 solver steps.

    The coefficients and the indicator are one unit: `lib/teacache_coeffs.py:37`
    holds exactly the `('wan21', '1.3b')` row this branch fits, and pairing it
    with any other indicator is the recorded PSNR-13.7 failure of the old
    `wan21/teacache.py` shim.

    Two recorded deviations from upstream: the terminal step is forced full
    (upstream sets `cutoff_steps = sample_steps*2`, i.e. never), matching this
    repository's Sea/Tea matrix convention, and the decision is taken on the
    cond branch instead of per branch on `cnt % 2` (plan sections 2.1 / 2.6).
    """

    name = "teacache"

    def __init__(
        self,
        *,
        num_steps: int,
        threshold: float,
        coefficients: Sequence[float] | None = None,
        ret_steps: int = TEACACHE_RET_STEPS,
        force_last: bool = True,
    ) -> None:
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.coefficients = tuple(
            float(value)
            for value in (coefficients if coefficients is not None else get_coeffs("wan21", "1.3b"))
        )
        self.ret_steps = int(ret_steps)
        self.force_last = bool(force_last)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.accumulated = 0.0
        self.previous_e0: torch.Tensor | None = None

    def decide(self, *, e: torch.Tensor, **_kwargs: Any) -> MethodDecision:
        current = e
        force = (
            self.step < self.ret_steps
            or (self.force_last and self.step == self.num_steps - 1)
            or self.previous_e0 is None
        )
        scalar: float | None = None
        if force:
            full = True
            reason = "forced_boundary"
            self.accumulated = 0.0
        else:
            raw = rel_l1(current, self.previous_e0)
            scalar = polynomial(raw, self.coefficients)
            self.accumulated += scalar
            full = self.accumulated >= self.threshold
            reason = "threshold_full" if full else "threshold_cache"
            if full:
                self.accumulated = 0.0
        # Upstream refreshes `previous_e0` on every call, cached or not.
        self.previous_e0 = current.detach().clone()
        self.step += 1
        return MethodDecision(
            full=full,
            reason=reason,
            gate_scalar=scalar,
            accumulated=self.accumulated,
            threshold=self.threshold,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def fixed_cache_steps(
    config: Mapping[str, Any],
    *,
    method: str,
    num_steps: int,
) -> frozenset[int]:
    """Validate and return a frozen schedule table entry.

    A table the offline search could not have emitted fails here rather than on
    the GPU, following `hunyuan_video/backend.py:402-418`.
    """

    raw = config.get("cache_steps")
    if not isinstance(raw, list) or any(
        isinstance(step, bool) or not isinstance(step, int) for step in raw
    ):
        raise TypeError(f"{method} cache_steps must be a list of integers")
    if raw != sorted(set(raw)):
        raise ValueError(f"{method} cache_steps must be sorted and unique")
    count = config.get("cache_count")
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{method} cache_count must be an integer")
        if count != len(raw):
            raise ValueError(f"{method} cache_count differs from cache_steps")
    outside = [step for step in raw if step < 0 or step >= int(num_steps)]
    if outside:
        raise ValueError(f"{method} cache steps outside trajectory: {outside}")
    relax = bool(config.get("relax_warmup", False))
    banned = sorted(forbidden_cache_steps(method, num_steps,
                                          relax_warmup=relax).intersection(raw))
    if banned:
        raise ValueError(f"{method} caches steps it must keep full: {banned}")
    return frozenset(raw)


def build_coarse_method(
    method: str,
    *,
    num_steps: int,
    config: Mapping[str, Any],
    scheduler_provider: Callable[[], Any],
) -> Any:
    """Build the single method object a coarse-residual adapter drives."""

    if method == "seacache":
        return WanSeaCacheGate(
            num_steps=num_steps,
            threshold=float(config["threshold"]),
            scheduler_provider=scheduler_provider,
            first_enhance=int(config.get("first_enhance", 1)),
            power_exp=float(config.get("power_exp", 3.0)),
            norm_mode=str(config.get("norm_mode", "mean")),
        )
    if method == "teacache":
        return WanTeaCacheGate(
            num_steps=num_steps,
            threshold=float(config["threshold"]),
            coefficients=get_coeffs("wan21", str(config.get("variant", "1.3b"))),
            ret_steps=int(config.get("ret_steps", TEACACHE_RET_STEPS)),
        )
    if method == "sencache":
        return SenCacheMethod(
            num_steps=num_steps,
            sensitivity_path=str(config["sensitivity_path"]),
            # No default: the image side freezes this per cell
            # (`flux/sencache.py:284` makes it a required argument too).
            threshold_start=float(config["threshold_start"]),
            threshold_main=float(config["threshold"]),
            first_enhance=int(config.get("first_enhance", SENCACHE_FIRST_ENHANCE)),
            max_skip=int(config.get("max_skip", SENCACHE_MAX_SKIP)),
            switch_ratio=float(config.get("switch_ratio", SENCACHE_SWITCH_RATIO)),
            ret_steps=int(config.get("ret_steps", SENCACHE_RET_STEPS)),
            cutoff_steps=int(config.get("cutoff_steps", SENCACHE_CUTOFF_STEPS)),
        )
    if method == "budcache":
        return ReuseMethod(
            FixedScheduleAction(
                int(num_steps),
                fixed_cache_steps(config, method=method, num_steps=num_steps),
            )
        )
    raise KeyError(f"{method} is not a Wan2.1 coarse-residual matrix method")


def _build_branch_method(
    method: str,
    *,
    num_steps: int,
    config: Mapping[str, Any],
    cache_steps: frozenset[int],
    scheduler_provider: Callable[[], Any],
) -> Any:
    if method == "taylorseer_o1":
        return TaylorSeerMethod(
            action=FixedScheduleAction(int(num_steps), cache_steps),
            max_order=TAYLORSEER_MAX_ORDER,
            # The warmup order clamp. O1's clamp is numerically invisible on a
            # table that keeps steps 0-2 full, but it is a paradigm requirement
            # (plan section 2.2 item 1) and is not omitted.
            first_enhance=int(config.get("first_enhance", TAYLORSEER_FIRST_ENHANCE)),
        )
    if method == "hicache_o2":
        return HiCacheMethod(
            action=FixedScheduleAction(int(num_steps), cache_steps),
            max_order=int(config.get("max_order", HICACHE_MAX_ORDER)),
            sigma=float(config.get("sigma", HICACHE_SIGMA)),
            first_enhance=int(config.get("first_enhance", HICACHE_FIRST_ENHANCE)),
        )
    if method == "l2p":
        return L2POutputMethod(
            action=FixedScheduleAction(int(num_steps), cache_steps),
            # Hard requirement: no weights file, no run (plan section 2.8).
            weights_path=str(config["weights_path"]),
            num_steps=int(num_steps),
            min_abs_weight=float(config.get("min_abs_weight", 0.0)),
        )
    if method == "meancache":
        raw_spans = config.get("jvp_spans") or {}
        if not isinstance(raw_spans, dict):
            raise TypeError("meancache jvp_spans must be a mapping")
        return MeanCacheMethod(
            action=FixedScheduleAction(int(num_steps), cache_steps),
            num_steps=int(num_steps),
            # FlowUniPCMultistepScheduler.set_timesteps appends the final sigma
            # (`.../wan/utils/fm_solvers_unipc.py:205-209`), so `scheduler.sigmas`
            # is exactly num_steps+1 long once the run has set its timesteps --
            # which `wan21/runner.py:398` does per generation, so the sigmas are
            # read through the provider at step time, not bound here.
            scheduler_provider=scheduler_provider,
            # The global fallback for every cached step the table gives no
            # per-edge span for. The matrix's own tables cover every edge, so
            # this only bites on a transplanted schedule (video SPX).
            jvp_span=int(config.get("jvp_span", MEANCACHE_JVP_SPAN)),
            jvp_spans={int(step): int(span) for step, span in raw_spans.items()},
        )
    raise KeyError(f"{method} is not a Wan2.1 per-branch matrix method")


def build_branch_methods(
    method: str,
    *,
    num_steps: int,
    config: Mapping[str, Any],
    scheduler_provider: Callable[[], Any],
) -> dict[str, Any]:
    """Build one method object per CFG branch.

    These four carry payload state that differs between cond and uncond -- slot
    histories, L2P predictions, velocity histories -- so the branches cannot
    share an object.  Both run the same frozen table, and the arbiter asserts
    they agree step by step.
    """

    cache_steps = fixed_cache_steps(config, method=method, num_steps=num_steps)
    return {
        branch: _build_branch_method(
            method,
            num_steps=num_steps,
            config=config,
            cache_steps=cache_steps,
            scheduler_provider=scheduler_provider,
        )
        for branch in ("cond", "uncond")
    }
