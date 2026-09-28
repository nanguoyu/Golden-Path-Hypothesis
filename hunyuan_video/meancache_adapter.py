from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from hunyuan_video.actions import MethodDecision
from hunyuan_video.adapter import _restore_instance_forward, _set_instance_forward
from hunyuan_video.methods.meancache import MeanCacheMethod


def patchify(latent: torch.Tensor, patch_size: Sequence[int]) -> torch.Tensor:
    """Inverse of `HYVideoDiffusionTransformer.unpatchify`.

    `reference/hunyuan_video/code/hyvideo/modules/models.py:686-698` maps the
    final-layer tokens `(N, t*h*w, c*pt*ph*pw)` onto the velocity prediction
    `(N, c, t*pt, h*ph, w*pw)`. MeanCache substitutes that final-layer output,
    so the latent it differences against has to live in the same token layout.
    """

    pt, ph, pw = (int(value) for value in patch_size)
    n, c, ot, oh, ow = latent.shape
    if ot % pt or oh % ph or ow % pw:
        raise ValueError(f"latent {tuple(latent.shape)} does not tile with patch {patch_size}")
    tt, th, tw = ot // pt, oh // ph, ow // pw
    value = latent.reshape(n, c, tt, pt, th, ph, tw, pw)
    # (n c t o h p w q) -> (n t h w c o p q), mirroring the unpatchify einsum.
    value = value.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return value.reshape(n, tt * th * tw, c * pt * ph * pw)


@dataclass(frozen=True)
class MeanCacheDecisionRecord:
    step: int
    action: str
    reason: str
    original_block_calls: int
    sigma_t: float
    sigma_s: float
    requested_jvp_span: int
    actual_jvp_span: int
    jvp_correction_used: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MeanCacheVelocityAdapter:
    """Skip Tencent blocks and substitute MeanCache's velocity at the final layer.

    Sibling of `L2POutputAdapter`: same block-skipping and same injection point,
    but the payload is an average-velocity integration that needs the current
    latent and the scheduler sigmas, so the method is asked for a decision with
    the patchified latent and reads sigmas through its scheduler provider.
    """

    def __init__(self, transformer: Any, method: MeanCacheMethod):
        self.transformer = transformer
        self.method = method
        self.decisions: list[MeanCacheDecisionRecord] = []
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self) -> None:
        self.method.reset()
        self.decisions.clear()
        self._current_decision: MethodDecision | None = None
        self._current_full = True
        self._latent_tokens: torch.Tensor | None = None
        self._block_calls = 0

    def __enter__(self) -> "MeanCacheVelocityAdapter":
        if self._installed:
            raise RuntimeError("adapter already installed")
        if not self.transformer.double_blocks or not self.transformer.single_blocks:
            raise ValueError("HunyuanVideo MeanCache requires double and single blocks")
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._transformer_pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._transformer_post, with_kwargs=True)
        )
        for index, block in enumerate(self.transformer.double_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._double_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))
        for index, block in enumerate(self.transformer.single_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._single_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))

        final_layer = self.transformer.final_layer

        def final_forward(module: Any, *args: Any, **kwargs: Any):
            return self._final_forward(module, *args, **kwargs)

        self._patches.append((final_layer, _set_instance_forward(final_layer, final_forward)))
        self._installed = True
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        for module, state in reversed(self._patches):
            _restore_instance_forward(module, state)
        self._handles.clear()
        self._patches.clear()
        self._installed = False

    def _transformer_pre(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        x = args[0] if args else kwargs["x"]
        self._latent_tokens = patchify(x.detach(), self.transformer.patch_size)
        self._block_calls = 0

    def _transformer_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        decision = self._current_decision
        if decision is None:
            raise RuntimeError("MeanCache adapter did not enter the first double block")
        fields = dict(self.method.last_payload_fields)
        if not fields:
            raise RuntimeError("MeanCache adapter did not reach the final layer")
        self.decisions.append(
            MeanCacheDecisionRecord(
                step=len(self.decisions),
                action="full" if self._current_full else "cache",
                reason=decision.reason,
                original_block_calls=self._block_calls,
                **fields,
            )
        )
        return output

    def _double_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        if index == 0:
            if self._latent_tokens is None:
                raise RuntimeError("transformer pre-hook did not capture the latent")
            self._current_decision = self.method.decide(latent=self._latent_tokens)
            self._current_full = self._current_decision.full
        if not self._current_full:
            return args[0], args[1]
        self._block_calls += 1
        return original(*args, **kwargs)

    def _single_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        offset = len(self.transformer.double_blocks) + index
        original = self._patches[offset][1][1]
        if not self._current_full:
            return args[0]
        self._block_calls += 1
        return original(*args, **kwargs)

    def _final_forward(self, _module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        final_index = len(self._patches) - 1
        original = self._patches[final_index][1][1]
        if self._current_full:
            return self.method.final_output(original(*args, **kwargs))
        return self.method.final_output()
