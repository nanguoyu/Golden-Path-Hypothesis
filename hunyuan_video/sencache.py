"""SenCache gate wiring for the pinned Tencent HunyuanVideo backend."""

from __future__ import annotations

from typing import Any

from hunyuan_video.adapter import CoarseBackboneAdapter


class SenCacheAdapter(CoarseBackboneAdapter):
    """Coarse residual reuse whose gate reads the pre-``img_in`` latent and timestep.

    SenCache scores ``x`` and ``t`` exactly as the transformer receives them
    (``reference/hunyuan_video/code/hyvideo/modules/models.py:595-596``), and
    neither survives to double block 0 where ``decide()`` runs — ``img`` there is
    already patch-embedded and ``vec`` already carries the text and guidance
    modulation.  The transformer pre-hook therefore hands both to the gate
    before the first block dispatches.
    """

    def _transformer_pre(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        super()._transformer_pre(module, args, kwargs)
        x = args[0] if args else kwargs["x"]
        t = args[1] if len(args) > 1 else kwargs["t"]
        self.method.observe_input(x, t)
