from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from flux.meancache_exact import FluxMeanCacheAdapter


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, **_kwargs):
        self.calls += 1
        return hidden_states * 0.25 + 1.0


def test_meancache_skips_model_call_and_records_jvp_payload() -> None:
    transformer = _Transformer()
    pipe = SimpleNamespace(
        transformer=transformer,
        scheduler=SimpleNamespace(sigmas=torch.linspace(1.0, 0.0, 6)),
    )
    adapter = FluxMeanCacheAdapter(
        pipe,
        cache_steps=(3,),
        num_steps=5,
        jvp_span=2,
    )
    adapter.install()
    outputs = []
    for step in range(5):
        outputs.append(
            transformer(hidden_states=torch.full((1, 2, 2), float(step)))
        )

    assert transformer.calls == 4
    assert [row["action"] for row in adapter.records] == [
        "full",
        "full",
        "full",
        "cache",
        "full",
    ]
    assert adapter.records[3]["jvp_correction_used"]
    assert torch.isfinite(outputs[3]).all()
    assert adapter.decisions()["summary"]["n_cached"] == 1
    adapter.restore()
    assert "forward" not in transformer.__dict__
