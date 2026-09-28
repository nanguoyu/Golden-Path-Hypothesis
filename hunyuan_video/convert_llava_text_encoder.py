#!/usr/bin/env python3
"""Extract HunyuanVideo's LLM text encoder from the llava checkpoint.

The official tree ships `hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py`
for this, and it is the file the official README tells you to run. It reaches
for `LlavaForConditionalGeneration.language_model`, an attribute that existed
when it was written and does not on transformers 5.x, where the layout is

    LlavaForConditionalGeneration
      .model              LlavaModel
        .language_model   LlamaModel        <- what the text encoder is
        .vision_tower
        .multi_modal_projector
      .lm_head

so the official script exits with AttributeError before writing anything.

USE THE OFFICIAL SCRIPT WHERE IT RUNS. On the transformers the official
`requirements.txt` pins (4.46.3) the attribute is there and the official
script is the faithful path; this file is for an environment on 5.x, such as
the one the FLUX and Qwen work uses. It is a separate file rather than a patch
because `hunyuan_video/backend.py` refuses to run against a Tencent worktree
that is not byte-identical to the pinned commit, so editing the official
script would take the whole harness down with it.

What it writes is what `hyvideo/text_encoder/__init__.py` loads: that file
calls `AutoModel.from_pretrained` on the output directory, which resolves to
the BASE `LlamaModel` and drops any `lm_head` it finds. On 4.x the text tower
is a `LlamaForCausalLM` and the head is discarded at load time, exactly as the
official script leaves it; on 5.x it is already a `LlamaModel`, so the same
encoder is written without the dead gigabyte.

Whichever wrote it, the directory must be written by the SAME transformers
that will read it: the tokenizer config names a class, and 5.x writes a name
4.x cannot resolve.

  python hunyuan_video/convert_llava_text_encoder.py \
      --input_dir  ~/hyv_ckpts/llava-llama-3-8b-v1_1-transformers \
      --output_dir ~/hyv_ckpts/text_encoder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor, LlavaForConditionalGeneration

# the official script's dtype; the sampler casts to its own precision later
DTYPE = torch.float16
PROBE_TOKENS = [[1, 450, 4996, 17354, 1701, 29916]]


def resolve_language_model(model: LlavaForConditionalGeneration) -> torch.nn.Module:
    """The llava wrapper's text tower, wherever this transformers keeps it.

    Both layouts are accepted rather than pinning one, because the checkpoint
    is the fixed thing here and the library is not: an environment rebuild that
    moves the attribute back should not silently write a different encoder.
    """
    for path in (("language_model",), ("model", "language_model")):
        node: object = model
        for name in path:
            node = getattr(node, name, None)
            if node is None:
                break
        else:
            return node  # type: ignore[return-value]
    raise SystemExit(
        "no language model on this LlavaForConditionalGeneration: tried .language_model and "
        f".model.language_model, top level holds {[n for n, _ in model.named_children()]}"
    )


def load_in(cls, path: Path, device: str):
    """`from_pretrained` at fp16, under either spelling of the dtype argument.

    transformers renamed `torch_dtype` to `dtype` at 5.0. Both spellings are
    tried rather than one being pinned because this conversion has to run in
    whichever environment will later READ its output: a directory written by
    one major version carries class names the other cannot resolve.
    """
    for kwarg in ("dtype", "torch_dtype"):
        try:
            return cls.from_pretrained(path, low_cpu_mem_usage=True,
                                       **{kwarg: DTYPE}).to(device).eval()
        except TypeError:
            continue
    raise SystemExit(f"{cls.__name__}.from_pretrained accepts neither dtype nor torch_dtype")


def final_hidden_state(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """The text tower's last hidden state, whichever wrapper it is inside.

    Which wrapper it is depends on the transformers version, and the two do not
    answer the same call: the 4.x `LlamaForCausalLM` returns logits and has no
    `last_hidden_state`, so reading that attribute would compare the encoder
    against nothing on exactly the version this repo pins.
    """
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    direct = getattr(out, "last_hidden_state", None)
    return out.hidden_states[-1] if direct is None else direct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="the downloaded llava-llama-3-8b-v1_1-transformers directory")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="where the text encoder goes; the sampler expects <model_base>/text_encoder")
    parser.add_argument("--device", default="cpu",
                        help="device to hold the model on while saving (default %(default)s)")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.input_dir)
    model = load_in(LlavaForConditionalGeneration, args.input_dir, args.device)
    language_model = resolve_language_model(model)
    print(f"[convert] {type(model).__name__} -> {type(language_model).__name__}, "
          f"{sum(p.numel() for p in language_model.parameters()) / 1e9:.2f} B parameters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    language_model.save_pretrained(args.output_dir)
    processor.tokenizer.save_pretrained(args.output_dir)
    print(f"[convert] wrote {args.output_dir}")

    ########################
    # Reload through the same call the sampler makes and check the hidden
    # states agree. A directory that loads but answers differently would only
    # surface as quietly wrong videos, and every later comparison would inherit
    # it -- so the encoder is checked here, once, against the model it came from.
    ########################
    ids = torch.tensor(PROBE_TOKENS, device=args.device)
    want = final_hidden_state(language_model, ids)
    del model, language_model
    reloaded = load_in(AutoModel, args.output_dir, args.device)
    print(f"[convert] reloaded as {type(reloaded).__name__}")
    got = final_hidden_state(reloaded, ids)
    if got.shape != want.shape:
        raise SystemExit(f"reloaded encoder changed shape: {tuple(got.shape)} != {tuple(want.shape)}")
    max_abs = float((got.float() - want.float()).abs().max())
    print(f"[convert] hidden-state max abs difference on reload: {max_abs:.3e}")
    if max_abs != 0.0:
        raise SystemExit("reloaded encoder does not reproduce the source hidden states exactly")
    print("[convert] OK")


if __name__ == "__main__":
    main()
