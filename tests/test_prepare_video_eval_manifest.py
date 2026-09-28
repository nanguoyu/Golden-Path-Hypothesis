import json
import pytest
from scripts.prepare_video_eval_manifest import convert


def test_preserves_prompt_order_and_newlines(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "output.jsonl"
    items = [{"prompt_id": "a", "prompt": "Two lines\ninside a prompt"},
             {"prompt_id": "b", "prompt": "Second prompt"}]
    source.write_text(json.dumps({"items": items}))
    assert convert(source, output) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_idx"] for row in rows] == [0, 1]
    assert [row["prompt"] for row in rows] == [item["prompt"] for item in items]


@pytest.mark.parametrize("items", [[], [{"prompt_id": "a", "prompt": ""}],
    [{"prompt_id": "a", "prompt": "one"}, {"prompt_id": "a", "prompt": "two"}],
    [{"prompt_id": "a", "prompt": "one", "prompt_idx": 2}]])
def test_rejects_invalid_manifest(tmp_path, items):
    source = tmp_path / "input.json"
    output = tmp_path / "output.jsonl"
    source.write_text(json.dumps({"items": items}))
    with pytest.raises(ValueError):
        convert(source, output)
    assert not output.exists()
