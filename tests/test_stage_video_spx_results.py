import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage_video_spx_results import parse_cell_name, stage
from RUN.video_spx.spx_cells import cell_name


class VideoSPXStagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.metrics = self.root / "metrics"
        self.metrics.mkdir()
        self.output = self.root / "out.tsv.gz"

    def fixture(self, backbone="wan21", dataset="penguin599", base_seed=54, idx=3):
        cell = f"sharedxreuse_{dataset}_K29_s{base_seed}"
        generation = self.root / "generation" / cell
        generation.mkdir(parents=True)
        prompt = "A bird.\nIt flies."
        decision = {
            "schema": f"{backbone}.baseline_screen_decisions.v1",
            "prompt_idx": idx, "prompt_id": f"{dataset}-prompt-{idx}",
            "prompt": prompt, "seed": base_seed + idx,
            "actual_cache_count": 29, "num_steps": 50,
        }
        per_video = {
            "idx": idx, "video": f"video_{idx:05d}.mp4",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "psnr": 25.5, "ssim": 0.91, "lpips": 0.12,
            "temporal_lpips_delta": -0.02,
        }
        metrics = {
            "schema": "video_pairwise_metrics.v2", "backbone": backbone,
            "acc": str(generation), "n_pairs": 1, "frame_indices": "all",
            "per_video": [per_video],
        }
        metric_path = self.metrics / f"{cell}.json"
        decision_path = generation / f"decisions_{idx:05d}.json"
        metric_path.write_text(json.dumps(metrics))
        decision_path.write_text(json.dumps(decision))
        return metric_path, decision_path, metrics, decision

    def output_rows(self):
        with gzip.open(self.output, "rt") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def baseline_fixture(self):
        path, decision_path, data, decision = self.fixture()
        cell = "seacache_penguin599_K29_s54"
        directory = decision_path.parent.with_name(cell)
        decision_path.parent.rename(directory)
        decision.update(mode="seacache", dataset="penguin599", budget="K29")
        decision_path = directory / decision_path.name
        decision_path.write_text(json.dumps(decision))
        data["acc"] = str(directory)
        path.unlink()
        path = self.metrics / f"{cell}.json"
        path.write_text(json.dumps(data))
        return path, decision_path, data, decision

    def test_both_backbones_keep_base_seed_actual_seed_and_prompt_id(self):
        for backbone in ("wan21", "hunyuan_video"):
            with self.subTest(backbone=backbone):
                metric_path, _, data, _ = self.fixture(backbone=backbone)
                self.assertEqual(stage(backbone, self.metrics, self.output), 1)
                row = self.output_rows()[0]
                self.assertEqual((row["seed"], row["actual_seed"], row["prompt_idx"]),
                                 ("54", "57", "3"))
                self.assertEqual(row["prompt_id"], "penguin599-prompt-3")
                self.assertEqual(row["cache_count_realized"], "29")
                self.assertEqual(float(row["temporal_delta"]), -0.02)
                metric_path.unlink()
                # Use a different temporary generation path in the next subtest.
                Path(data["acc"]).rename(Path(data["acc"] + f"-{backbone}-done"))

    def test_off_budget_count_is_preserved_and_relocation_works(self):
        path, decision_path, data, decision = self.fixture()
        decision["actual_cache_count"] = 28
        decision_path.write_text(json.dumps(decision))
        data["acc"] = "/unavailable/" + Path(data["acc"]).name
        path.write_text(json.dumps(data))
        stage("wan21", self.metrics, self.output, self.root / "generation")
        self.assertEqual(self.output_rows()[0]["cache_count_realized"], "28")

    def test_missing_generation_input_fails_without_writing(self):
        _, decision_path, _, _ = self.fixture()
        decision_path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing required input"):
            stage("wan21", self.metrics, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_metric_is_not_filled_with_zero(self):
        path, _, data, _ = self.fixture()
        del data["per_video"][0]["ssim"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "missing required field 'ssim'"):
            stage("wan21", self.metrics, self.output)

    def test_actual_seed_mismatch_fails(self):
        _, path, _, decision = self.fixture()
        decision["seed"] = 54
        path.write_text(json.dumps(decision))
        with self.assertRaisesRegex(ValueError, "seed is not cell base seed plus idx"):
            stage("wan21", self.metrics, self.output)

    def test_duplicate_prompt_idx_fails(self):
        path, _, data, _ = self.fixture()
        data["per_video"].append(dict(data["per_video"][0]))
        data["n_pairs"] = 2
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "duplicate evaluated video"):
            stage("wan21", self.metrics, self.output)

    def test_prompt_mismatch_fails(self):
        _, path, _, decision = self.fixture()
        decision["prompt"] = "An unrelated prompt."
        path.write_text(json.dumps(decision))
        with self.assertRaisesRegex(ValueError, "prompt differs from decisions"):
            stage("wan21", self.metrics, self.output)

    def test_missing_realized_count_fails(self):
        _, path, _, decision = self.fixture()
        del decision["actual_cache_count"]
        path.write_text(json.dumps(decision))
        with self.assertRaisesRegex(ValueError, "missing required field 'actual_cache_count'"):
            stage("wan21", self.metrics, self.output)

    def test_same_index_in_different_datasets_is_not_merged(self):
        self.fixture()
        self.fixture(dataset="vbench944", base_seed=42)
        self.assertEqual(stage("wan21", self.metrics, self.output), 2)
        self.assertEqual({row["dataset"] for row in self.output_rows()},
                         {"penguin599", "vbench944"})

    def test_producer_cell_names_round_trip(self):
        for policy in ("reuse", "mean_vel", "mean_vel_global", "di_two_anchor"):
            name = cell_name("shared", policy, "vbench944", "K41", 42)
            self.assertEqual(parse_cell_name(name),
                             ("shared", policy, "vbench944", "K41", 42))

    def test_baseline_output_has_method_and_matching_seeds(self):
        self.baseline_fixture()
        stage("wan21", self.metrics, self.output, kind="baseline")
        row = self.output_rows()[0]
        self.assertEqual(row["method"], "seacache")
        self.assertNotIn("payload", row)
        self.assertEqual((row["seed"], row["actual_seed"]), ("54", "57"))
        self.assertEqual(row["prompt_id"], "penguin599-prompt-3")

    def test_baseline_identity_mismatch_fails(self):
        _, path, _, decision = self.baseline_fixture()
        decision["mode"] = "teacache"
        path.write_text(json.dumps(decision))
        with self.assertRaisesRegex(ValueError, "mode differs from cell filename"):
            stage("wan21", self.metrics, self.output, kind="baseline")

    def test_baseline_infinite_psnr_is_preserved_for_downstream_filtering(self):
        path, _, data, _ = self.baseline_fixture()
        data["per_video"][0]["psnr"] = float("inf")
        path.write_text(json.dumps(data))
        stage("wan21", self.metrics, self.output, kind="baseline")
        self.assertEqual(float(self.output_rows()[0]["psnr"]), float("inf"))


if __name__ == "__main__":
    unittest.main()
