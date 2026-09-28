import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis.sp_cross import PAYLOADS, W1_ROWS


class OptionalNativeReferenceTests(unittest.TestCase):
    def test_four_by_five_grid_runs_without_native_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, schedule in enumerate(W1_ROWS):
                for j, payload in enumerate(PAYLOADS):
                    cell = root / "cells" / "flux" / "k41" / f"{schedule}x{payload}_s42"
                    cell.mkdir(parents=True)
                    (cell / "metrics.json").write_text(json.dumps({
                        "summary": {"psnr": {"mean": 20.0 + i + 0.1 * j, "n": 2}}
                    }))
            output = root / "report.json"
            result = subprocess.run([
                sys.executable, "analysis/sp_cross.py", "--root", str(root / "cells"),
                "--output", str(output),
            ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["panels"][0]["n_cells"], 20)
            self.assertEqual(len(report["panels"][0]["cells"]), 20)


if __name__ == "__main__":
    unittest.main()
