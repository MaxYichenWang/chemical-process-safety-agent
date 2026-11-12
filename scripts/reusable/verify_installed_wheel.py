"""Check an installed wheel outside its checkout using synthetic inputs only.

Run with the wheel environment's Python, without an editable installation:
    /path/to/venv/bin/python scripts/reusable/verify_installed_wheel.py \
        --fixture /absolute/path/to/tests/fixtures/sample_eval_dataset.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.fixture.resolve().read_text(encoding="utf-8"))
    if len(dataset) < 2:
        parser.error("The synthetic fixture must contain at least two entries")

    # Set output configuration before importing any package modules. The fixture
    # and responses are synthetic; no graph or judge client is constructed.
    with tempfile.TemporaryDirectory(prefix="cps-wheel-smoke-") as directory:
        root = Path(directory)
        previous_cwd = Path.cwd()
        previous_output = os.environ.get("EVAL_DATASET_DIR")
        os.environ["EVAL_DATASET_DIR"] = str(root / "evaluation")
        os.chdir(root)
        try:
            from src.evaluation import run
            from src.evaluation.scoring.contract import sha256_json

            if not Path(run.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError("Package resolved outside the wheel environment")
            # An installed-but-incompatible RAGAS must fail, rather than skip.
            importlib.import_module("ragas.metrics.collections")
            for name in ("index.html", "app.js", "styles.css"):
                assert files("src.web").joinpath("static", name).is_file(), name
            assert files("src.web").joinpath("__main__.py").is_file()
            assert files("scripts").joinpath("demo_agentic_rag.py").is_file()

            command = Path(sys.executable).with_name("cps-eval")
            subprocess.run([str(command), "--help"], check=True, capture_output=True)
            path = root / "dataset.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            subprocess.run(
                [str(command), "generate", "--dataset", str(path), "--dry-run"],
                check=True, capture_output=True,
            )
            generation = {
                "run_metadata": {
                    "dataset_path": str(path),
                    "dataset_sha256": sha256_json(dataset),
                },
                "pipeline_results": [
                    {
                        "query": entry["query"],
                        "final_response": "Synthetic local response.",
                        "context_gathered": [], "agents_called": [],
                        "planned_agents": [], "llm_calls": [],
                    }
                    for entry in dataset
                ],
            }
            destination = run.generation_path("wheel-smoke")
            destination.parent.mkdir(parents=True)
            destination.write_text(json.dumps(generation), encoding="utf-8")
            assert run.main(["score", "--run-id", "wheel-smoke", "--layer", "deterministic", "--no-save-logs"]) == 0
            assert run.main(["aggregate", "--run-id", "wheel-smoke"]) == 0
            report = run.report_path("wheel-smoke").read_bytes()
            path.write_text(json.dumps(dataset[::-1]), encoding="utf-8")
            assert run.main(["score", "--run-id", "wheel-smoke", "--layer", "deterministic", "--no-save-logs"]) == 1
            assert run.main(["aggregate", "--run-id", "wheel-smoke"]) == 1
            assert run.report_path("wheel-smoke").read_bytes() == report
        finally:
            os.chdir(previous_cwd)
            if previous_output is None:
                os.environ.pop("EVAL_DATASET_DIR", None)
            else:
                os.environ["EVAL_DATASET_DIR"] = previous_output
    print("Installed wheel: imports, assets, command, scoring and integrity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
