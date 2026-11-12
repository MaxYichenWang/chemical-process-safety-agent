"""Run a model sweep: for each run, set per-role provider/model env vars, execute an
eval tier, and diff the result against the baseline with improvement_analyzer.py.

Requires ChromaDB + Parquet (it executes the real pipeline) → run on the pipeline
machine. Use --dry-run anywhere to print the plan without calling the evaluator.

Usage:
    # Inspect the plan (no eval executed, no DB needed):
    python scripts/reusable/run_model_sweep.py --config configs/model_sweep.example.yaml --dry-run

    # Execute every run at the tier configured in the sweep's defaults:
    python scripts/reusable/run_model_sweep.py --config configs/model_sweep.example.yaml

    # Override the configured tier without editing YAML:
    python scripts/reusable/run_model_sweep.py --config configs/model_sweep.example.yaml --tier T0

    # Execute a single run:
    python scripts/reusable/run_model_sweep.py --config configs/model_sweep.example.yaml --run strong_supervisor --tier T0

    # Include the adversarial dataset when executing the underlying evaluator:
    python scripts/reusable/run_model_sweep.py --config configs/model_sweep.example.yaml --run strong_supervisor --tier T0 --include-adversarial
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MODEL_SWEEP_PATH = PROJECT_ROOT / "src" / "evaluation" / "model_sweep.py"
_MODEL_SWEEP = None


def _model_sweep_helpers():
    """Load the lightweight sweep helper without importing src.evaluation."""
    global _MODEL_SWEEP
    if _MODEL_SWEEP is None:
        spec = importlib.util.spec_from_file_location(
            "_cps_model_sweep",
            _MODEL_SWEEP_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load model sweep helper from {_MODEL_SWEEP_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_SWEEP = module
    return _MODEL_SWEEP.build_run_env, _MODEL_SWEEP.load_sweep

_TIER_SAMPLES = {"T0": 1, "T1": 3, "T2": None}  # None => full dataset


def _resolve_eval_defaults(
    defaults: dict,
    tier_override: str | None = None,
    sample_override: int | None = None,
    include_adversarial_override: bool | None = None,
    max_concurrent_override: int | None = None,
) -> dict:
    """Resolve tier/sample/concurrency settings for one sweep invocation.

    CLI tier overrides deliberately reset the sample count to that tier's
    default. Otherwise a config like `tier: T1` + `sample_per_source: 3` would
    keep sampling 3 per source even when the user asks for `--tier T0`.
    """
    resolved = dict(defaults)
    if tier_override:
        tier = tier_override.upper()
        if tier not in _TIER_SAMPLES:
            raise ValueError(
                f"unknown tier '{tier_override}' (expected one of {sorted(_TIER_SAMPLES)})"
            )
        resolved["tier"] = tier
        resolved["sample_per_source"] = _TIER_SAMPLES[tier]

    if sample_override is not None:
        resolved["sample_per_source"] = sample_override

    if include_adversarial_override is not None:
        resolved["include_adversarial"] = include_adversarial_override

    if max_concurrent_override is not None:
        resolved["max_concurrent"] = max_concurrent_override
    return resolved


def _eval_cmd(defaults: dict) -> list[str]:
    tier = str(defaults.get("tier", "T0")).upper()
    sample = defaults.get("sample_per_source", _TIER_SAMPLES.get(tier, 3))
    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.evaluators.end_to_end_evaluator",
        "--max-concurrent",
        str(defaults.get("max_concurrent", 10)),
        "--sample-seed",
        str(defaults.get("seed", 42)),
        "--save-logs",
    ]
    if sample is not None:
        cmd += ["--sample-per-source", str(sample)]
    if defaults.get("include_adversarial", False):
        cmd.append("--include-adversarial")
    return cmd


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", help="Run only this named run (default: all).")
    ap.add_argument(
        "--tier",
        type=str.upper,
        choices=sorted(_TIER_SAMPLES),
        help=(
            "Override defaults.tier for this invocation. "
            "T0 uses --sample-per-source 1, T1 uses 3, T2 uses the full dataset."
        ),
    )
    ap.add_argument(
        "--sample-per-source",
        type=int,
        help=(
            "Override the tier's sample count. Omit for tier defaults; "
            "T2 omits --sample-per-source unless this is set."
        ),
    )
    ap.add_argument(
        "--include-adversarial",
        action="store_true",
        default=None,
        help=(
            "Forward --include-adversarial to the evaluator. Can also be set "
            "with defaults.include_adversarial: true in the sweep config."
        ),
    )
    ap.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=(
            "Override defaults.max_concurrent passed to the evaluator. "
            "Defaults to value in sweep config or 10."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print env + command for each run without executing.",
    )
    args = ap.parse_args()

    build_run_env, load_sweep = _model_sweep_helpers()
    sweep = load_sweep(args.config)
    defaults = _resolve_eval_defaults(
        sweep.get("defaults", {}) or {},
        tier_override=args.tier,
        sample_override=args.sample_per_source,
        include_adversarial_override=args.include_adversarial,
        max_concurrent_override=args.max_concurrent,
    )
    baseline_results = defaults.get("baseline_results")
    if baseline_results and not Path(baseline_results).is_file():
        ap.error(f"baseline results not found: {baseline_results}")
    names = [args.run] if args.run else [r["name"] for r in sweep["runs"]]
    exit_code = 0

    for name in names:
        env_overrides = build_run_env(sweep, name)
        cmd = _eval_cmd(defaults)
        run_id = f"eval_sweep_{uuid4().hex}"
        cmd += ["--run-id", run_id]
        print(f"\n{'='*70}\nRUN: {name}\n{'='*70}")
        for k in sorted(env_overrides):
            print(f"  {k}={env_overrides[k]}")
        print(f"  CMD: {' '.join(cmd)}")

        # Make the self-preference-bias risk visible in the report header
        # itself, not just in a doc -- a run whose judge is the same model as
        # the pipeline under test is not an independent measurement.
        judge_model = env_overrides.get("EVAL_LLM_MODEL")
        if judge_model:
            pipeline_model = env_overrides.get("AGENT_SUPERVISOR_MODEL")
            same_model = judge_model == pipeline_model
            print(
                f"  judge_is_same_model: {same_model} "
                f"(judge={judge_model}, pipeline_supervisor={pipeline_model})"
            )
        else:
            print("  judge_is_same_model: N/A (this run doesn't override the judge; uses env/default)")

        if args.dry_run:
            continue

        run_env = {**os.environ, **env_overrides}
        run_env.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            proc = subprocess.run(cmd, env=run_env)
        except OSError as exc:
            print(f"  !! could not start eval for run '{name}': {exc}")
            exit_code = 1
            continue
        if proc.returncode != 0:
            print(f"  !! eval failed for run '{name}' (exit {proc.returncode})")
            exit_code = 1
            continue

        results_dir = Path(run_env.get("EVAL_DATASET_DIR", "data/evaluation")) / "results"
        result_path = results_dir / f"{run_id}_results.json"
        if not result_path.is_file():
            print(f"  !! eval produced no results for run '{name}': {result_path}")
            exit_code = 1
            continue

        if baseline_results:
            from src.evaluation.analyzers.improvement_analyzer import (
                ImprovementAnalyzer,
            )

            analyzer = ImprovementAnalyzer()
            report = analyzer.compare_runs(Path(baseline_results), result_path)
            out = results_dir / f"sweep_{name}_vs_baseline.md"
            out.write_text(analyzer.generate_report(report))
            print(f"  wrote {out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
