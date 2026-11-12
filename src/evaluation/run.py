"""The staged eval CLI: ``generate`` / ``score`` / ``aggregate``.

Implements a four-stage architecture: generation is separated from scoring,
scoring is separated by layer (deterministic / deepeval / ragas), and
``aggregate`` merges whatever score files exist into one ``results.json`` +
``report.md``. On-disk layout this module reads and writes:

    data/evaluation/runs/<run_id>/
      generation.json                   # run_metadata + pipeline_results
      scores/
        deterministic.json
        deepeval.<judge>.json
        ragas.<judge>.json
      results.json                      # aggregate output
      report.md

Only ``generate`` (real, non-dry-run) needs the live pipeline (langchain,
ChromaDB, a provider API key) -- those imports are deferred inside
``cmd_generate`` so ``--help`` and every ``--dry-run`` invocation work with
none of that available. ``score --layer deterministic`` and ``aggregate``
need no judge/key/corpus at all.

Usage:
    python -m src.evaluation.run generate --dataset ... --run-id my_run --save-logs
    python -m src.evaluation.run score --run-id my_run --layer deterministic
    python -m src.evaluation.run score --run-id my_run --layer deepeval --judge deepseek-v4-flash --max-concurrency 8 --save-logs
    python -m src.evaluation.run score --run-id my_run --layer ragas --judge deepseek-v4-flash --max-concurrency 8 --save-logs
    python -m src.evaluation.run aggregate --run-id my_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.config import (
    ADVERSARIAL_DATASET_FILE,
    CHROMADB_DIR,
    EVAL_DATASET_DIR,
    EVAL_DATASET_FILE,
    EVAL_MAX_CONCURRENT,
)
from src.evaluation.scoring.contract import (
    DatasetMismatchError,
    GenerationMismatchError,
    build_score_file,
    sanitize_judge_id,
    sanitize_tag,
    sha256_json,
    utc_now_iso,
    verify_generation_sha,
)
from src.evaluation.utils import (
    generate_run_id,
    finalize_pipeline_results,
    load_eval_dataset,
    sample_dataset_per_source,
)

logger = logging.getLogger(__name__)

RUNS_DIR = EVAL_DATASET_DIR / "runs"


# ---------------------------------------------------------------------------
# On-disk layout
# ---------------------------------------------------------------------------


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def generation_path(run_id: str) -> Path:
    return run_dir(run_id) / "generation.json"


def scores_dir(run_id: str) -> Path:
    return run_dir(run_id) / "scores"


def results_path(run_id: str) -> Path:
    return run_dir(run_id) / "results.json"


def report_path(run_id: str) -> Path:
    return run_dir(run_id) / "report.md"


def score_file_name(
    layer: str, judge_id: Optional[str] = None, tag: Optional[str] = None
) -> str:
    if layer == "deterministic":
        base = "deterministic"
    else:
        if not judge_id:
            raise ValueError(f"layer '{layer}' requires a judge id")
        base = f"{layer}.{sanitize_judge_id(judge_id)}"
    tag_part = sanitize_tag(tag)
    if tag_part:
        base = f"{base}.{tag_part}"
    return f"{base}.json"


def cache_file_path(
    run_id: str, layer: str, judge_id: str, tag: Optional[str] = None
) -> Path:
    """Per-(run, layer, judge, tag) resumable scoring cache, kept out of ``scores/*.json``
    globbing (``aggregate`` only reads files matching the score-file contract shape).

    ``tag`` is part of the cache identity so re-scoring the same judge under a
    different setting (e.g. a new temperature) does **not** silently resume from
    the previous setting's cached per-entry scores."""
    name = f"{layer}.{sanitize_judge_id(judge_id)}"
    tag_part = sanitize_tag(tag)
    if tag_part:
        name = f"{name}.{tag_part}"
    return scores_dir(run_id) / ".cache" / f"{name}.json"


def find_generation_file(run_id: str) -> Optional[Path]:
    """Return the on-disk generation file for *run_id*.

    Prefers the plain ``generation.json`` a fresh ``generate`` writes,
    falling back to a gzip-compressed ``generation.json.gz`` -- the shape
    the legacy migrator (``historical run importer``)
    emits for large historical runs (git-lfs is used for these rather than
    committing e.g. a 33.7 MB plain JSON file).
    """
    plain = generation_path(run_id)
    if plain.exists():
        return plain
    gz = plain.with_suffix(plain.suffix + ".gz")
    if gz.exists():
        return gz
    return None


def load_generation(run_id: str) -> Tuple[Dict[str, Any], Path]:
    """Load and parse a run's generation file, transparently decompressing ``.gz``.

    Returns ``(generation_dict, path_used)``. Callers should hash the
    *parsed* dict (:func:`~src.evaluation.scoring.contract.sha256_json`),
    not the file's raw bytes -- gzip's own framing (mtime, compression
    level) is not deterministic across writers, so a content hash is the
    only stable ``generation_sha256`` that survives (re-)compression.
    """
    path = find_generation_file(run_id)
    if path is None:
        raise FileNotFoundError(
            f"no generation.json or generation.json.gz found for run_id={run_id!r} under {run_dir(run_id)}"
        )
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh), path
    with open(path, encoding="utf-8") as fh:
        return json.load(fh), path


# ---------------------------------------------------------------------------
# Dataset resolution -- shared by generate (to build pipeline_results) and
# score/aggregate (to reconstruct the exact same dataset from run_metadata,
# since generation.json stores only run_metadata + pipeline_results, not the
# dataset itself).
# ---------------------------------------------------------------------------


def resolve_dataset(
    dataset_path: Path,
    *,
    include_adversarial: bool,
    sample_size_per_source: Optional[int],
    sample_seed: Optional[int],
) -> List[Dict[str, Any]]:
    """Reproduce the exact dataset a generation was run against.

    Mirrors ``EndToEndEvaluator.evaluate()``'s load -> adversarial-append ->
    sample pipeline via the same shared helpers, so a generation's
    ``run_metadata`` (dataset_path/include_adversarial/sample_size_per_source/
    sample_seed) is sufficient to deterministically reconstruct the dataset
    at score time without re-storing it in ``generation.json``.
    """
    dataset = load_eval_dataset(Path(dataset_path))
    if include_adversarial:
        try:
            dataset = list(dataset) + load_eval_dataset(ADVERSARIAL_DATASET_FILE)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - matches evaluate()'s own warn-and-continue
            logger.warning("Could not load adversarial dataset: %s", exc)
    if sample_size_per_source is not None:
        dataset = sample_dataset_per_source(
            dataset, sample_size_per_source, seed=sample_seed
        )
    return dataset


def rebind_embedded_dataset(
    embedded: List[Dict[str, Any]], current: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Re-express an embedded dataset snapshot as today's entries, in snapshot order.

    A migrated run embeds its dataset verbatim, which freezes it: a later repair
    (``l1_applicable``, ``answer_gradable``, ``defects``, ``domain_relevance``)
    never reaches it, so it aggregates with untouched denominators while every
    non-embedded run in the same table is re-quantified. Comparing the two then
    attributes a measurement fix to the architecture.

    Re-binding is only safe when the snapshot and the current dataset are the
    *same questions*. The join key is ``query``, and it is deliberately strict:
    equal lengths, no duplicate queries on either side, and every snapshot query
    present. Anything else raises -- a partial re-bind would silently score
    responses against another question's reference answer, which is worse than
    not re-binding at all.

    Entry *content* may legitimately differ (that is the point: the repaired
    ``expected_answer`` and the new annotations are what we want to pick up).
    Only the question identity must be preserved.
    """
    if len(embedded) != len(current):
        raise ValueError(
            f"cannot re-bind: snapshot has {len(embedded)} entries, current dataset has "
            f"{len(current)} -- these are different question sets, not a repair of one"
        )

    by_query: Dict[str, Dict[str, Any]] = {}
    for entry in current:
        query = entry.get("query")
        if query is None:
            raise ValueError(
                "cannot re-bind: a current dataset entry has no 'query' key"
            )
        if query in by_query:
            raise ValueError(
                f"cannot re-bind: current dataset has duplicate query {query[:80]!r}, "
                "so the snapshot-to-current mapping is ambiguous"
            )
        by_query[query] = entry

    rebound: List[Dict[str, Any]] = []
    missing: List[str] = []
    seen: set = set()
    for entry in embedded:
        query = entry.get("query")
        if query in seen:
            raise ValueError(
                f"cannot re-bind: snapshot has duplicate query {str(query)[:80]!r}"
            )
        seen.add(query)
        match = by_query.get(query)
        if match is None:
            missing.append(str(query)[:80])
            continue
        rebound.append(match)

    if missing:
        raise ValueError(
            f"cannot re-bind: {len(missing)} snapshot query/queries are absent from the "
            f"current dataset (first: {missing[0]!r}) -- the question set changed, so this "
            "run is not comparable with runs scored against the current dataset"
        )
    return rebound


def current_dataset_for_rebind(run_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load today's dataset for a migrated run whose recorded path is a placeholder.

    ``historical run importer`` writes a human-readable sentence into
    ``dataset_path`` rather than a real path, precisely because it is not
    resolvable. Fall back to the default dataset file in that case instead of
    trying to open the sentence.
    """
    recorded = run_metadata.get("dataset_path")
    path = Path(recorded) if recorded else Path(EVAL_DATASET_FILE)
    if not path.is_file():
        path = Path(EVAL_DATASET_FILE)
    return resolve_dataset(
        path,
        include_adversarial=bool(run_metadata.get("include_adversarial", False)),
        sample_size_per_source=run_metadata.get("sample_size_per_source"),
        sample_seed=run_metadata.get("sample_seed"),
    )


def dataset_for_generation(
    generation: Dict[str, Any],
    run_metadata: Dict[str, Any],
    *,
    rebind: bool = False,
) -> List[Dict[str, Any]]:
    """Resolve the dataset a generation was produced against.

    Embedded historical snapshots preserve their original rows. External
    datasets are reconstructed from the recorded path and sampling settings
    and checked against the recorded digest. An explicit rebind maps embedded
    questions to updated annotations in snapshot order.
    """
    completion = run_metadata.get("completion", {})
    if completion.get("status") in {"incomplete", "failed"}:
        raise DatasetMismatchError(
            f"Generation is {completion['status']}: {completion.get('error', '')}. "
            "Preserve this diagnostic artifact and rerun generation before scoring."
        )
    if rebind and "dataset" in generation:
        return rebind_embedded_dataset(
            generation["dataset"], current_dataset_for_rebind(run_metadata)
        )
    if "dataset" in generation:
        dataset = generation["dataset"]
    else:
        dataset = resolve_dataset(
            Path(run_metadata.get("dataset_path", EVAL_DATASET_FILE)),
            include_adversarial=bool(run_metadata.get("include_adversarial", False)),
            sample_size_per_source=run_metadata.get("sample_size_per_source"),
            sample_seed=run_metadata.get("sample_seed"),
        )
    recorded_sha = run_metadata.get("dataset_sha256")
    if recorded_sha and recorded_sha != sha256_json(dataset):
        raise DatasetMismatchError(
            "dataset_sha256 mismatch: restore the exact dataset used for generation. "
            "For an embedded historical snapshot, use --rebind-dataset explicitly "
            "to apply annotations to the same questions."
        )
    return dataset


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def skip_gt_validation_reason(chromadb_dir: Path, force_skip: bool) -> Optional[str]:
    """Return a reason string if ground-truth validation should be skipped, else ``None``.

    Pure path check -- never imports ``chromadb`` -- so it is safe to call
    (and unit test) with no corpus and no chromadb package installed.
    """
    if force_skip:
        return "--skip-gt-validation passed"
    if not Path(chromadb_dir).exists():
        return f"no corpus present at {chromadb_dir} (auto-skipped)"
    return None


def _collect_ablation_flags() -> Dict[str, bool]:
    """Snapshot every ablation flag currently defined in ``agentic_rag.config``.

    Reads by attribute name rather than a hardcoded import list, so new
    ablation flags (e.g. ``CRITIC_ENABLED`` / ``WORKER_QUERY_REFORMULATION_ENABLED``)
    appear here automatically as soon as they're added to
    ``agentic_rag.config`` -- no further change to this module needed.
    """
    from src.agentic_rag import config as agentic_config

    flag_names = (
        "SUPERVISOR_PREMISE_CHECK_ENABLED",
        "HYBRID_RERANK_ENABLED",
        "CRITIC_ENABLED",
        "WORKER_QUERY_REFORMULATION_ENABLED",
    )
    return {
        name: bool(getattr(agentic_config, name))
        for name in flag_names
        if hasattr(agentic_config, name)
    }


def cmd_generate(args: argparse.Namespace) -> int:
    run_id = args.run_id or generate_run_id()
    dataset_path = Path(args.dataset)
    dataset = resolve_dataset(
        dataset_path,
        include_adversarial=args.include_adversarial,
        sample_size_per_source=args.sample_size_per_source,
        sample_seed=args.sample_seed,
    )
    dataset_sha256 = sha256_json(dataset)
    gen_path = generation_path(run_id)

    if not dataset:
        print("error: dataset is empty; no generation was written", file=sys.stderr)
        return 1

    skip_reason = skip_gt_validation_reason(CHROMADB_DIR, args.skip_gt_validation)

    if args.dry_run:
        print(f"[dry-run] run_id={run_id}")
        print(f"[dry-run] generation path: {gen_path}")
        print(
            f"[dry-run] dataset: {dataset_path} (include_adversarial={args.include_adversarial}) "
            f"-> {len(dataset)} entries, dataset_sha256={dataset_sha256[:12]}..."
        )
        print(f"[dry-run] max_concurrent={args.max_concurrent}")
        print(f"[dry-run] system={args.system}")
        print(
            f"[dry-run] gt-validation: {'skipped -- ' + skip_reason if skip_reason else 'would run'}"
        )
        return 0

    if gen_path.exists():
        logger.warning("Overwriting existing generation at %s", gen_path)

    # Everything below needs the live pipeline (langchain / ChromaDB / a
    # provider API key) -- imported lazily so --dry-run and --help never
    # require it.
    from src.evaluation.evaluators.end_to_end_evaluator import EndToEndEvaluator
    from src.evaluation.evaluators.orchestration_evaluator import OrchestrationEvaluator
    from src.evaluation.provenance import build_run_metadata, collect_role_configs
    from src.evaluation.utils import (
        configure_evaluation_file_logging,
        default_eval_log_path,
    )

    if args.save_logs:
        log_path = (
            Path(args.log_file) if args.log_file else default_eval_log_path(run_id)
        )
        configure_evaluation_file_logging(log_path)
        logger.info("File logging enabled: %s", log_path)

    if skip_reason:
        logger.info("Skipping ground-truth validation: %s", skip_reason)
    else:
        try:
            from src.evaluation.utils import validate_ground_truth_chunks

            validate_ground_truth_chunks(dataset, CHROMADB_DIR)
        except ImportError as exc:
            logger.warning(
                "chromadb unavailable for ground-truth validation (%s). Skipping.", exc
            )

    evaluator = EndToEndEvaluator(CHROMADB_DIR, system=args.system)
    wall_start_utc = utc_now_iso()
    wall_start = time.monotonic()
    pipeline_results = evaluator.run_batch(
        dataset, args.max_concurrent, run_id=run_id, checkpoint_every=10
    )
    wall_seconds = time.monotonic() - wall_start
    wall_end_utc = utc_now_iso()

    pipeline_results, completion = finalize_pipeline_results(
        dataset, pipeline_results,
        stopped_early_error=getattr(evaluator, "_last_fail_fast_error", None),
    )
    pipeline_error_count = sum(1 for r in pipeline_results if "error" in r)
    cost = OrchestrationEvaluator().evaluate_cost(pipeline_results)
    pipeline_cost_usd = cost.get("total_usd", 0.0) if isinstance(cost, dict) else 0.0

    run_metadata = build_run_metadata(
        run_id=run_id,
        tier=args.tier or "unspecified",
        system=args.system,
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256,
        include_adversarial=args.include_adversarial,
        sample_size_per_source=args.sample_size_per_source,
        sample_seed=args.sample_seed,
        max_concurrent=args.max_concurrent,
        roles=collect_role_configs(),
        ablation_flags=_collect_ablation_flags(),
        wall_start_utc=wall_start_utc,
        wall_end_utc=wall_end_utc,
        wall_seconds=wall_seconds,
        pipeline_cost_usd=pipeline_cost_usd,
        n_entries=len(dataset),
        pipeline_error_count=pipeline_error_count,
    )
    run_metadata["completion"] = completion

    generation = {"run_metadata": run_metadata, "pipeline_results": pipeline_results}
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gen_path, "w", encoding="utf-8") as fh:
        json.dump(generation, fh, indent=2, default=str)

    print(f"generation written: {gen_path}")
    print(
        f"  n_entries={len(dataset)} pipeline_error_count={pipeline_error_count} "
        f"wall_seconds={wall_seconds:.1f} pipeline_cost_usd={pipeline_cost_usd:.4f}"
    )
    if completion.get("error"):
        print(f"error: {completion['error']}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _apply_judge_overrides(args: argparse.Namespace) -> None:
    """Apply CLI judge overrides as ``EVAL_LLM_*`` env vars.

    Mirrors ``end_to_end_evaluator.py``'s ``__main__`` convention: every
    ``judge_factory`` call reads these live from ``os.environ``, so setting
    them here (rather than plumbing new kwargs through the default
    factories) is sufficient and keeps ``deepeval_layer``/``ragas_layer``'s
    default factories usable unmodified.
    """
    import os

    for flag, env_name in (
        (args.judge, "EVAL_LLM_MODEL"),
        (getattr(args, "judge_provider", None), "EVAL_LLM_PROVIDER"),
        (getattr(args, "judge_base_url", None), "EVAL_LLM_BASE_URL"),
        (getattr(args, "judge_api_key_env", None), "EVAL_LLM_API_KEY_ENV"),
    ):
        if flag:
            os.environ[env_name] = flag

    # Judge temperature is not a factory kwarg -- judge_factory reads
    # EVAL_LLM_TEMPERATURE live from os.environ, and _resolve_judge_metadata
    # reads it back for the score-file judge block, so setting it here both
    # configures the judge and records the value. ``0.0`` is meaningful for
    # this project's judges (DeepSeek/Qwen accept it); only genuine OpenAI
    # reasoning models reject it, and judge_factory already forces those to 1.0.
    judge_temp = getattr(args, "judge_temperature", None)
    if judge_temp is not None:
        os.environ["EVAL_LLM_TEMPERATURE"] = str(judge_temp)


def _resolve_judge_metadata(judge_id: str) -> Dict[str, Any]:
    """Build the score-file ``judge`` block: provider/model/temperature + same-family flag."""
    from src.agentic_rag.llm_factory import resolve_role_config

    cfg = resolve_role_config("eval_judge")
    pipeline_model = resolve_role_config("worker").model
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "same_family_as_pipeline": cfg.model == pipeline_model,
        "requested_judge_id": judge_id,
    }


def _score_deterministic_payload(
    dataset: List[Dict[str, Any]], pipeline_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Wrap :func:`score_deterministic`'s output in the standard score-file payload shape.

    This layer has no per-entry judge scores -- every metric is already
    rolled up -- so the full metric tree becomes ``aggregates`` and
    ``per_entry`` is empty. ``n_scored``/``n_failed`` count entries that did
    / did not participate in quality metrics (i.e. pipeline-error stubs are
    ``n_failed``), matching :func:`~src.evaluation.scoring.deterministic._split_ok`.
    """
    from src.evaluation.scoring.deterministic import score_deterministic

    aggregates = score_deterministic(dataset, pipeline_results)
    n_failed = sum(1 for r in pipeline_results if "error" in r)
    n_scored = len(pipeline_results) - n_failed
    return {
        "per_entry": {},
        "aggregates": aggregates,
        "n_scored": n_scored,
        "n_failed": n_failed,
        "failures": [],
    }


def _score_deepeval_payload(
    dataset: List[Dict[str, Any]],
    pipeline_results: List[Dict[str, Any]],
    *,
    run_id: str,
    judge_id: str,
    generation_sha256: str,
    max_concurrency: int = 8,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    from src.evaluation.judge_factory import make_deepeval_judge_model
    from src.evaluation.scoring.cache import ScoringCache
    from src.evaluation.scoring.deepeval_layer import score_deepeval

    judge_model = make_deepeval_judge_model()
    cache = ScoringCache(cache_file_path(run_id, "deepeval", judge_id, tag))
    return score_deepeval(
        dataset,
        pipeline_results,
        judge_model=judge_model,
        judge_id=judge_id,
        generation_sha=sha256_json([generation_sha256, sha256_json(dataset)]),
        cache=cache,
        max_concurrency=max_concurrency,
    )


def _score_ragas_payload(
    dataset: List[Dict[str, Any]],
    pipeline_results: List[Dict[str, Any]],
    *,
    max_concurrency: int,
) -> Dict[str, Any]:
    from src.evaluation.scoring.ragas_layer import score_ragas

    return score_ragas(dataset, pipeline_results, max_concurrency=max_concurrency)


def cmd_score(args: argparse.Namespace) -> int:
    run_id = args.run_id

    if getattr(args, "save_logs", False):
        from src.evaluation.utils import (
            configure_evaluation_file_logging,
            default_eval_log_path,
        )

        layer_name = getattr(args, "layer", "score")
        log_path = (
            Path(args.log_file)
            if getattr(args, "log_file", None)
            else default_eval_log_path(f"{run_id}_{layer_name}")
        )
        configure_evaluation_file_logging(log_path)
        logger.info("File logging enabled: %s", log_path)

    try:
        generation, _gen_path = load_generation(run_id)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    generation_sha256 = sha256_json(generation)
    run_metadata = generation.get("run_metadata", {}) or {}
    pipeline_results = generation["pipeline_results"]

    rebind = bool(getattr(args, "rebind_dataset", False))
    try:
        dataset = dataset_for_generation(generation, run_metadata, rebind=rebind)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if rebind and "dataset" in generation:
        print(
            f"re-bound the embedded dataset snapshot to the current dataset "
            f"({len(dataset)} entries matched by query); dataset repairs now apply to this run"
        )
    if len(dataset) != len(pipeline_results):
        print(
            f"error: reconstructed dataset has {len(dataset)} entries but generation.json "
            f"has {len(pipeline_results)} pipeline_results -- the dataset file may have "
            "changed since this generation was produced.",
            file=sys.stderr,
        )
        return 1

    dataset_sha256 = sha256_json(dataset)
    if args.layer in ("deepeval", "ragas") and not args.judge:
        print(f"error: --judge is required for --layer {args.layer}", file=sys.stderr)
        return 1

    out_name = score_file_name(args.layer, args.judge, getattr(args, "tag", None))
    if args.dry_run:
        print(f"[dry-run] run_id={run_id} layer={args.layer} judge={args.judge}")
        print(f"[dry-run] generation_sha256={generation_sha256[:12]}...")
        print(f"[dry-run] score file -> {scores_dir(run_id) / out_name}")
        return 0

    started_utc = utc_now_iso()
    wall_start = time.monotonic()

    judge: Optional[Dict[str, Any]] = None
    # Real per-call counting/cost where trackable. deepeval_layer reports
    # real judge_calls/judge_cost_usd via judge_factory.judge_call_stats() for
    # every non-bare-OpenAI-string judge (DeepSeek/Qwen/Anthropic/etc, i.e.
    # this project's actual judges); a bare OpenAI-native judge string has no
    # hook DeepEval exposes, so that specific case reports None (untracked),
    # not a fake 0 -- no silent zeros extends to cost accounting too.
    # ragas_layer has no equivalent hook yet (RAGAS's llm_factory() returns a
    # native provider client this project doesn't wrap) -- left honestly at
    # None until a RAGAS-side interception point exists.
    judge_calls: Optional[int] = None
    judge_cost_usd: Optional[float] = None

    if args.layer == "deterministic":
        payload = _score_deterministic_payload(dataset, pipeline_results)
    elif args.layer == "deepeval":
        _apply_judge_overrides(args)
        payload = _score_deepeval_payload(
            dataset,
            pipeline_results,
            run_id=run_id,
            judge_id=args.judge,
            generation_sha256=generation_sha256,
            max_concurrency=args.max_concurrency,
            tag=getattr(args, "tag", None),
        )
        judge = _resolve_judge_metadata(args.judge)
        judge_calls = payload.get("judge_calls")
        judge_cost_usd = payload.get("judge_cost_usd")
    elif args.layer == "ragas":
        _apply_judge_overrides(args)
        payload = _score_ragas_payload(
            dataset, pipeline_results, max_concurrency=args.max_concurrency
        )
        judge = _resolve_judge_metadata(args.judge)
    else:
        print(f"error: unknown layer '{args.layer}'", file=sys.stderr)
        return 1

    wall_seconds = time.monotonic() - wall_start
    score_file = build_score_file(
        layer=args.layer,
        generation_sha256=generation_sha256,
        dataset_sha256=dataset_sha256,
        per_entry=payload["per_entry"],
        aggregates=payload["aggregates"],
        started_utc=started_utc,
        wall_seconds=wall_seconds,
        n_scored=payload["n_scored"],
        n_failed=payload["n_failed"],
        failures=payload.get("failures"),
        judge=judge,
        judge_calls=judge_calls,
        judge_cost_usd=judge_cost_usd,
        tag=getattr(args, "tag", None),
    )

    out_path = scores_dir(run_id) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(score_file, fh, indent=2, default=str)

    print(f"score file written: {out_path}")
    print(
        f"  n_scored={payload['n_scored']} n_failed={payload['n_failed']} wall_seconds={wall_seconds:.1f}"
    )
    return 0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


_BIAS_METRICS = {
    "deepeval": ["faithfulness", "hallucination", "correctness"],
    "ragas": ["faithfulness", "answer_correctness"],
}


def _inject_agreement_metrics(
    metrics: Dict[str, Any],
    faithfulness: Dict[str, Any],
    judge_layers: Dict[str, Dict[str, Any]],
    primary_judge: Optional[str] = None,
) -> None:
    """Compute the self-preference-bias and cross-framework-agreement
    statistics from the already-loaded score files and fold them into ``metrics``
    so ``report_generator`` surfaces them. Pure function of score files -- runs
    offline locally. Populates nothing (rather than a fake 0.0) when the required
    layers/judges are absent. ``primary_judge`` (the headline/DeepEval judge)
    orders each bias delta as ``primary - secondary`` so a self-family judge that
    inflates its own scores shows a negative delta."""
    from src.evaluation.scoring.agreement import (
        cross_framework_agreement,
        judge_bias,
    )

    def _model(sf: Dict[str, Any]) -> Optional[str]:
        return (sf.get("judge") or {}).get("model")

    deepeval_files = [
        sf for sf in judge_layers.values() if sf.get("layer") == "deepeval"
    ]
    ragas_files = [sf for sf in judge_layers.values() if sf.get("layer") == "ragas"]

    # Cross-framework agreement: prefer a DeepEval+RAGAS pair scored by the
    # *same* judge, so the correlation isolates the framework, not the judge.
    if deepeval_files and ragas_files:
        pair = next(
            (
                (d, r)
                for d in deepeval_files
                for r in ragas_files
                if _model(d) is not None and _model(d) == _model(r)
            ),
            (deepeval_files[0], ragas_files[0]),
        )
        cross = cross_framework_agreement(pair[0], pair[1], metric="faithfulness")
        if cross:
            faithfulness["cross_framework"] = cross

    # Self-preference bias: within one framework, delta between the primary
    # judge and a second judge over the same generation.
    bias_results: List[Dict[str, Any]] = []
    by_layer: Dict[str, List[Dict[str, Any]]] = {}
    for sf in judge_layers.values():
        by_layer.setdefault(sf.get("layer"), []).append(sf)
    for layer_name, files in by_layer.items():
        if len(files) < 2:
            continue
        primary = next((f for f in files if _model(f) == primary_judge), None)
        if primary is None:
            # Fall back to the cross-family judge, then to glob order.
            cross_fam = [
                f
                for f in files
                if not (f.get("judge") or {}).get("same_family_as_pipeline")
            ]
            primary = cross_fam[0] if cross_fam else files[0]
        secondary = next((f for f in files if f is not primary), None)
        if secondary is None:
            continue
        bias = judge_bias(
            primary, secondary, _BIAS_METRICS.get(layer_name, ["faithfulness"])
        )
        if bias:
            bias_results.append(bias)
    if bias_results:
        metrics["judge_comparison"] = bias_results


def _build_applicability(
    generation: Dict[str, Any],
    run_metadata: Dict[str, Any],
    layers: Dict[str, Dict[str, Any]],
    *,
    rebind: bool = False,
    dataset: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Dataset applicability plus quarantine-adjusted reference metrics.

    Best effort by design: an archived run whose dataset cannot be
    reconstructed still aggregates, it simply reports no applicability block.
    Failing the aggregate here would make dataset repairs retroactively break
    runs they were never meant to touch.
    """
    from src.evaluation.applicability import (
        REFERENCE_DEPENDENT_METRICS,
        dataset_applicability,
        domain_breakdown,
        recompute_reference_metrics,
    )

    if dataset is None:
        try:
            dataset = dataset_for_generation(generation, run_metadata, rebind=rebind)
        except Exception as exc:  # noqa: BLE001 - archived datasets can be unavailable
            logger.warning("applicability skipped: could not resolve dataset (%s)", exc)
            return None

    if len(dataset) != len(generation.get("pipeline_results", []) or []):
        logger.warning(
            "applicability skipped: resolved dataset has %d entries but the generation has %d",
            len(dataset),
            len(generation.get("pipeline_results", []) or []),
        )
        return None

    block: Dict[str, Any] = {"dataset": dataset_applicability(dataset)}

    for key, score_file in layers.items():
        per_entry = score_file.get("per_entry") or {}
        aggregates = score_file.get("aggregates") or {}
        if not isinstance(per_entry, dict) or not per_entry:
            continue
        adjusted = recompute_reference_metrics(per_entry, aggregates, dataset)
        breakdown = domain_breakdown(
            per_entry,
            dataset,
            sorted(
                set(aggregates)
                & (REFERENCE_DEPENDENT_METRICS | {"faithfulness", "hallucination"})
            ),
        )
        if adjusted or breakdown:
            entry: Dict[str, Any] = {}
            if adjusted:
                entry["reference_metrics_adjusted"] = adjusted
            if breakdown:
                entry["by_domain_relevance"] = breakdown
            block.setdefault("by_layer", {})[key] = entry

    return block


def cmd_aggregate(args: argparse.Namespace) -> int:
    run_id = args.run_id
    try:
        generation, _gen_path = load_generation(run_id)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    generation_sha256 = sha256_json(generation)
    run_metadata = generation.get("run_metadata", {}) or {}

    rebind = bool(getattr(args, "rebind_dataset", False))
    dataset = None
    try:
        dataset = dataset_for_generation(generation, run_metadata, rebind=rebind)
    except FileNotFoundError:
        if rebind or not run_metadata.get("dataset_sha256"):
            raise
        # Existing scores remain usable when the original external file is archived.
        dataset_sha256 = run_metadata["dataset_sha256"]
    else:
        if len(dataset) != len(generation["pipeline_results"]):
            raise DatasetMismatchError("dataset length differs from pipeline_results")
        dataset_sha256 = sha256_json(dataset)

    score_dir = scores_dir(run_id)
    score_paths = (
        sorted(p for p in score_dir.glob("*.json")) if score_dir.exists() else []
    )

    if args.dry_run:
        print(f"[dry-run] run_id={run_id}")
        print(f"[dry-run] generation_sha256={generation_sha256[:12]}...")
        print(f"[dry-run] score files found: {[p.name for p in score_paths]}")
        print(f"[dry-run] would write -> {results_path(run_id)}, {report_path(run_id)}")
        return 0

    # Verify every score file before merging any of them -- a single
    # mismatch must abort the whole aggregate with no results.json written.
    layers: Dict[str, Dict[str, Any]] = {}
    for p in score_paths:
        with open(p, encoding="utf-8") as fh:
            score_file = json.load(fh)
        verify_generation_sha(score_file, generation_sha256, source=str(p))
        if score_file.get("dataset_sha256") != dataset_sha256:
            raise DatasetMismatchError(
                f"dataset_sha256 mismatch in {p}: re-score every layer against "
                "the same dataset and use --rebind-dataset consistently."
            )
        layer_name = score_file.get("layer", p.stem)
        tag_part = sanitize_tag(score_file.get("tag"))
        if layer_name == "deterministic":
            key = "deterministic"
        else:
            judge_id = (score_file.get("judge") or {}).get("model") or "unknown"
            key = f"{layer_name}.{sanitize_judge_id(judge_id)}"
        if tag_part:
            key = f"{key}.{tag_part}"
        layers[key] = score_file

    det = layers.get("deterministic")
    metrics: Dict[str, Any] = (
        json.loads(json.dumps(det["aggregates"])) if det is not None else {}
    )

    judge_layers = {k: v for k, v in layers.items() if k != "deterministic"}
    faithfulness = metrics.setdefault("faithfulness", {})

    # The headline/primary judge is the one used for the DeepEval layer (
    # locks qwen3.7-flash). When a framework was scored by several judges (e.g. a
    # self-preference bias pass), surface the *primary* judge's numbers in the
    # headline tables, not whichever file sorts first on disk.
    deepeval_sfs = [sf for sf in judge_layers.values() if sf.get("layer") == "deepeval"]
    ragas_sfs = [sf for sf in judge_layers.values() if sf.get("layer") == "ragas"]
    primary_judge = (
        (deepeval_sfs[0].get("judge") or {}).get("model") if deepeval_sfs else None
    )

    def _pick_primary(sfs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not sfs:
            return None
        for sf in sfs:
            if (sf.get("judge") or {}).get("model") == primary_judge:
                return sf
        return sfs[0]

    prim_de = _pick_primary(deepeval_sfs)
    prim_ra = _pick_primary(ragas_sfs)
    if prim_de is not None:
        faithfulness["deepeval_metrics"] = prim_de["aggregates"]
    if prim_ra is not None:
        faithfulness["ragas_metrics"] = prim_ra["aggregates"]

    _inject_agreement_metrics(metrics, faithfulness, judge_layers, primary_judge)

    applicability = _build_applicability(
        generation,
        run_metadata,
        layers,
        rebind=bool(getattr(args, "rebind_dataset", False)),
        dataset=dataset,
    )

    layers_present = sorted(layers.keys())
    layers_missing = [] if det is not None else ["deterministic"]

    # Pipeline / judge / total cost reported separately, rather than
    # the old "$0.8186" that silently excluded thousands of unbilled judge
    # calls. judge_cost_usd is None (not summed as 0) if no judge score file
    # reports a real value (e.g. every judge layer used an untrackable
    # bare-OpenAI-native judge, or no judge layer has been scored yet).
    pipeline_cost_usd = (run_metadata or {}).get("pipeline_cost_usd")
    judge_cost_total = 0.0
    judge_calls_total = 0
    judge_cost_known = False
    for score_file in judge_layers.values():
        jc = score_file.get("judge_cost_usd")
        calls = score_file.get("judge_calls")
        if jc is not None:
            judge_cost_total += jc
            judge_cost_known = True
        if calls is not None:
            judge_calls_total += calls

    cost_summary = {
        "pipeline_cost_usd": pipeline_cost_usd,
        "judge_cost_usd": judge_cost_total if judge_cost_known else None,
        "judge_calls": judge_calls_total if judge_cost_known else None,
        "total_cost_usd": (
            round(pipeline_cost_usd + judge_cost_total, 8)
            if judge_cost_known and pipeline_cost_usd is not None
            else None
        ),
    }

    results = {
        "run_id": run_id,
        "generation_sha256": generation_sha256,
        "dataset_sha256": dataset_sha256,
        "run_metadata": run_metadata,
        "layers_present": layers_present,
        "layers_missing": layers_missing,
        "by_judge": judge_layers,
        "cost_summary": cost_summary,
        "applicability": applicability,
        "metrics": metrics,
    }

    out_results = results_path(run_id)
    out_results.parent.mkdir(parents=True, exist_ok=True)
    if out_results.exists() and not getattr(args, "no_archive", False):
        import shutil

        hist_dir = run_dir(run_id) / "results_history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_now_iso().replace(":", "-").replace("+00-00", "Z")
        archived = hist_dir / f"results.{stamp}.json"
        shutil.copy2(out_results, archived)
        print(f"previous results archived: {archived}")
    with open(out_results, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    from src.evaluation.reporting.report_generator import generate_report

    report_md = generate_report(metrics, run_id)
    out_report = report_path(run_id)
    with open(out_report, "w", encoding="utf-8") as fh:
        fh.write(report_md)

    print(f"results written: {out_results}")
    print(f"report written: {out_report}")
    print(f"layers present: {layers_present}")
    if layers_missing:
        print(f"layers missing: {layers_missing}")
    return 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.run",
        description=(
            "Staged eval CLI: generate -> score -> aggregate, with an on-disk "
            "layout and integrity contract (see this module's docstring)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser(
        "generate", help="Run the pipeline over a dataset and write generation.json"
    )
    p_gen.add_argument(
        "--run-id", default=None, help="Run identifier (default: auto-generated)"
    )
    p_gen.add_argument(
        "--tier",
        default=None,
        help="Free-form tier label recorded in run_metadata (e.g. T1, T2)",
    )
    p_gen.add_argument(
        "--dataset",
        default=str(EVAL_DATASET_FILE),
        help="Path to the main dataset JSON",
    )
    p_gen.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Append the adversarial split",
    )
    p_gen.add_argument(
        "--sample-size-per-source",
        type=int,
        default=None,
        help="Cap entries sampled per dataset source",
    )
    p_gen.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "RNG seed for --sample-size-per-source. This only seeds dataset sampling -- "
            "DeepSeek's native API exposes no inference seed, so pipeline generation itself "
            "is not reproducible via this flag."
        ),
    )
    p_gen.add_argument(
        "--system",
        default="full",
        choices=["full", "vanilla", "single_agent", "no_retrieval"],
        help=(
            "Which system to run: the full multi-agent graph (default) or a "
            "Stage-E comparator baseline (vanilla / single_agent / no_retrieval)."
        ),
    )
    p_gen.add_argument("--max-concurrent", type=int, default=EVAL_MAX_CONCURRENT)
    p_gen.add_argument(
        "--skip-gt-validation",
        action="store_true",
        help=(
            "Skip the ground-truth chunk-id audit against ChromaDB. Auto-skipped "
            "when no corpus directory is present even without this flag."
        ),
    )
    p_gen.add_argument(
        "--save-logs",
        action="store_true",
        default=True,
        help="Save python logs to file (default: True)",
    )
    p_gen.add_argument(
        "--no-save-logs",
        action="store_false",
        dest="save_logs",
        help="Disable saving python logs to file",
    )
    p_gen.add_argument("--log-file", default=None)
    p_gen.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve run_id/dataset/paths and print them without running the pipeline.",
    )
    p_gen.set_defaults(func=cmd_generate)

    p_score = sub.add_parser(
        "score",
        help="Score one generation with one layer (deterministic / deepeval / ragas)",
    )
    p_score.add_argument("--run-id", required=True)
    p_score.add_argument(
        "--layer", required=True, choices=["deterministic", "deepeval", "ragas"]
    )
    p_score.add_argument(
        "--judge", default=None, help="Judge model id (required for deepeval/ragas)"
    )
    p_score.add_argument("--judge-provider", default=None)
    p_score.add_argument("--judge-base-url", default=None)
    p_score.add_argument("--judge-api-key-env", default=None)
    p_score.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help=(
            "Judge sampling temperature (sets EVAL_LLM_TEMPERATURE; recorded in "
            "the score file). Use 0.0 for the most reproducible judging. Only "
            "affects the judge -- scoring runs against the frozen generation.json, "
            "never re-runs the pipeline."
        ),
    )
    p_score.add_argument(
        "--tag",
        default=None,
        help=(
            "Variant label (e.g. 'temp0') appended to the score filename, resume "
            "cache and aggregate key, so re-scoring the same (layer, judge) under "
            "different settings keeps every run instead of overwriting it. Change "
            "the tag whenever you change --judge-temperature."
        ),
    )
    p_score.add_argument(
        "--max-concurrency", type=int, default=4, help="RAGAS row concurrency"
    )
    p_score.add_argument(
        "--save-logs",
        action="store_true",
        default=True,
        help="Save python logs to file (default: True)",
    )
    p_score.add_argument(
        "--no-save-logs",
        action="store_false",
        dest="save_logs",
        help="Disable saving python logs to file",
    )
    p_score.add_argument("--log-file", default=None, help="Custom log file path")
    p_score.add_argument(
        "--rebind-dataset",
        action="store_true",
        help=(
            "For a migrated run that embeds its dataset snapshot: map the snapshot onto the "
            "current dataset by query, so later dataset repairs (l1_applicable, "
            "answer_gradable, defects) apply to this run too. Aborts unless the question set "
            "is unchanged. Pass the same flag to `aggregate`."
        ),
    )
    p_score.add_argument("--dry-run", action="store_true")
    p_score.set_defaults(func=cmd_score)

    p_agg = sub.add_parser(
        "aggregate",
        help="Merge every score file for a run into results.json + report.md",
    )
    p_agg.add_argument("--run-id", required=True)
    p_agg.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not snapshot the previous results.json to results_history/ before rewriting.",
    )
    p_agg.add_argument(
        "--rebind-dataset",
        action="store_true",
        help=(
            "Resolve applicability against the current dataset for a migrated run that "
            "embeds its dataset snapshot. Use whenever the run's score files were produced "
            "with `score --rebind-dataset`."
        ),
    )
    p_agg.add_argument("--dry-run", action="store_true")
    p_agg.set_defaults(func=cmd_aggregate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (GenerationMismatchError, DatasetMismatchError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
