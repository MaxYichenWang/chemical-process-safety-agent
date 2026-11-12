"""Loader + env builder for model-sweep configs (configs/model_sweep.example.yaml).

Turns a declarative sweep file into per-run environment-variable overrides that the
agentic RAG pipeline reads via ``src/agentic_rag/llm_factory.py``. Stdlib + PyYAML
only — no heavy pipeline imports — so it can be inspected/tested without ChromaDB.

CLI (dry-run inspection, no eval executed):

    python -m src.evaluation.model_sweep --config configs/model_sweep.yaml --print
    python -m src.evaluation.model_sweep --config configs/model_sweep.yaml --run strong_supervisor --print

To actually run a sweep, see scripts/reusable/run_model_sweep.py (needs ChromaDB + Parquet).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

_ROLE_PREFIX = {
    "supervisor": "AGENT_SUPERVISOR",
    "worker": "AGENT_WORKER",
    "reasoning": "AGENT_REASONING",
    "critic": "AGENT_CRITIC",
    "data_analyst": "AGENT_DATA_ANALYST",
}
_JUDGE_PREFIX = "EVAL_LLM"
_PROVIDER_TYPES = {"openai", "openai_compatible", "anthropic"}


def load_sweep(path: str | Path) -> dict:
    """Load and validate a sweep YAML file."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "runs" not in data:
        raise ValueError("sweep config must be a mapping with a top-level 'runs' list")
    names = [r.get("name") for r in data["runs"]]
    if any(not n for n in names):
        raise ValueError("every run must have a 'name'")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate run names: {names}")
    return data


def _resolve_provider(provider_key: str, providers: dict) -> Tuple[str, Dict[str, Any]]:
    """Resolve a role's provider field to (provider_type, settings).

    *provider_key* may be a provider TYPE (openai/anthropic/openai_compatible) or
    the NAME of an entry in the top-level ``providers`` map (e.g. together/groq).
    """
    if provider_key in providers:
        conf = providers[provider_key] or {}
        ptype = conf.get("provider", provider_key)
        return ptype, conf
    return provider_key, {}


def _merged_role_entries(sweep: dict, run_name: str) -> Dict[str, Dict[str, Any]]:
    """Resolve every role for *run_name* to a plain (non-env, non-kwargs) dict.

    Shared by ``build_run_env`` (formats as ``AGENT_<ROLE>_*`` env vars, for the
    subprocess-based sweep runner) and ``build_run_kwargs`` (formats as
    ``make_chat_model(role, **kwargs)``-ready dicts, for in-process callers such
    as tests or a future non-subprocess runner). Roles unspecified in the run
    inherit from the 'baseline' run if one exists.
    """
    runs = {r["name"]: r for r in sweep["runs"]}
    if run_name not in runs:
        raise KeyError(f"run '{run_name}' not found; have {sorted(runs)}")

    providers = sweep.get("providers", {}) or {}
    temps = sweep.get("temperatures", {}) or {}

    # Merge baseline roles (if present) under the selected run's roles.
    merged_roles: Dict[str, dict] = {}
    base = runs.get("baseline_gpt5mini") or runs.get("baseline")
    if base and base["name"] != run_name:
        merged_roles.update(base.get("roles", {}) or {})
    merged_roles.update(runs[run_name].get("roles", {}) or {})

    resolved: Dict[str, Dict[str, Any]] = {}
    for role, entry in merged_roles.items():
        if role not in _ROLE_PREFIX:
            raise ValueError(f"unknown role '{role}' in run '{run_name}'")
        ptype, pconf = _resolve_provider(entry.get("provider", "openai"), providers)
        if ptype not in _PROVIDER_TYPES:
            raise ValueError(
                f"role '{role}' resolves to unknown provider type '{ptype}' "
                f"(expected one of {sorted(_PROVIDER_TYPES)})"
            )
        temp = entry.get("temperature", temps.get(role))
        resolved[role] = {
            "provider": ptype,
            "model": str(entry["model"]),
            "base_url": pconf.get("base_url") or None,
            "api_key_env": pconf.get("api_key_env") or None,
            "temperature": float(temp) if temp is not None else None,
        }
    return resolved


def _resolve_judge_entry(sweep: dict, run_name: str) -> Optional[Dict[str, Any]]:
    """Resolve the ``eval_judge`` config for *run_name*, if this sweep configures one.

    Precedence: a per-run ``judge:`` entry (same shape as a role entry --
    ``{provider, model, [temperature]}``, with ``provider`` resolved through
    the same ``providers:`` block as roles) overrides ``defaults.eval_llm_*``
    for that run only. With neither a per-run ``judge:`` nor any
    ``defaults.eval_llm_*`` set, returns ``None`` -- meaning this sweep file
    says nothing about the judge, and ``EVAL_LLM_*`` is left entirely to
    whatever env vars / built-in default are already in place (byte-for-byte
    today's behavior for any sweep file written before this existed).

    Deliberately varying the judge per run (rather than the historical "keep
    it fixed for comparability" policy -- see ``EVALUATION.md``) is a
    documented, intentional escape hatch for same-provider-judge experiments
    (see ``EVALUATION.md``); it produces scores that
    are not directly comparable to a run using a different judge.
    """
    runs = {r["name"]: r for r in sweep["runs"]}
    if run_name not in runs:
        raise KeyError(f"run '{run_name}' not found; have {sorted(runs)}")

    providers = sweep.get("providers", {}) or {}
    defaults = sweep.get("defaults", {}) or {}

    entry = runs[run_name].get("judge")
    if entry is None:
        if not defaults.get("eval_llm_model") and not defaults.get("eval_llm_provider"):
            return None
        entry = {
            "provider": defaults.get("eval_llm_provider", "openai"),
            "model": defaults.get("eval_llm_model"),
            "max_tokens": defaults.get("eval_llm_max_tokens"),
        }

    if not entry.get("model"):
        raise ValueError(f"run '{run_name}': 'judge' entry must include 'model'")

    ptype, pconf = _resolve_provider(entry.get("provider", "openai"), providers)
    if ptype not in _PROVIDER_TYPES:
        raise ValueError(
            f"run '{run_name}': judge resolves to unknown provider type '{ptype}' "
            f"(expected one of {sorted(_PROVIDER_TYPES)})"
        )
    temp = entry.get("temperature")
    max_toks = entry.get("max_tokens", defaults.get("eval_llm_max_tokens"))
    return {
        "provider": ptype,
        "model": str(entry["model"]),
        "base_url": pconf.get("base_url") or None,
        "api_key_env": pconf.get("api_key_env") or None,
        "temperature": float(temp) if temp is not None else None,
        "max_tokens": int(max_toks) if max_toks is not None else None,
    }


def build_run_env(sweep: dict, run_name: str) -> Dict[str, str]:
    """Build the environment-variable override dict for a single run.

    For the subprocess-based sweep runner (``scripts/reusable/run_model_sweep.py``),
    which execs a separate eval process and therefore has to hand config across
    the process boundary via env vars. For in-process use (tests, notebooks, a
    direct pipeline call) prefer ``build_run_kwargs`` -- it resolves the same
    config without ever touching ``os.environ``.
    """
    resolved = _merged_role_entries(sweep, run_name)

    env: Dict[str, str] = {}
    for role, cfg in resolved.items():
        prefix = _ROLE_PREFIX[role]
        env[f"{prefix}_PROVIDER"] = cfg["provider"]
        env[f"{prefix}_MODEL"] = cfg["model"]
        if cfg["base_url"]:
            env[f"{prefix}_BASE_URL"] = cfg["base_url"]
        if cfg["api_key_env"]:
            env[f"{prefix}_API_KEY_ENV"] = cfg["api_key_env"]
        if cfg["temperature"] is not None:
            env[f"{prefix}_TEMPERATURE"] = str(cfg["temperature"])

    # Judge: fixed across runs by default for comparable scores (see
    # EVALUATION.md), but a run's own `judge:` entry can override
    # this deliberately -- see _resolve_judge_entry's docstring.
    judge = _resolve_judge_entry(sweep, run_name)
    if judge is not None:
        env["EVAL_LLM_PROVIDER"] = judge["provider"]
        env["EVAL_LLM_MODEL"] = judge["model"]
        if judge["base_url"]:
            env["EVAL_LLM_BASE_URL"] = judge["base_url"]
        if judge["api_key_env"]:
            env["EVAL_LLM_API_KEY_ENV"] = judge["api_key_env"]
        if judge["temperature"] is not None:
            env["EVAL_LLM_TEMPERATURE"] = str(judge["temperature"])
        if judge.get("max_tokens") is not None:
            env["EVAL_LLM_MAX_TOKENS"] = str(judge["max_tokens"])
    return env


def build_run_kwargs(sweep: dict, run_name: str) -> Dict[str, Dict[str, Any]]:
    """Build ``{role: kwargs}`` for a single run, ready to call
    ``src.agentic_rag.llm_factory.make_chat_model(role, **kwargs)`` directly.

    This is the in-process counterpart to ``build_run_env``: it resolves the same
    per-role provider/model/base_url/api_key_env/temperature config, but as
    explicit keyword arguments instead of environment variables -- so a single
    process can exercise multiple sweep runs (e.g. in a test, or a future
    non-subprocess runner) without mutating ``os.environ`` between them.

    Only non-``None`` fields are included per role, since ``make_chat_model``
    treats an omitted/``None`` kwarg as "fall back to env var / default" --
    passing every field through (including the ``None``s) would be equally
    correct, but omitting them keeps the returned dict a legible answer to
    "what does this run actually override".

    If this sweep configures a judge for *run_name* (via a per-run ``judge:``
    entry or ``defaults.eval_llm_*`` -- see ``_resolve_judge_entry``), the
    returned dict also has an ``"eval_judge"`` key, ready for
    ``src.evaluation.judge_factory.make_ragas_judge_llm(**kwargs["eval_judge"])``
    / ``make_deepeval_judge_model(**kwargs["eval_judge"])``. Absent entirely
    when the sweep doesn't configure a judge, matching the per-role omission
    behavior above.
    """
    resolved = _merged_role_entries(sweep, run_name)
    kwargs = {
        role: {k: v for k, v in cfg.items() if v is not None}
        for role, cfg in resolved.items()
    }
    judge = _resolve_judge_entry(sweep, run_name)
    if judge is not None:
        kwargs["eval_judge"] = {k: v for k, v in judge.items() if v is not None}
    return kwargs


def iter_run_envs(sweep: dict) -> Iterator[Tuple[str, Dict[str, str]]]:
    """Yield (run_name, env_overrides) for every run in the sweep."""
    for r in sweep["runs"]:
        yield r["name"], build_run_env(sweep, r["name"])


def iter_run_kwargs(sweep: dict) -> Iterator[Tuple[str, Dict[str, Dict[str, Any]]]]:
    """Yield (run_name, {role: kwargs}) for every run in the sweep."""
    for r in sweep["runs"]:
        yield r["name"], build_run_kwargs(sweep, r["name"])


def _main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a model-sweep config (dry run).")
    ap.add_argument("--config", required=True, help="Path to sweep YAML.")
    ap.add_argument("--run", help="Only show this run (default: all).")
    ap.add_argument(
        "--print", action="store_true", help="Print resolved env overrides."
    )
    args = ap.parse_args(argv)

    sweep = load_sweep(args.config)
    selected = [args.run] if args.run else [r["name"] for r in sweep["runs"]]
    for name in selected:
        env = build_run_env(sweep, name)
        print(f"\n# === run: {name} ===")
        if args.print:
            for k in sorted(env):
                print(f"export {k}={env[k]}")
        else:
            print(f"  {len(env)} env overrides ({', '.join(sorted(env))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
