"""Tests for the model-sweep loader / env builder."""

import textwrap
from pathlib import Path

import pytest

from src.evaluation.model_sweep import (
    build_run_env,
    build_run_kwargs,
    iter_run_envs,
    iter_run_kwargs,
    load_sweep,
)

_SWEEP = textwrap.dedent(
    """
    defaults:
      tier: T1
      eval_llm_model: gpt-5.4-mini
    providers:
      openai:
        api_key_env: OPENAI_API_KEY
      anthropic:
        api_key_env: ANTHROPIC_API_KEY
      together:
        provider: openai_compatible
        api_key_env: TOGETHER_API_KEY
        base_url: https://api.together.xyz/v1
    temperatures:
      supervisor: 1.0
      worker: 1.0
      reasoning: 1.0
      critic: 1.0
      data_analyst: 1.0
    runs:
      - name: baseline_gpt5mini
        roles:
          supervisor:   {provider: openai, model: gpt-5-mini}
          worker:       {provider: openai, model: gpt-5-mini}
          reasoning:    {provider: openai, model: gpt-5-mini}
          critic:       {provider: openai, model: gpt-5-mini}
          data_analyst: {provider: openai, model: gpt-5-mini}
      - name: strong_supervisor
        roles:
          supervisor:   {provider: openai, model: gpt-5}
      - name: anthropic_supervisor
        roles:
          supervisor:   {provider: anthropic, model: claude-sonnet-4-6}
      - name: oss_workers
        roles:
          worker:       {provider: together, model: meta-llama/Llama-3.3-70B-Instruct-Turbo}
      - name: together_selfjudge
        roles:
          worker:       {provider: together, model: meta-llama/Llama-3.3-70B-Instruct-Turbo}
        judge: {provider: together, model: meta-llama/Llama-3.3-70B-Instruct-Turbo}
      - name: judge_temperature_override
        roles:
          supervisor:   {provider: openai, model: gpt-5-mini}
        judge: {provider: openai, model: gpt-5-nano, temperature: 0.5}
    """
)

_SWEEP_NO_JUDGE_DEFAULTS = textwrap.dedent(
    """
    providers:
      openai:
        api_key_env: OPENAI_API_KEY
    runs:
      - name: only_run
        roles:
          supervisor:   {provider: openai, model: gpt-5-mini}
    """
)


@pytest.fixture
def sweep_path(tmp_path: Path) -> Path:
    p = tmp_path / "sweep.yaml"
    p.write_text(_SWEEP)
    return p


def test_load_and_validate(sweep_path):
    sweep = load_sweep(sweep_path)
    assert [r["name"] for r in sweep["runs"]][0] == "baseline_gpt5mini"


def test_baseline_run_full_env(sweep_path):
    env = build_run_env(load_sweep(sweep_path), "baseline_gpt5mini")
    assert env["AGENT_SUPERVISOR_MODEL"] == "gpt-5-mini"
    assert env["AGENT_SUPERVISOR_PROVIDER"] == "openai"
    assert env["EVAL_LLM_MODEL"] == "gpt-5.4-mini"  # judge fixed


def test_partial_run_inherits_baseline(sweep_path):
    env = build_run_env(load_sweep(sweep_path), "strong_supervisor")
    assert env["AGENT_SUPERVISOR_MODEL"] == "gpt-5"      # overridden
    assert env["AGENT_WORKER_MODEL"] == "gpt-5-mini"     # inherited from baseline


def test_anthropic_provider_resolution(sweep_path):
    env = build_run_env(load_sweep(sweep_path), "anthropic_supervisor")
    assert env["AGENT_SUPERVISOR_PROVIDER"] == "anthropic"
    assert env["AGENT_SUPERVISOR_API_KEY_ENV"] == "ANTHROPIC_API_KEY"


def test_named_provider_applies_base_url(sweep_path):
    env = build_run_env(load_sweep(sweep_path), "oss_workers")
    assert env["AGENT_WORKER_PROVIDER"] == "openai_compatible"
    assert env["AGENT_WORKER_BASE_URL"] == "https://api.together.xyz/v1"
    assert env["AGENT_WORKER_API_KEY_ENV"] == "TOGETHER_API_KEY"


def test_iter_run_envs_covers_all_runs(sweep_path):
    names = [name for name, _ in iter_run_envs(load_sweep(sweep_path))]
    assert names == ["baseline_gpt5mini", "strong_supervisor",
                     "anthropic_supervisor", "oss_workers",
                     "together_selfjudge", "judge_temperature_override"]


def test_duplicate_run_names_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("runs:\n  - name: a\n  - name: a\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_sweep(bad)


def test_unknown_run_raises(sweep_path):
    with pytest.raises(KeyError):
        build_run_env(load_sweep(sweep_path), "nope")


# ---------------------------------------------------------------------------
# Judge (EVAL_LLM_*) resolution: defaults.eval_llm_model (existing, model-only)
# vs a per-run `judge:` entry (new -- full provider/model/base_url/api_key_env/
# temperature resolution, same shape as a role entry).
# ---------------------------------------------------------------------------

def test_judge_defaults_to_eval_llm_model_when_no_per_run_override(sweep_path):
    """Runs without their own `judge:` entry get the sweep-wide default --
    unchanged behavior, but now also resolves provider/api_key_env, not just
    the bare model string."""
    env = build_run_env(load_sweep(sweep_path), "baseline_gpt5mini")
    assert env["EVAL_LLM_MODEL"] == "gpt-5.4-mini"
    assert env["EVAL_LLM_PROVIDER"] == "openai"
    assert env["EVAL_LLM_API_KEY_ENV"] == "OPENAI_API_KEY"


def test_per_run_judge_override_wins_over_defaults(sweep_path):
    env = build_run_env(load_sweep(sweep_path), "together_selfjudge")
    assert env["EVAL_LLM_MODEL"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert env["EVAL_LLM_PROVIDER"] == "openai_compatible"
    assert env["EVAL_LLM_BASE_URL"] == "https://api.together.xyz/v1"
    assert env["EVAL_LLM_API_KEY_ENV"] == "TOGETHER_API_KEY"


def test_per_run_judge_override_does_not_leak_to_other_runs(sweep_path):
    """The together_selfjudge run's override must not affect sibling runs --
    each run resolves its judge independently."""
    sweep = load_sweep(sweep_path)
    env = build_run_env(sweep, "strong_supervisor")
    assert env["EVAL_LLM_MODEL"] == "gpt-5.4-mini"
    assert env["EVAL_LLM_PROVIDER"] == "openai"


def test_judge_temperature_override(sweep_path):
    env = build_run_env(load_sweep(sweep_path), "judge_temperature_override")
    assert env["EVAL_LLM_MODEL"] == "gpt-5-nano"
    assert env["EVAL_LLM_TEMPERATURE"] == "0.5"


def test_judge_entry_without_model_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "providers:\n  openai:\n    api_key_env: OPENAI_API_KEY\n"
        "runs:\n  - name: r\n    roles:\n      supervisor: {provider: openai, model: gpt-5-mini}\n"
        "    judge: {provider: openai}\n"
    )
    with pytest.raises(ValueError, match="must include 'model'"):
        build_run_env(load_sweep(bad), "r")


def test_no_judge_configured_omits_eval_llm_vars(tmp_path):
    """When neither a per-run `judge:` nor any defaults.eval_llm_* is set,
    EVAL_LLM_* must be entirely absent -- this sweep file says nothing about
    the judge, so it must not inject anything the caller didn't ask for."""
    p = tmp_path / "sweep.yaml"
    p.write_text(_SWEEP_NO_JUDGE_DEFAULTS)
    env = build_run_env(load_sweep(p), "only_run")
    assert not any(k.startswith("EVAL_LLM_") for k in env)


# ---------------------------------------------------------------------------
# build_run_kwargs / iter_run_kwargs: the in-process counterpart to
# build_run_env / iter_run_envs, for callers (tests, a future non-subprocess
# runner) that want to pass provider/model directly to
# src.agentic_rag.llm_factory.make_chat_model(role, **kwargs) instead of
# mutating os.environ.
# ---------------------------------------------------------------------------

def test_build_run_kwargs_baseline_run(sweep_path):
    kwargs = build_run_kwargs(load_sweep(sweep_path), "baseline_gpt5mini")
    assert kwargs["supervisor"]["provider"] == "openai"
    assert kwargs["supervisor"]["model"] == "gpt-5-mini"
    assert kwargs["supervisor"]["temperature"] == 1.0


def test_build_run_kwargs_partial_run_inherits_baseline(sweep_path):
    kwargs = build_run_kwargs(load_sweep(sweep_path), "strong_supervisor")
    assert kwargs["supervisor"]["model"] == "gpt-5"       # overridden
    assert kwargs["worker"]["model"] == "gpt-5-mini"       # inherited from baseline


def test_build_run_kwargs_anthropic_provider(sweep_path):
    kwargs = build_run_kwargs(load_sweep(sweep_path), "anthropic_supervisor")
    assert kwargs["supervisor"]["provider"] == "anthropic"
    assert kwargs["supervisor"]["api_key_env"] == "ANTHROPIC_API_KEY"


def test_build_run_kwargs_named_provider_base_url(sweep_path):
    kwargs = build_run_kwargs(load_sweep(sweep_path), "oss_workers")
    assert kwargs["worker"]["provider"] == "openai_compatible"
    assert kwargs["worker"]["base_url"] == "https://api.together.xyz/v1"
    assert kwargs["worker"]["api_key_env"] == "TOGETHER_API_KEY"


def test_build_run_kwargs_omits_none_fields(sweep_path):
    """openai has no base_url in this sweep's providers: block, so it should be
    absent from the kwargs dict entirely (not present-as-None)."""
    kwargs = build_run_kwargs(load_sweep(sweep_path), "baseline_gpt5mini")
    assert "base_url" not in kwargs["supervisor"]


def test_build_run_kwargs_feeds_make_chat_model_directly(sweep_path, monkeypatch):
    """The whole point: this dict should be directly usable as
    make_chat_model(role, **kwargs) without any env-var mutation."""

    import sys
    import types

    fake_mod = types.ModuleType("langchain_openai")

    captured = {}

    class _FakeChatOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)

    from src.agentic_rag.llm_factory import make_chat_model

    kwargs = build_run_kwargs(load_sweep(sweep_path), "strong_supervisor")
    make_chat_model("supervisor", **kwargs["supervisor"])
    assert captured["model"] == "gpt-5"


def test_iter_run_kwargs_covers_all_runs(sweep_path):
    names = [name for name, _ in iter_run_kwargs(load_sweep(sweep_path))]
    assert names == ["baseline_gpt5mini", "strong_supervisor",
                     "anthropic_supervisor", "oss_workers",
                     "together_selfjudge", "judge_temperature_override"]


def test_build_run_kwargs_unknown_run_raises(sweep_path):
    with pytest.raises(KeyError):
        build_run_kwargs(load_sweep(sweep_path), "nope")


def test_build_run_kwargs_includes_eval_judge_when_configured(sweep_path):
    kwargs = build_run_kwargs(load_sweep(sweep_path), "together_selfjudge")
    assert kwargs["eval_judge"]["provider"] == "openai_compatible"
    assert kwargs["eval_judge"]["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert kwargs["eval_judge"]["base_url"] == "https://api.together.xyz/v1"
    assert kwargs["eval_judge"]["api_key_env"] == "TOGETHER_API_KEY"


def test_build_run_kwargs_includes_eval_judge_from_defaults(sweep_path):
    """Even without a per-run `judge:` entry, build_run_kwargs should surface
    the sweep-wide default judge -- previously build_run_kwargs said nothing
    about the judge at all."""
    kwargs = build_run_kwargs(load_sweep(sweep_path), "baseline_gpt5mini")
    assert kwargs["eval_judge"]["provider"] == "openai"
    assert kwargs["eval_judge"]["model"] == "gpt-5.4-mini"


def test_build_run_kwargs_omits_eval_judge_when_not_configured(tmp_path):
    p = tmp_path / "sweep.yaml"
    p.write_text(_SWEEP_NO_JUDGE_DEFAULTS)
    kwargs = build_run_kwargs(load_sweep(p), "only_run")
    assert "eval_judge" not in kwargs


def test_build_run_kwargs_eval_judge_feeds_judge_factory_directly(sweep_path, monkeypatch):
    """The whole point: this dict should be directly usable as
    make_ragas_judge_llm(**kwargs["eval_judge"]) / make_deepeval_judge_model(...)
    without any env-var mutation."""
    import sys
    import types

    fake_openai_mod = types.ModuleType("openai")
    fake_ragas_mod = types.ModuleType("ragas")
    fake_ragas_llms_mod = types.ModuleType("ragas.llms")
    captured = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw

    def _fake_llm_factory(model, **kw):
        captured["model"] = model
        captured["llm_kwargs"] = kw
        return object()

    fake_openai_mod.AsyncOpenAI = _FakeAsyncOpenAI
    fake_ragas_llms_mod.llm_factory = _fake_llm_factory
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas_mod)
    monkeypatch.setitem(sys.modules, "ragas.llms", fake_ragas_llms_mod)

    from src.evaluation.judge_factory import make_ragas_judge_llm

    kwargs = build_run_kwargs(load_sweep(sweep_path), "judge_temperature_override")
    make_ragas_judge_llm(**kwargs["eval_judge"])
    assert captured["model"] == "gpt-5-nano"
    assert captured["llm_kwargs"]["temperature"] == 0.5
    assert captured["llm_kwargs"]["provider"] == "openai"


# ---------------------------------------------------------------------------
# Real deliverable regression: configs/model_sweep.example.yaml itself must
# keep loading and resolving cleanly, including the newly added
# gemini_supervisor / qwen_workers / deepseek_supervisor / kimi_supervisor /
# minimax_workers / deepseek_pipeline_projudge / deepseek_pipeline_selfjudge /
# deepseek_pipeline_qwenjudge presets (the last two were renamed from their
# original -- inverted -- names; see the comment above them in the YAML).
# ---------------------------------------------------------------------------

_EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "model_sweep.example.yaml"


@pytest.mark.skipif(not _EXAMPLE_CONFIG.exists(), reason="configs/model_sweep.example.yaml not found")
def test_example_sweep_config_loads_and_resolves_all_runs():
    sweep = load_sweep(_EXAMPLE_CONFIG)
    for run_name, _ in iter_run_envs(sweep):
        env = build_run_env(sweep, run_name)
        assert env, f"run '{run_name}' resolved to an empty env dict"


@pytest.mark.skipif(not _EXAMPLE_CONFIG.exists(), reason="configs/model_sweep.example.yaml not found")
def test_example_sweep_config_projudge_run_overrides_judge():
    """deepseek_pipeline_projudge -- pipeline on flash, judge on the
    stronger pro tier (a cross-tier judge, not a self-judge -- see the
    inverted-names note above)."""
    sweep = load_sweep(_EXAMPLE_CONFIG)
    env = build_run_env(sweep, "deepseek_pipeline_projudge")
    # Pipeline roles stay on the cheaper flash tier...
    assert env["AGENT_SUPERVISOR_MODEL"] == "deepseek-v4-flash"
    # ...but the judge is explicitly overridden to the stronger/pricier tier.
    assert env["EVAL_LLM_PROVIDER"] == "openai_compatible"
    assert env["EVAL_LLM_MODEL"] == "deepseek-v4-pro"
    assert env["EVAL_LLM_BASE_URL"] == "https://api.deepseek.com/v1"
    assert env["EVAL_LLM_API_KEY_ENV"] == "DEEPSEEK_API_KEY"


@pytest.mark.skipif(not _EXAMPLE_CONFIG.exists(), reason="configs/model_sweep.example.yaml not found")
def test_example_sweep_config_selfjudge_run_uses_same_model_as_pipeline():
    """deepseek_pipeline_selfjudge -- pipeline AND judge both on
    deepseek-v4-flash, a genuine self-judge."""
    sweep = load_sweep(_EXAMPLE_CONFIG)
    env = build_run_env(sweep, "deepseek_pipeline_selfjudge")
    assert env["AGENT_SUPERVISOR_MODEL"] == "deepseek-v4-flash"
    assert env["EVAL_LLM_MODEL"] == "deepseek-v4-flash"
    assert env["EVAL_LLM_MODEL"] == env["AGENT_SUPERVISOR_MODEL"]


@pytest.mark.skipif(not _EXAMPLE_CONFIG.exists(), reason="configs/model_sweep.example.yaml not found")
def test_example_sweep_config_qwenjudge_run_is_cross_family():
    """deepseek_pipeline_qwenjudge -- pipeline on DeepSeek, judge on
    qwen3.7-flash (cross-family from the pipeline -- the project's chosen
    primary judge)."""
    sweep = load_sweep(_EXAMPLE_CONFIG)
    env = build_run_env(sweep, "deepseek_pipeline_qwenjudge")
    assert env["AGENT_SUPERVISOR_MODEL"] == "deepseek-v4-flash"
    assert env["EVAL_LLM_MODEL"] == "qwen3.7-flash"
    assert env["EVAL_LLM_PROVIDER"] == "openai_compatible"
    assert env["EVAL_LLM_BASE_URL"] == "https://ws-48q2hywrewc4xw5p.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    assert env["EVAL_LLM_API_KEY_ENV"] in ("ALICLOUD_API_KEY", "DASHSCOPE_API_KEY")
    assert env["EVAL_LLM_TEMPERATURE"] == "0.0"


@pytest.mark.skipif(not _EXAMPLE_CONFIG.exists(), reason="configs/model_sweep.example.yaml not found")
def test_example_sweep_config_other_runs_keep_default_judge():
    """The selfjudge run's override must not leak into sibling runs in the
    real deliverable config."""
    sweep = load_sweep(_EXAMPLE_CONFIG)
    env = build_run_env(sweep, "glm_supervisor")
    assert env["EVAL_LLM_MODEL"] == "gpt-5.4-mini"
    assert env["EVAL_LLM_PROVIDER"] == "openai"


@pytest.mark.skipif(not _EXAMPLE_CONFIG.exists(), reason="configs/model_sweep.example.yaml not found")
def test_example_sweep_config_gemini_and_qwen_use_openai_compatible():
    sweep = load_sweep(_EXAMPLE_CONFIG)
    gemini_env = build_run_env(sweep, "gemini_supervisor")
    assert gemini_env["AGENT_SUPERVISOR_PROVIDER"] == "openai_compatible"
    assert gemini_env["AGENT_SUPERVISOR_BASE_URL"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert gemini_env["AGENT_SUPERVISOR_API_KEY_ENV"] == "GEMINI_API_KEY"

    qwen_env = build_run_env(sweep, "qwen_workers")
    assert qwen_env["AGENT_WORKER_PROVIDER"] == "openai_compatible"
    assert qwen_env["AGENT_WORKER_API_KEY_ENV"] in ("ALICLOUD_API_KEY", "DASHSCOPE_API_KEY")

    glm_env = build_run_env(sweep, "glm_supervisor")
    assert glm_env["AGENT_SUPERVISOR_PROVIDER"] == "openai_compatible"
    assert glm_env["AGENT_SUPERVISOR_BASE_URL"] == "https://api.z.ai/api/paas/v4/"
    assert glm_env["AGENT_SUPERVISOR_API_KEY_ENV"] == "ZAI_API_KEY"
    assert glm_env["AGENT_SUPERVISOR_MODEL"] == "glm-5.2"


@pytest.mark.skipif(not _EXAMPLE_CONFIG.exists(), reason="configs/model_sweep.example.yaml not found")
def test_example_sweep_config_no_leftover_placeholders():
    """REPLACE_WITH_CURRENT_*_MODEL_ID placeholders should all be filled in."""
    sweep = load_sweep(_EXAMPLE_CONFIG)
    for run_name, _ in iter_run_envs(sweep):
        env = build_run_env(sweep, run_name)
        for key, value in env.items():
            assert "REPLACE_WITH_CURRENT" not in value, f"{run_name}: {key} still a placeholder"
