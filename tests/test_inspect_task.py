"""Offline tests for votingfacts_inspect.py (no API key: both models are Inspect's mockllm).

Run: uv run --with inspect-ai --with pytest pytest tests/
"""

import json
import sys
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

sys.path.insert(0, str(Path(__file__).parent.parent))
from votingfacts_inspect import parse_verdict, votingfacts  # noqa: E402

N_ITEMS = 15


@pytest.fixture(autouse=True)
def offline_token_estimates(monkeypatch):
    """Inspect's default token estimator downloads a tiktoken vocabulary on first use. These tests
    don't depend on token counts, so use a rough character heuristic and stay fully offline."""
    import inspect_ai.model._model as model_mod
    import inspect_ai.model._tokens as tokens_mod

    for mod in (tokens_mod, model_mod):
        monkeypatch.setattr(mod, "count_text_tokens", lambda text: max(1, len(text) // 4))


def mock(outputs: list[str]):
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs],
    )


def verdict(v: str, direction: str = "na", authority: str = "no_source") -> str:
    return json.dumps(
        {"verdict": v, "error_direction": direction, "source_authority": authority, "reasoning": "t"}
    )


def run(tmp_path, grader_outputs: list[str], epochs: int = 1, **task_args):
    n = len(list(votingfacts(**task_args).dataset)) * epochs
    log = inspect_eval(
        votingfacts(grader=mock(grader_outputs), **task_args),
        model=mock(["The election is on a date."] * n),
        log_dir=str(tmp_path),
        display="none",
        max_samples=1,  # serialise so mock outputs are consumed in dataset order
        epochs=epochs,
    )[0]
    assert log.status == "success", log.error
    by_key = {s.name: {k: e.value for k, e in s.metrics.items()} for s in log.results.scores}
    return log, {
        "accuracy": by_key["correct"]["mean"],
        "r1_accuracy": by_key["correct"]["r1_accuracy"],
        "suppressive_errors": by_key["suppressive_error"]["total"],
        "cited_official_rate": by_key["cited_official"]["mean"],
    }


def r1_flags():
    return [s.metadata["risk_tier"] == "R1" for s in votingfacts().dataset]


# ----------------------------------------------------------------------------- dataset


def test_dataset_shape():
    task = votingfacts()
    ids = [s.id for s in task.dataset]
    assert len(ids) == N_ITEMS == len(set(ids))
    assert {s.metadata["risk_tier"] for s in task.dataset} == {"R1", "R2", "R3"}
    assert all(s.target[0] for s in task.dataset)


def test_filters():
    assert all(s.metadata["risk_tier"] == "R1" for s in votingfacts(risk_tier="R1").dataset)
    assert {s.metadata["election_id"] for s in votingfacts(election="fi-2027").dataset} == {"fi-2027"}


# ----------------------------------------------------------------------------- scoring


def test_all_correct(tmp_path):
    _, m = run(tmp_path, [verdict("correct", authority="cited_official")] * N_ITEMS)
    assert m == {
        "accuracy": 1.0,
        "r1_accuracy": 1.0,
        "suppressive_errors": 0.0,
        "cited_official_rate": 1.0,
    }


def test_suppressive_errors_counted(tmp_path):
    _, m = run(tmp_path, [verdict("incorrect", "suppressive")] * N_ITEMS)
    assert m["accuracy"] == 0.0
    assert m["r1_accuracy"] == 0.0
    assert m["suppressive_errors"] == float(N_ITEMS)


def test_over_inclusive_error_is_not_suppressive(tmp_path):
    _, m = run(tmp_path, [verdict("incorrect", "over_inclusive")] * N_ITEMS)
    assert m["accuracy"] == 0.0
    assert m["suppressive_errors"] == 0.0


def test_safe_redirect_is_not_counted_as_correct(tmp_path):
    _, m = run(tmp_path, [verdict("safe_redirect")] * N_ITEMS)
    assert m["accuracy"] == 0.0


def test_r1_accuracy_isolates_the_high_stakes_items(tmp_path):
    flags = r1_flags()
    n_r1 = sum(flags)
    assert 0 < n_r1 < N_ITEMS

    # R1 items right, everything else wrong
    _, m = run(tmp_path / "a", [verdict("correct" if r1 else "incorrect") for r1 in flags])
    assert m["r1_accuracy"] == 1.0
    assert m["accuracy"] == pytest.approx(n_r1 / N_ITEMS)

    # R1 items wrong, everything else right
    _, m = run(tmp_path / "b", [verdict("incorrect" if r1 else "correct") for r1 in flags])
    assert m["r1_accuracy"] == 0.0
    assert m["accuracy"] == pytest.approx((N_ITEMS - n_r1) / N_ITEMS)


def test_r1_accuracy_is_nan_without_r1_items(tmp_path):
    n_r2 = len(list(votingfacts(risk_tier="R2").dataset))
    _, m = run(tmp_path, [verdict("correct")] * n_r2, risk_tier="R2")
    assert m["r1_accuracy"] != m["r1_accuracy"]  # NaN, not a misleading 0.0
    assert m["accuracy"] == 1.0


def test_metrics_are_correct_across_epochs(tmp_path):
    # Each item is judged twice: once a suppressive error, once correct, whatever order Inspect
    # runs them in. Over two epochs that is accuracy 0.5 and 0.5 suppressive errors per item.
    outs = [verdict("incorrect", "suppressive"), verdict("correct")] * N_ITEMS
    _, m = run(tmp_path, outs, epochs=2)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["r1_accuracy"] == pytest.approx(0.5)
    assert m["suppressive_errors"] == pytest.approx(N_ITEMS / 2)


def test_self_graded_flag(tmp_path):
    # no grader: the model under test grades itself, and the log says so
    log = inspect_eval(
        votingfacts(),
        model=mock(["answer", verdict("correct")] * N_ITEMS),
        log_dir=str(tmp_path),
        display="none",
        max_samples=1,
    )[0]
    assert log.status == "success", log.error
    assert all(s.scores["voter_info_judge"].metadata["self_graded"] for s in log.samples)


def test_graded_run_is_not_flagged_self_graded(tmp_path):
    log, _ = run(tmp_path, [verdict("correct")] * N_ITEMS)
    assert not any(s.scores["voter_info_judge"].metadata["self_graded"] for s in log.samples)


# ----------------------------------------------------------------------------- parsing


def test_parse_verdict_tolerates_fences_and_junk():
    text = 'Here you go:\n```json\n{"verdict": "Correct", "error_direction": "bogus"}\n```'
    v = parse_verdict(text)
    assert v["verdict"] == "correct" and v["error_direction"] == "na"


def test_error_direction_only_kept_for_incorrect_answers():
    v = parse_verdict('{"verdict":"correct","error_direction":"suppressive"}')
    assert v["error_direction"] == "na"
    v = parse_verdict('{"verdict":"incorrect","error_direction":"suppressive"}')
    assert v["error_direction"] == "suppressive"


def test_parse_verdict_rejects_bad_output():
    with pytest.raises(ValueError):
        parse_verdict("no json here")
    with pytest.raises(ValueError):
        parse_verdict('{"verdict": "maybe"}')
