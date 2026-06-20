import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project" / "no_edit"))

from benchmark import BenchmarkResult, compute_score


def result(throughput_tok_per_sec: float, mean_e2el_ms: float) -> BenchmarkResult:
    return BenchmarkResult(
        throughput_tok_per_sec=throughput_tok_per_sec,
        mean_e2el_ms=mean_e2el_ms,
    )


def test_zero_throughput_scores_zero():
    assert compute_score(result(0.0, 100.0)) == 0.0


def test_negative_throughput_scores_zero():
    assert compute_score(result(-1.0, 100.0)) == 0.0


def test_latency_penalty_formula():
    assert compute_score(result(1000.0, 1000.0)) == 500.0


def test_zero_latency_scores_equal_throughput():
    assert compute_score(result(1234.5, 0.0)) == 1234.5


def test_score_decreases_as_latency_increases():
    low_latency = compute_score(result(1000.0, 100.0))
    high_latency = compute_score(result(1000.0, 1000.0))

    assert high_latency < low_latency


def test_score_increases_as_throughput_increases():
    low_throughput = compute_score(result(500.0, 100.0))
    high_throughput = compute_score(result(1000.0, 100.0))

    assert high_throughput > low_throughput
