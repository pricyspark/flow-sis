import pytest

from flowsis.cli.eval.profile_base import percentile, summarize_timings


def test_percentile_uses_nearest_observed_sample() -> None:
    samples = [5.0, 1.0, 4.0, 2.0, 3.0]

    assert percentile(samples, 0.0) == 1.0
    assert percentile(samples, 0.5) == 3.0
    assert percentile(samples, 0.9) == 5.0
    assert percentile(samples, 1.0) == 5.0


def test_summarize_timings_reports_latency_distribution_and_fps() -> None:
    summary = summarize_timings([50.0, 60.0, 70.0])

    assert summary == {
        "count": 3,
        "mean_ms": 60.0,
        "median_ms": 60.0,
        "p90_ms": 70.0,
        "p95_ms": 70.0,
        "min_ms": 50.0,
        "max_ms": 70.0,
        "fps_from_median": 16.667,
    }


@pytest.mark.parametrize("quantile", [-0.1, 1.1])
def test_percentile_rejects_invalid_quantile(quantile: float) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        percentile([1.0], quantile)


def test_timing_helpers_reject_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="empty"):
        summarize_timings([])
