import json
from pathlib import Path

from flowsis.cli.common import append_log_event


def test_append_log_event_writes_json_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "training_log.jsonl"

    append_log_event(log_path, "train_epoch", {"epoch": 0, "loss": 1.25})
    append_log_event(log_path, "validation_epoch", {"epoch": 0, "loss": 1.5})

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {"event": "train_epoch", "epoch": 0, "loss": 1.25},
        {"event": "validation_epoch", "epoch": 0, "loss": 1.5},
    ]
