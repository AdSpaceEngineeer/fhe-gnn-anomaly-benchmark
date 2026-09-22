import json
import math
import pytest
from harness.params import ROOT
from harness.reporting import read_server_timings, comparison_markdown, SERVER_TIMINGS
from harness.utils import read_json


def test_optional_server_timings_missing(tmp_path):
    assert read_server_timings(tmp_path) == (None, [])


def test_bert_style_server_timings(tmp_path):
    values = {"Encrypted computation": 2.0, "I/O": .25, "Total": 2.25}
    (tmp_path / SERVER_TIMINGS).write_text(json.dumps(values))
    assert read_server_timings(tmp_path) == (values, [])


@pytest.mark.parametrize("text", ['[]', '{}', '{"I/O": -1}', '{"I/O": NaN}', '{"I/O": Infinity}',
                                 '{"I/O": "2s"}', '{"I/O": true}', '{"I/O": 1, "I/O": 2}', 'broken'])
def test_bad_server_timings_do_not_change_main_result(tmp_path, text):
    (tmp_path / SERVER_TIMINGS).write_text(text)
    result, warnings = read_server_timings(tmp_path)
    assert result is None and len(warnings) == 1


def test_report_summary_retains_main_metrics():
    report = read_json(ROOT / "examples/frozen_plaintext_report.json")
    original = json.dumps(report, sort_keys=True)
    text = comparison_markdown(report)
    assert "0.915254" in text and "Plaintext debug" in text
    assert "not measured, not zero" in text
    assert json.dumps(report, sort_keys=True) == original
    report["runs"][0]["server_reported_steps"] = {"Encrypted computation": 1234., "I/O": 0.25}
    text = comparison_markdown(report)
    assert "1234" in text and "not independently verified" in text
    assert str(report["runs"][0]["stages"]["evaluate"]["wall_seconds"]) != "1234"


def test_failed_run_is_not_fabricated():
    text = comparison_markdown({"status": "error", "runs": []})
    assert "No completed inference" in text and "Recall" not in text
