from types import SimpleNamespace
import json
import sys
import psutil
import pytest
from harness.utils import sample_process_tree, memory_sampling_summary, run_measured
from harness.reporting import comparison_markdown, memory_table, stage_measurements
from harness.params import ROOT


def process(rss=100, children=(), failure=None):
    def memory():
        if failure:
            raise failure
        return SimpleNamespace(rss=rss)
    return SimpleNamespace(children=lambda recursive: list(children),
                           memory_info=memory, num_threads=lambda: 1)


def test_snapshot_complete():
    assert sample_process_tree(process(children=[process(50)])) == (150, 2, True)


@pytest.mark.parametrize('failure', [psutil.NoSuchProcess(123), psutil.AccessDenied(123)])
def test_snapshot_partial(failure):
    assert sample_process_tree(process(children=[process(failure=failure)])) == (100, 2, False)


def test_snapshot_zero_is_unavailable():
    assert sample_process_tree(process(0)) == (None, 1, False)


@pytest.mark.parametrize('args,status', [((None, 0, 0, 0), 'unavailable'),
    ((None, 3, 0, 3), 'unavailable'), ((100, 3, 3, 0), 'available'), ((100, 3, 2, 1), 'partial')])
def test_sampling_status(args, status):
    result = memory_sampling_summary(*args)
    assert result['memory_sampling_status'] == status
    if status == 'unavailable':
        assert result['sampled_process_tree_peak_rss_bytes'] is None


def test_unavailable_sampling_keeps_stage_success(tmp_path, monkeypatch):
    monkeypatch.setattr('harness.utils.sample_process_tree', lambda process: (None, 1, False))
    result = run_measured([sys.executable, '-c', 'import time; time.sleep(.15)'], tmp_path / 'log', 10)
    assert result['memory_sampling_status'] == 'unavailable'
    assert result['sampled_process_tree_peak_rss_bytes'] is None
    assert result['memory_samples_observed'] == 0


def test_failed_run_table_shows_memory():
    m = dict(memory_sampling_summary(None, 0, 0, 0), process_lifetime_peak_rss_bytes=2**20)
    table = comparison_markdown(dict(runs=[], description_stage=m))
    assert '| description_stage | unavailable | 0 | — | 1 |' in table


def test_legacy_zero_not_displayed_as_measured_zero():
    table = '\n'.join(memory_table({'keygen': {'sampled_process_tree_peak_rss_bytes': 0}}))
    assert 'unavailable (legacy; coverage unknown)' in table
    assert '| — | — | — |' in table


@pytest.mark.parametrize('workload', ['gcn', 'tam'])
def test_published_examples(workload):
    path = ROOT / 'examples' / (workload + '_plaintext_report.json')
    report = json.loads(path.read_text())
    for _, stage in stage_measurements(report):
        assert stage['memory_sampling_provenance'] == 'legacy_peak_only'
        assert stage['memory_samples_observed'] is None
    table = (path.with_name(workload + '_plaintext_comparison.md')).read_text(encoding='utf-8')
    assert table == comparison_markdown(report)


def test_readme_guidance_is_under_100_words():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    guidance = text.split('### Guidance for first-time FHE engineers\n')[1].split('\n## ')[0]
    assert len(guidance.split()) < 100


def test_model_internals_identifies_frozen_weights():
    from harness.params import WORKLOADS
    from harness.utils import sha256
    text = (ROOT / 'docs/model-internals.md').read_text(encoding='utf-8')
    for policy in WORKLOADS.values():
        path = ROOT / 'artifacts' / policy['artifact_id']
        assert sha256(path / 'manifest.json') in text
        assert sha256(path / 'weights.json') in text


def test_comparison_handles_all_memory_unavailable():
    report = json.loads((ROOT / 'examples/gcn_plaintext_report.json').read_text())
    for _, stage in stage_measurements(report):
        stage.update(memory_sampling_summary(None, 0, 0, 0))
        stage['process_lifetime_peak_rss_bytes'] = None
    assert '| Peak stage RAM (MiB, maximum) | — | — |' in comparison_markdown(report)
