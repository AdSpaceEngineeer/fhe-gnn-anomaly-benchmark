"""Missing-value presentation must not invent measurements or hardware."""
import copy
import json
import platform
import re
import sys
from urllib.parse import unquote, urlsplit

import pytest

from harness.params import ROOT, KEY_POLICY
from harness.reporting import _number, _text, comparison_markdown, hardware_table
from harness.run_submission import hardware_summary
from harness.utils import run_measured


@pytest.mark.parametrize('value,expected', [(None, 'Not reported'), (0, '0'), (1.5, '1.5')])
def test_numeric_labels(value, expected):
    assert _number(value) == expected


@pytest.mark.parametrize('value', [None, '', '  '])
def test_missing_text_labels(value):
    assert _text(value) == 'Not reported'


def test_hardware_is_not_collected_without_opt_in(monkeypatch):
    def forbidden():
        raise AssertionError('Hardware collection must be opt-in')
    for name in ('system', 'machine', 'processor'):
        monkeypatch.setattr(platform, name, forbidden)
    monkeypatch.setattr('os.cpu_count', forbidden)
    summary = hardware_summary()
    assert summary == dict.fromkeys(('os', 'machine', 'processor', 'logical_cpus'))
    assert json.loads(json.dumps(summary))['logical_cpus'] is None


def test_opt_in_blank_hardware_stays_null(monkeypatch):
    monkeypatch.setattr(platform, 'system', lambda: 'Example OS')
    monkeypatch.setattr(platform, 'machine', lambda: '')
    monkeypatch.setattr(platform, 'processor', lambda: '   ')
    monkeypatch.setattr('os.cpu_count', lambda: None)
    summary = hardware_summary(True)
    assert summary == dict(os='Example OS', machine=None, processor=None, logical_cpus=None)
    text = '\n'.join(hardware_table({'hardware': summary}))
    assert '| Operating system | Example OS |' in text
    assert '| CPU description | Not reported |' in text
    assert '| System logical CPUs | Not reported |' in text


def test_hardware_table_shows_provided_values_and_escapes_text():
    text = '\n'.join(hardware_table({'hardware': dict(os='OS', machine='arch',
        processor='Example | CPU\nmodel', logical_cpus=16)}))
    assert '| CPU description | Example \\| CPU model |' in text
    assert '| System logical CPUs | 16 |' in text


@pytest.mark.parametrize('hardware', [None, {}, {'processor': '', 'logical_cpus': None}])
def test_failed_or_legacy_report_without_hardware(hardware):
    text = comparison_markdown({'runs': [], 'hardware': hardware})
    assert '| CPU description | Not reported |' in text
    assert '| System logical CPUs | Not reported |' in text
    assert 'Eligible FHE comparison: **Not reported**' in text
    assert 'No completed inference runs' in text


def test_unavailable_threads_and_quality_are_not_zero():
    report = json.loads((ROOT / 'examples/gcn_plaintext_report.json').read_text())
    run = report['runs'][0]
    run['stages']['evaluate'].pop('sampled_process_tree_peak_os_threads', None)
    run['verification']['quality']['roc_auc'] = None
    before = copy.deepcopy(report)
    text = comparison_markdown(report)
    assert '| Sampled evaluator OS threads (peak) | Not reported | Not reported |' in text
    assert '| ROC-AUC (primary) | 0.998725 | Not reported |' in text
    assert '| Public/evaluation keys (MiB) | Not reported | 0 |' in text
    assert '| Maximum score error (worst run) | 0 | 0 |' in text
    assert report == before  # Rendering does not rewrite measurements or source hashes.


def test_thread_peak_uses_available_samples_without_fabricating_zero():
    report = json.loads((ROOT / 'examples/gcn_plaintext_report.json').read_text())
    report['runs'].append(copy.deepcopy(report['runs'][0]))
    report['runs'][0]['stages']['evaluate']['sampled_process_tree_peak_os_threads'] = None
    report['runs'][1]['stages']['evaluate']['sampled_process_tree_peak_os_threads'] = 7
    assert '| Sampled evaluator OS threads (peak) | Not reported | 7 |' in comparison_markdown(report)


@pytest.mark.parametrize('sampled,expected', [(None, None), (0, 0)])
def test_missing_thread_samples_stay_null_but_observed_zero_is_preserved(tmp_path, monkeypatch, sampled, expected):
    monkeypatch.setattr('harness.utils.sample_process_tree', lambda process: (None, sampled, False))
    result = run_measured([sys.executable, '-c', 'import time; time.sleep(.15)'], tmp_path / 'log', 10)
    assert result['sampled_process_tree_peak_os_threads'] == expected
    assert json.loads(json.dumps(result))['sampled_process_tree_peak_os_threads'] == expected


def test_network_and_key_policies_unchanged():
    assert KEY_POLICY == 'fresh_per_invocation_reused_across_repeats'
    report = json.loads((ROOT / 'examples/gcn_plaintext_report.json').read_text())
    assert report['network_transfer_seconds'] is None
    assert report['key_rotation_seconds'] is None
    text = comparison_markdown(report)
    assert 'Key/public-workload uploads are counted once and amortized; no network-transfer duration is measured.' in text
    assert 'Not measured—local execution' not in text
    assert 'Not applicable—fixed key set per invocation' not in text


def test_active_documentation_links_and_retired_instructions():
    docs = [ROOT / 'README.md']
    for directory in ('docs', 'artifacts', 'examples', 'submissions', 'scripts'):
        docs.extend((ROOT / directory).rglob('*.md'))
    problems = []
    for path in docs:
        text = path.read_text(encoding='utf-8')
        for stale in ('docs/task-status.md', 'To finish task 1', 'obtain the original output folder'):
            if stale.casefold() in text.casefold():
                problems.append(f'{path.relative_to(ROOT)}: stale instruction/link {stale}')
        for target in re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)', text):
            target = target.strip().strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            local = path.parent / unquote(parsed.path)
            if not local.exists():
                problems.append(f'{path.relative_to(ROOT)}: missing link target {target}')
    assert not problems, '\n'.join(problems)

