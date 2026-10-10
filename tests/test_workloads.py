import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
from harness.model import predict
from harness.generate_input import load_bundle
from harness.params import ROOT, WORKLOADS
from harness.threading_report import validate_thread_report
from harness.security import validate_description
from harness.utils import write_json, sha256, write_blobs
from harness.verify_result import verify
from harness.reporting import read_server_timings


def test_runner_requires_explicit_workload():
    from harness.run_submission import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--submission', 'plaintext_debug'])


def test_public_context_rejects_extra_files_without_crypto():
    from harness.security import inspect_public_context
    with pytest.raises(ValueError, match='only the public'):
        inspect_public_context({'security': {'validator': 'seal_tc128'}},
                               {'context.bin': b'', 'secret.bin': b''}, 1)


def fixture(path, workload):
    rng = np.random.default_rng(42)
    f = WORKLOADS[workload]['features']
    x = rng.normal(size=(6, f))
    idx = [4, 6, 7] if workload == 'gcn' else [0, 2, 9]
    names = ['public_%d' % i for i in range(f)]
    for i, name in zip(idx, WORKLOADS[workload]['sensitive']):
        names[i] = name
    graph = dict(rows=list(range(6)), cols=list(range(6)), values=[1.] * 6)
    if workload == 'gcn':
        shapes = [('encoder_1', f, 3), ('encoder_2', 3, 2), ('decoder_1', 2, 3), ('decoder_2', 3, f)]
    else:
        shapes = [('gcn1', f, 3), ('gcn2', 3, 2)]
    weights = {}
    for name, a, b in shapes:
        weights[name + '.weight'] = rng.normal(size=(a, b)).tolist()
        weights[name + '.bias'] = rng.normal(size=b).tolist()
    if workload == 'tam':
        weights.update({'prelu1.weight': [.25], 'prelu2.weight': [.2]})
    data = dict(features=x.tolist(), labels=[0, 1] * 3, feature_names=names, sensitive_indices=idx,
                adjacency=graph, splits=dict(train=[0, 1], val=[2, 3], test=[4, 5]))
    public = dict(workload=workload, adjacency=graph, weights=weights, sensitive_indices=idx,
                  activation=WORKLOADS[workload]['activation'])
    if workload == 'tam':
        data['scoring_adjacency'] = graph
        public.update(scoring_adjacency=graph, norm_epsilon=1e-12)
    scores = predict(x, public)
    path.mkdir()
    for name, value in [('data.json', data), ('weights.json', weights),
                        ('reference.json', dict(scores=scores.tolist(), threshold=.5))]:
        write_json(path / name, value)
    manifest = dict(format_version=2, id='fixture-' + workload, workload=workload,
                    activation=public['activation'], purpose='unit_test', atol=.001, rtol=.001,
                    sha256={name: sha256(path / name) for name in ('data.json', 'weights.json', 'reference.json')})
    if workload == 'tam':
        manifest['norm_epsilon'] = 1e-12
    write_json(path / 'manifest.json', manifest)
    return path


@pytest.mark.parametrize('workload', ['gcn', 'tam'])
def test_schema_and_privacy(tmp_path, workload):
    bundle = load_bundle(fixture(tmp_path / workload, workload), workload)
    assert not bundle['registered']
    assert not {'labels', 'reference', 'features', 'country_code', 'account_id'} & set(bundle['public'])
    assert np.array(bundle['sensitive']).shape == (6, 3)
    assert len(bundle['public']['x_public'][0]) == WORKLOADS[workload]['features'] - 3


@pytest.mark.parametrize('workload', ['gcn', 'tam'])
def test_complete_pipeline(tmp_path, workload):
    source = fixture(tmp_path / workload, workload)
    out = tmp_path / 'run'
    result = subprocess.run([sys.executable, str(ROOT / 'harness/run_submission.py'),
                             '--workload', workload, '--submission', 'plaintext_debug',
                             '--artifacts', str(source), '--debug-plaintext', '--threads', '1',
                             '--out', str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'passed' and not report['eligible_for_comparison']
    assert report['primary_metrics'] == ['accuracy', 'roc_auc']
    stages = report['runs'][0]['stages']
    assert stages['evaluate']['submission_thread_report']['compute_threads'] == 1
    assert stages['evaluate']['sampled_process_tree_peak_os_threads'] >= 1
    for stage in stages.values():
        assert stage['memory_sampling_status'] in ('available', 'partial', 'unavailable')
        assert stage['memory_samples_observed'] <= stage['memory_sample_attempts']
        assert (stage['sampled_process_tree_peak_rss_bytes'] is None) == (stage['memory_samples_observed'] == 0)
    table = (out / 'comparison.md').read_text(encoding='utf-8')
    assert 'ROC-AUC (primary)' in table and 'Reported evaluator compute threads' in table
    assert 'Memory sampling by stage' in table


def test_mismatched_workload_rejected(tmp_path):
    with pytest.raises(ValueError, match='mismatch'):
        load_bundle(fixture(tmp_path / 'g', 'gcn'), 'tam')


def test_tam_dense_cosine_equivalence():
    from harness.workloads.tam import predict as tam_predict
    x = np.array([[1., -2.], [3., 4.], [-1., 2.]])
    a = sp.eye(3, format='csr')
    b = sp.csr_matrix([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
    weights = {'gcn1.weight': np.eye(2), 'gcn2.weight': np.eye(2),
               'gcn1.bias': np.zeros(2), 'gcn2.bias': np.zeros(2),
               'prelu1.weight': [.5], 'prelu2.weight': [.5]}
    u = np.where(x >= 0, x, .25 * x)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    expected = 1 - np.sum((u @ u.T) * b.toarray(), axis=1) / np.asarray(b.sum(axis=1)).ravel()
    np.testing.assert_allclose(tam_predict(x, a, b, dict(weights=weights, norm_epsilon=1e-12)), expected)


def test_relu_not_polynomial():
    from harness.workloads.gcn import predict as gcn_predict, LAYERS
    weights = {name + suffix: (np.eye(2) if suffix == '.weight' else np.zeros(2))
               for name in LAYERS for suffix in ('.weight', '.bias')}
    scores = gcn_predict(np.array([[-2., 3.]]), sp.eye(1), dict(weights=weights, sensitive_indices=[0]))
    np.testing.assert_allclose(scores, [4.])


def test_tam_zero_norm_guard():
    from harness.workloads.tam import predict as tam_predict
    weights = {'gcn1.weight': np.eye(2), 'gcn2.weight': np.eye(2),
               'gcn1.bias': np.zeros(2), 'gcn2.bias': np.zeros(2),
               'prelu1.weight': [.5], 'prelu2.weight': [.5]}
    scores = tam_predict(np.zeros((2, 2)), sp.eye(2), sp.eye(2), dict(weights=weights, norm_epsilon=1e-12))
    np.testing.assert_allclose(scores, 1.)


def test_tam_uses_distinct_scoring_graph(tmp_path):
    b = load_bundle(fixture(tmp_path / 'tam', 'tam'), 'tam')
    public = dict(b['public'], scoring_adjacency=dict(rows=[0,1,2,3,4,5], cols=[1,2,3,4,5,0], values=[1.]*6))
    assert not np.allclose(predict(np.asarray(b['data']['features']), public), b['reference']['scores'])


def test_checksum_tamper(tmp_path):
    path = fixture(tmp_path / 'gcn', 'gcn')
    (path / 'weights.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(path, 'gcn')


@pytest.mark.parametrize('value', [{}, {'compute_threads': True}, {'compute_threads': 0},
    {'compute_threads': 2, 'worker_processes': 1, 'threading_model': ''}])
def test_thread_reporting_required(value):
    with pytest.raises(ValueError):
        validate_thread_report(value)


def test_128_bit_policy():
    description = dict(is_fhe=True, security=dict(classical_bits=127, evidence='test'), parameters={'a':1})
    with pytest.raises(ValueError, match='128'):
        validate_description(description)
    description['security']['classical_bits'] = 128
    description.update(encoding='real', packing='none', activation='submission-defined')
    assert validate_description(description)['status'] == 'evidence_requires_review'
    description['security']['validator'] = 'seal_tc128'
    description['parameters'] = dict(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,40,60])
    with pytest.raises(ValueError, match='bound'):
        validate_description(description)


def test_plaintext_requires_flag():
    with pytest.raises(ValueError, match='debug-plaintext'):
        validate_description({'is_fhe': False})
    assert not validate_description({'is_fhe': False}, debug=True)['eligible']


def test_quality_report_on_numerical_failure():
    result = verify([.1,.4,.2,.3], [.1,.9,.2,.8], [0,1,0,1], [0,1,2,3], .5, .001, .001)
    assert not result['passed'] and result['quality']['accuracy'] == .5
    assert result['quality']['roc_auc'] == 1.


def test_nan_rejected():
    with pytest.raises(ValueError):
        verify([float('nan')], [1.], [1], [0], .5, .001, .001)


def test_optional_timing_validation(tmp_path):
    assert read_server_timings(tmp_path) == (None, [])
    write_json(tmp_path / 'server_reported_steps.json', {'Encrypted computation': 1., 'I/O': .1})
    assert read_server_timings(tmp_path)[0]['I/O'] == .1
    write_json(tmp_path / 'server_reported_steps.json', {'I/O': -1})
    assert read_server_timings(tmp_path)[1]


def test_payload_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        write_blobs(tmp_path / 'bad', {'../outside': b'no'})


@pytest.mark.parametrize('workload', ['gcn', 'tam'])
def test_registered_frozen_bundle(workload):
    bundle = load_bundle(ROOT / 'artifacts' / WORKLOADS[workload]['artifact_id'], workload)
    assert bundle['registered']
    assert bundle['public']['activation'] == WORKLOADS[workload]['activation']


def ckks_scaffold():
    from harness.utils import load_adapter
    return load_adapter(ROOT / 'submissions/toy_ckks/adapter.py')


def fake_ckks_backend():
    """Numeric test doubles only: never cryptography or benchmark evidence."""
    from types import SimpleNamespace
    calls = {'decrypt': 0, 'contexts': []}

    class Context:
        def __init__(self, secret=True, threads=1):
            self.secret, self.threads = secret, threads
            calls['contexts'].append(self)

        def generate_relin_keys(self):
            pass

        def serialize(self, **options):
            return json.dumps({'secret': options['save_secret_key']}).encode()

        def has_secret_key(self):
            return self.secret

    class Vector:
        def __init__(self, context, values):
            self.context, self.value = context, float(values[0])

        def __add__(self, other):
            return Vector(self.context, [self.value + (other.value if isinstance(other, Vector) else other)])

        def __sub__(self, other):
            return Vector(self.context, [self.value - other.value])

        def __mul__(self, other):
            return Vector(self.context, [self.value * (other.value if isinstance(other, Vector) else other)])

        def serialize(self):
            return json.dumps(self.value).encode()

        def decrypt(self):
            assert self.context.has_secret_key()
            calls['decrypt'] += 1
            return [self.value]

    fake = SimpleNamespace(SCHEME_TYPE=SimpleNamespace(CKKS='CKKS'), ckks_vector=Vector,
        context=lambda scheme, n_threads, **kw: Context(threads=n_threads),
        context_from=lambda blob, n_threads: Context(json.loads(blob)['secret'], n_threads),
        ckks_vector_from=lambda ctx, blob: Vector(ctx, [json.loads(blob)]))
    return fake, calls


def test_ckks_requires_activation_and_rejects_tam():
    adapter = ckks_scaffold()
    adapter.configure('gcn', 1)
    assert adapter.describe()['unsupported_operations']
    assert adapter.describe()['supported_workloads'] == ['gcn']
    validate_description(adapter.describe())  # Declared bounds only; no context created.
    with pytest.raises(NotImplementedError, match='Unsupported operation: encrypted ReLU'):
        adapter.encrypted_relu(None, None)
    with pytest.raises(ValueError, match='GCN only'):
        adapter.configure('tam', 1)


def test_ckks_readiness_rejected_before_keygen(tmp_path):
    source = fixture(tmp_path / 'gcn', 'gcn')
    out = tmp_path / 'run'
    result = subprocess.run([sys.executable, str(ROOT / 'harness/run_submission.py'),
                             '--workload', 'gcn', '--submission', 'toy_ckks',
                             '--artifacts', str(source), '--threads', '1', '--out', str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'error' and report['runs'] == []
    assert not report['eligible_for_comparison'] and 'keygen' not in report
    assert 'encrypted_relu()' in report['error']
    assert not (out / 'io/client/keys').exists()


def test_ckks_readiness_command_needs_no_fhe_dependency():
    result = subprocess.run([sys.executable, str(ROOT / 'submissions/toy_ckks/adapter.py')],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Unsupported operation: encrypted ReLU' in result.stderr


def test_ckks_interface_and_gcns_linear_wiring_with_test_doubles(tmp_path, monkeypatch):
    adapter = ckks_scaffold()
    fake, calls = fake_ckks_backend()
    monkeypatch.setitem(adapter.keygen.__globals__, 'backend', lambda: fake)
    adapter.configure('gcn', 2)
    private, public_keys = adapter.keygen(2)
    assert json.loads(private['context.bin'])['secret']
    assert not json.loads(public_keys['context.bin'])['secret']
    bundle = load_bundle(fixture(tmp_path / 'gcn', 'gcn'), 'gcn')
    encrypted = adapter.encrypt(bundle['sensitive'], private, 2)
    with pytest.raises(NotImplementedError, match='Unsupported operation'):
        adapter.evaluate(encrypted, bundle['public'], public_keys, 2, tmp_path)
    assert calls['decrypt'] == 0
    activations = []

    def mock_relu(value, context):
        assert not context.has_secret_key()
        activations.append(value.value)
        return fake.ckks_vector(context, [max(value.value, 0)])

    # Plain numerical test double, not an encrypted activation implementation.
    adapter.encrypted_relu = mock_relu
    result = adapter.evaluate(encrypted, bundle['public'], public_keys, 2, tmp_path)
    assert calls['decrypt'] == 0 and len(activations) == 6 * (3 + 2 + 3)
    decoded = adapter.decrypt(result, private, 2)
    np.testing.assert_allclose(decoded, bundle['reference']['scores'], atol=1e-12, rtol=1e-12)
    assert calls['decrypt'] == 6
    assert all(context.threads == 2 for context in calls['contexts'])


def test_ckks_demo_reaches_placeholder_with_test_doubles(monkeypatch, capsys):
    adapter = ckks_scaffold()
    fake, calls = fake_ckks_backend()
    namespace = adapter.keygen.__globals__
    monkeypatch.setitem(namespace, 'backend', lambda: fake)
    assert namespace['main'](['--demo', '--threads', '1']) == 2
    output = capsys.readouterr()
    assert json.loads(output.out)['decrypted'] == [.75, 1.2, .4]
    assert calls['decrypt'] == 3
    assert 'Unsupported operation: encrypted ReLU' in output.err
