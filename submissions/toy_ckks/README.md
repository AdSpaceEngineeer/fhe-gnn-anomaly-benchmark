# Copyable toy CKKS submission

“Toy” means a readable, unoptimized FHE implementation. It evaluates the same
frozen four-layer Scam_List_GCN and 100,000-event dataset as other submissions.
It is not a smaller model, a retrained model or a different benchmark workload.

## Copy, install, run

From the repository root, inside your Python environment:

```bash
python -c "import shutil; shutil.copytree('submissions/toy_ckks', 'submissions/my_method')"
python -m pip install -r requirements.txt -r submissions/my_method/requirements.txt
python harness/run_submission.py --submission my_method --threads 2
```

Copy the whole directory: `adapter.py`, `packed_ckks.py`, `requirements.txt` and
this README. The submission name follows its directory. No weight selection,
training or model-preparation command is required. Frozen weights, graph and
public features arrive through the adapter interface.

## Implementation

- The client packs only the three sensitive features into 64-slot event rows.
  One degree-32768 CKKS ciphertext holds 256 such rows.
- The evaluator adds the five public features, then evaluates all four frozen
  feature projections and sparse normalized graph aggregations.
- Rotations and plaintext diagonal masks move complete event rows, including
  edges crossing ciphertext blocks. No edge pruning or graph renormalization occurs.
- Hidden activation is `z + z*z/8`. The output is the mean squared error over
  sensitive columns 4, 6 and 7. Unused output slots are masked.
- The client decrypts one score per event in the original row order.
  There is no intermediate decryption or access to the reference scores/labels.

Any internal weight encoding/packing happens inside evaluation and is timed.
The helper does not train, load a different model or use plaintext sensitive values.

## Parameters and security

- TenSEAL 0.3.16 native Microsoft SEAL bindings.
- CKKS polynomial degree 32768, scale `2**40`.
- Modulus chain: `[60] + [40]*16 + [60]`, total 760 bits, below SEAL's
  degree-32768 TC128 bound of 881 bits.
- Explicit `SEC_LEVEL_TYPE.TC128`; standard SEAL secret/error distributions.
- Relinearization keys and positive power-of-two rotation keys.
- Sixteen rescaling levels cover four projections, four graph aggregations,
  three two-level polynomial activations, and two scoring levels.
- Fresh keys per invocation, reused for repeats. Server files contain only
  serialized parameters, public key, relinearization keys and rotation keys.
- The harness validates the actual serialized parameters/public key against
  the declaration. Source/threat-model review remains necessary.

Sources: [SEAL security bounds](https://github.com/microsoft/SEAL/blob/v4.1.2/native/src/seal/util/hestdparms.h),
[CKKS arithmetic/rescaling](https://github.com/microsoft/SEAL/blob/v4.1.2/native/examples/5_ckks_basics.cpp).

## Optional server timings

The adapter writes `server_reported_steps.json` inside the supplied
`intermediate_dir`, using BERT's flat name-to-seconds format:

- `Encrypted computation`: arithmetic, plaintext encoding and public packing work.
- `I/O`: native key/ciphertext serialization, deserialization and temporary file access.
- `Context and other setup`: remaining adapter setup/bookkeeping.
- `Total`: the measured adapter evaluation interval.

These timers are additional self-reported detail. The worker separately records
its own input/output file access. Neither replaces total harness stage wall time,
which includes process/import overhead. The timing JSON itself is metadata,
not an encrypted intermediate; it is excluded from intermediate-value storage.

## Resources and validation status

This is not a fast demonstration: the fixed graph has 100,000 events. Key
generation alone can take minutes and use several GiB. Packed graph arithmetic
can take much longer. The implementation has no artificial eight-node limit,
but that is not a performance guarantee. Use an appropriately allocated machine;
the native arithmetic is largely single-threaded even when a larger thread
budget is supplied. Do not launch many copies to bypass shared-resource rules.

The revised implementation is checked through code/interface and non-cryptographic
packing/algebra tests. A live cryptographic check was stopped during key generation;
**a completed encrypted run of this revision, including the full workload, is
not claimed**. Historical scalar-toy metrics do not validate this implementation.

Ordinary tests and pushes do not launch live CKKS. A maintainer may explicitly
opt into an internal encrypted integration test later:

```bash
RUN_FHE_TESTS=1 python -m pytest -q tests/test_ckks.py
```

PowerShell: set `$env:RUN_FHE_TESTS="1"` first. Internal fixture tests are not
full-workload benchmark measurements. Successful benchmark runs produce
`report.json` and `comparison.md`; never share their `io/` key directories.
