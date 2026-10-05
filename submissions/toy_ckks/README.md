# Incomplete CKKS example for GCN

This submission demonstrates key setup, encryption, serialization and decryption
with TenSEAL. It includes a direct scalar implementation of GCN linear operations
and reconstruction scoring, but **does not implement encrypted ReLU**. It is not
a complete or verified encrypted GCN submission and does not support TAM.

## Copy and install

After installing the benchmark's common dependencies, copy this optional example:

```bash
cp -r submissions/toy_ckks submissions/my_method
python -m pip install -r submissions/my_method/requirements.txt
```

Run the small key/encryption/decryption demonstration:

```bash
python submissions/my_method/adapter.py --demo --threads 2
```

It encrypts and decrypts three illustrative values, prints the round-trip values,
then stops at `encrypted_relu()` with `Unsupported operation: encrypted ReLU`
and exit code 2. The values are not anomaly scores; this is not a smaller GCN
workload or an FHE benchmark result. Keys remain in memory and are not printed.
Without `--demo`, the command checks readiness without performing cryptography.

## Complete the activation placeholder

In your copied `adapter.py`, replace the body of **`encrypted_relu(ciphertext,
public_context)`** with your encrypted activation method. The reference function
is ReLU, `max(z, 0)`. The method accepts and returns a one-element CKKS vector.
The evaluator must not decrypt inputs or obtain the client secret key.

Document your method in the submission README and `describe()` metadata. Only
after implementing it, set `activation_ready = True`. Setting the flag alone
does not replace the placeholder: evaluation will still raise the unsupported
operation error at the first activation.

The sample context meets SEAL's default 128-bit parameter bound, but its modulus
chain is for the round-trip demonstration and initial linear operations, **not
a provisioned full GCN circuit**. Select and document parameters sufficient for
your complete method while retaining at least 128-bit security. Update security
evidence and thread reporting if your backend or parallelism changes.

No activation approximation, lookup table or performance optimization is supplied.
One ciphertext per scalar makes full-dataset execution potentially very expensive
in time, memory and storage. The example is intended to explain the interface,
not prescribe a practical full-workload implementation.

## Run the completed submission

```bash
python harness/run_submission.py --workload gcn --submission my_method --threads 2 --out measurements/my-gcn-run
```

The unchanged example is rejected during the description/readiness check, before
key generation or full-dataset encryption. This prevents expensive execution of
a known incomplete submission. Its report records an error and no completed
inference or FHE performance result. Once completed, the submission is checked
against the same frozen GCN dataset, weights, threshold and security requirements
as any other submission.

See the [interface contract](../../docs/submission-contract.md) and
[GCN model](../../docs/scam-list-gcn.md).
