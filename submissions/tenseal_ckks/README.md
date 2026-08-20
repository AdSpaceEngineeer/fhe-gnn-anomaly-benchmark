# TenSEAL CKKS submission

This is a real FHE test-drive backend using open-source TenSEAL 0.3.16 over
Microsoft SEAL. It encrypts one 72-feature prefix/suffix identifier histogram
per inference-subgraph node. Public graph features, topology, and model weights
remain plaintext, matching the benchmark encryption boundary.

Install with `python -m pip install -e ".[tenseal]"`. Start with batch size 1;
the node-packed reference strategy prioritizes clarity and correctness rather
than optimized packing. Because every retained length still occupies one
ciphertext per node, communication may remain flat across `k`. That is an
observed packing trade-off, not a benchmark error.

The declared CKKS parameters use polynomial modulus degree 16384, a 280-bit
coefficient-modulus chain, scale 2^40, and the library's standard 128-bit
security validation. Results remain experimental and must report their exact
hardware, library version, quality, and measured costs.

Run the canonical sweep with:

```bash
python -m pip install -e ".[tenseal]"
fhe-gnn-benchmark sweep submissions/tenseal_ckks/submission.json \
  prepared/yelpchi baselines/yelpchi/model.json tenseal-results \
  --batch-size 1 --num-runs 1
```

The checked-in smoke run evaluated all five canonical retention points with
actual key generation, encryption, homomorphic GNN arithmetic, and decryption.
On the recorded Windows host, online latency was 10.54–11.07 seconds per target,
peak RAM was about 1.27 GiB, storage about 444 MiB, and communication about
272 MiB. Maximum score error against plaintext-truncated inference was below
`8e-9`. These batch-1 quality values are diagnostic; full-test Recall/F1 remains
a separate required experiment.
