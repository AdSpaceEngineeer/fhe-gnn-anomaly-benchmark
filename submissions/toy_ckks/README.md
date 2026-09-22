# Toy CKKS submission

Real TenSEAL 0.3.16 / Microsoft SEAL encrypted arithmetic for a tiny graph.
This revised example evaluates all four GCN layers, three `z + z²/8`
activations and the mean squared error over the three sensitive columns.
The earlier standalone toy had only one hidden activation and used a sum score;
its numerical outputs must not be compared with this workload.

## Parameters and security

- CKKS ring degree 32768, coefficient-modulus bits `[60] + [40]*12 + [60]`.
- Scale `2**40`, automatic rescaling, modulus switching and relinearization.
- No bootstrapping or Galois keys; one scalar per ciphertext.
- Total coefficient modulus is 600 bits, below SEAL's tc128 bound of 881
  for degree 32768. Standard SEAL ternary secrets/error distribution are used.
- The harness checks that the serialized public context has no secret key and
  matches the declared parameters using the SEAL parameter fingerprint.

Evidence: [SEAL security bounds](https://github.com/microsoft/SEAL/blob/v4.1.2/native/src/seal/util/hestdparms.h),
[default context security](https://github.com/microsoft/SEAL/blob/v4.1.2/native/src/seal/context.h),
[TenSEAL context construction](https://github.com/OpenMined/TenSEAL/blob/v0.3.16/tenseal/cpp/context/tensealcontext.cpp).
The 128-bit requirement is classical computational security, not a claim about
post-quantum strength, side channels or protection from malicious submissions.

## Execution

```bash
python -m pip install -r requirements.txt
python -m pip install -r submissions/toy_ckks/requirements.txt
python harness/run_submission.py --submission toy_ckks --threads 2 --num-runs 2
```

## Limits

Supports at most eight nodes, `poly2` activation and the published eight-feature
schema. The default `toy-v1` bundle contains three nodes and initialized,
untrained weights. Its labels/threshold are illustrative. It establishes
arithmetic agreement and adapter integration, not fraud-detection validity or
full-scale FHE performance. Parameters are sized for this circuit; a different
range/shape may need a different submission. No client-assisted intermediate
decryption is performed.

Graph weights and feature weights are multiplied in plaintext before encrypted
scalar multiplication. This is the same linear map as `A_norm @ X @ W`, with
fewer CKKS levels. It does not change the frozen graph, weights or operation order
relative to the activation. Dense matrix multiplication is decomposed into
scalar products and sums; this implementation is intentionally not optimized.
