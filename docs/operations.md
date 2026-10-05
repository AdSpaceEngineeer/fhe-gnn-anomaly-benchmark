# Ciphertext operation comparison

This table maps the reference mathematics to encrypted computation when the
three selected features are ciphertext and graphs/weights remain plaintext.
It does not prescribe an implementation or a universal operation-cost ordering.

| Operation | Scam_List_GCN | Scam_List_TAM |
|---|---|---|
| Ciphertext-plaintext addition | Merge public contribution and biases | Same |
| Ciphertext-ciphertext addition | Graph aggregation and reconstruction-error reduction | Graph aggregation, norms and affinity reduction |
| Ciphertext-plaintext multiplication | Graph coefficients and score averaging | Graph coefficients, PReLU slopes and degree averaging |
| Ciphertext-plaintext matrix multiplication | Four projections: 8->64->32->64->8 | Two projections: 10->64->32 |
| Ciphertext-ciphertext subtraction | Reconstructed minus original sensitive features | No reconstruction subtraction |
| Ciphertext-ciphertext multiplication | Squared reconstruction errors | Embedding squares, normalization and cosine products |
| Piecewise activation | Three ReLUs | Two learned-slope PReLUs |
| Norm-dependent normalization | None | Divide by max(L2 norm, 1e-12): squares, sum, square root/reciprocal and guard |
| Plaintext-ciphertext subtraction | Not required for score | One minus mean affinity |

ReLU and PReLU require an encrypted treatment of the sign-dependent branch;
ordinary approximate arithmetic alone is not exact comparison. TAM normalization
adds a nonlinear function of encrypted embeddings. The workload retains these
standard functions. Submitters supply their own implementations and report
accuracy, numerical error and FHE measurements against the frozen reference.

Packing may introduce rotations; a scheme may require relinearization, rescaling,
modulus switching or bootstrapping. These are implementation costs, not additional
reference GNN layers. Training, synthetic-data generation, graph truncation and
client-side thresholding are outside encrypted evaluation.
