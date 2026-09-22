# Submission: replace with your method name

## Technical description

Describe your scheme/algorithm and how it evaluates the four GCN layers and
mean squared reconstruction score. List numerical approximations and ranges.

## Security parameters

Provide at least 128-bit classical security. List every key type, ring dimensions,
moduli, secret/error distributions, scale or plaintext modulus, bootstrapping or
scheme-switching parameters and estimator/standard evidence. Include estimator
version, command, assumptions and output where relevant. Do not publish keys.

## Encoding and packing

Explain the encoding of the three normalized sensitive fields and the tensor
layout, slot utilization, rotations, padding and precision.

## Installation

List Python dependencies in `requirements.txt`; document native builds if used.

## Execution

```bash
python -m pip install -r requirements.txt
python -m pip install -r submissions/YOUR_NAME/requirements.txt
python harness/run_submission.py --submission YOUR_NAME --threads 2
```

## Limitations and results

List supported artifact IDs, input ranges, resource needs and approximation
failures. Attach the generated `report.json`; do not attach `io/`.
