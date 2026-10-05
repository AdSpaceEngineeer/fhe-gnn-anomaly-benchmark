# Schemes and execution

The harness is implementation-agnostic. A submission provides cryptographic
parameters, encoding, packing, encrypted evaluation and security evidence through
the [adapter interface](submission-contract.md). At least 128-bit classical
security is required. Backend dependencies remain submission-specific.

The same stage runner, security checks, process measurements, quality metrics
and report generation serve both workloads. Reference inference, graphs, weights,
sensitive feature indices and decision thresholds remain workload-specific.

The benchmark does not prescribe encrypted activation or normalization algorithms.
Submitters document their choices in their README and `describe()` output and
report their compute-thread configuration through `thread_report()`.
