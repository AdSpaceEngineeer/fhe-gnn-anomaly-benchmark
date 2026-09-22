# Internal test fixtures, not benchmark workloads

`arithmetic/` preserves the former three-node initialized-weight example for
software regression tests only. It is not registered and is not the workload
used by the copyable CKKS submission's quickstart. The single published workload
is `artifacts/scam-list-gcn-100k-v1`.

Tests may use deliberately small matrices or synthetic graphs to check individual
operations and block boundaries. Their results must not be presented as execution
of the full trained benchmark.
