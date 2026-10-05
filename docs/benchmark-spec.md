# Benchmark specification

This benchmark evaluates FHE inference for graph-based scam, crime and fraud
prevention workloads. Three numeric features are encrypted in each workload;
graph structure, remaining model features and frozen weights are public.
The workloads use synthetic data and do not establish deployment performance
on real financial crime.

| Workload | Input | Network | Score |
|---|---|---|---|
| Scam_List_GCN | 100,000 transfer events, 8 features | 8 -> 64 -> 32 -> 64 -> 8; three ReLUs | Mean squared reconstruction error on three sensitive features |
| Scam_List_TAM | 39,357 accounts, 10 features | 10 -> 64 -> 32; two learned PReLUs | One minus mean neighbor cosine similarity |

TAM is the higher computational challenge workload because it adds norm-dependent
normalization and learned piecewise activation to encrypted graph inference.
This is not a claim of better detection or guaranteed higher wall time. Models
use different datasets and cannot isolate architecture effects in a direct
cross-workload accuracy or runtime comparison.

Training and graph preparation are outside the timed benchmark. The repository
publishes model weights, preprocessing, graph structures, splits, thresholds and
reference scores. Users implement submissions, select a workload and run inference.
Accuracy and ROC-AUC are primary. Other quality and systems metrics are specified
in [Measurements](measurements.md).

The ReLU GCN replaces the earlier polynomial reference. The polynomial model,
its artifacts and its results are not active workloads; their prior versions
remain in Git history. No polynomial activation or encrypted approximation is
part of either current reference model.

See the [submission interface](submission-contract.md), [GCN dataset](dataset.md),
[TAM dataset](dataset-tam.md) and [operation comparison](operations.md).
