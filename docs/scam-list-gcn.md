# Scam_List_GCN

Scam_List_GCN is a simplified attribute-reconstruction GCN inspired by Ding et al.'s
[DOMINANT implementation](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch).
It retains graph aggregation and an encoder/decoder, but omits DOMINANT's
structure-reconstruction branch. The frozen reference uses ReLU, not a polynomial
activation. It was retrained for 80 epochs with seed 42; the final epoch is frozen.

Let Abar = D^(-1/2)(A+I)D^(-1/2), using degrees after adding self-loops.

| Layer | Equation | Width |
|---|---|---|
| Encoder 1 | H1 = ReLU(Abar X W1 + b1) | 8 -> 64 |
| Encoder 2 | Z = ReLU(Abar H1 W2 + b2) | 64 -> 32 |
| Decoder 1 | Hd = ReLU(Abar Z W3 + b3) | 32 -> 64 |
| Decoder 2 | Xhat = Abar Hd W4 + b4 | 64 -> 8 |
| Anomaly score | s_i = mean((Xhat[i,S] - X[i,S])^2), S={4,6,7} | One score/event |

The output layer has no activation. All eight features are reconstructed, but
only the three protected features enter the anomaly score. The client decrypts
scores; the harness predicts anomaly when `score >= 4.11271162092986`.

## Plaintext baseline

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.992750 | 0.993400 |
| ROC-AUC | 0.997935 | 0.998725 |
| Recall | 0.901538 | 0.902709 |
| F1 | 0.909938 | 0.917397 |
| Precision | 0.918495 | 0.932570 |
| Average Precision | 0.960706 | 0.971232 |

The dataset is described in [Dataset](dataset.md). Loss uses normal training
nodes only, with double reconstruction weight on sensitive features. Full-graph
message passing includes all nodes; this is transductive evaluation. The threshold
is chosen on validation labels and remains fixed for test and encrypted inference.

Published weights and reference scores are in
[`scam-list-gcn-relu-100k-v2`](../artifacts/scam-list-gcn-relu-100k-v2/).
The harness evaluates float64 reference arithmetic using the frozen float32
weights. Recorded training metrics are retained separately. No training is needed
to submit an FHE implementation.
