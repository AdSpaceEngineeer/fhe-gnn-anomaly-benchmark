"""Plaintext reference for a degree-2 message-passing GNN.

Training is provided only to produce frozen benchmark weights. It is outside
the timed FHE inference workload. The encrypted server computation ends at the
logits; sigmoid and thresholding are client-side postprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .identifiers import EncodedIdentifier, IDENTIFIER_ALPHABET


IDENTIFIER_FEATURE_COUNT = 2 * len(IDENTIFIER_ALPHABET)


def identifier_feature_matrix(encodings: Sequence[EncodedIdentifier]) -> np.ndarray:
    """Create normalized prefix/suffix character histograms.

    The canonical logical representation is one encrypted one-hot vector per
    retained character. Summing those vectors produces 36 prefix and 36 suffix
    features. Implementations may pack the representation differently without
    changing these semantics.
    """

    if not encodings:
        raise ValueError("encodings must not be empty")
    matrix = np.zeros((len(encodings), IDENTIFIER_FEATURE_COUNT), dtype=np.float64)
    for row, encoded in enumerate(encodings):
        if len(encoded.values) != encoded.retained_characters:
            raise ValueError("encoded value count does not match retained characters")
        for value, side in zip(encoded.values, encoded.sides, strict=True):
            matrix[row, side * len(IDENTIFIER_ALPHABET) + value] += 1 / encoded.full_length
    return matrix


def mean_neighbour_features(features: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Aggregate directed incoming edge features with a zero-degree fallback."""

    values = np.asarray(features, dtype=np.float64)
    links = np.asarray(edges, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if links.ndim != 2 or links.shape[1] != 2:
        raise ValueError("edges must have shape (edge_count, 2)")
    if len(links) and (links.min() < 0 or links.max() >= len(values)):
        raise ValueError("edge endpoint lies outside the node range")
    aggregate = np.zeros_like(values)
    degree = np.zeros(len(values), dtype=np.int64)
    if len(links):
        sources = links[:, 0]
        targets = links[:, 1]
        np.add.at(aggregate, targets, values[sources])
        np.add.at(degree, targets, 1)
    nonzero = degree > 0
    aggregate[nonzero] /= degree[nonzero, None]
    return aggregate


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -40, 40)
    return 1 / (1 + np.exp(-clipped))


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select a deterministic validation threshold, preferring higher recall."""

    expected = np.asarray(labels, dtype=np.int8).reshape(-1)
    observed = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(expected) != len(observed) or not len(expected):
        raise ValueError("labels and scores must be non-empty and aligned")
    candidates = np.unique(np.concatenate(([0.0], observed, [1.0])))
    best = (-1.0, -1.0, 0.5)
    for threshold in candidates:
        predicted = observed >= threshold
        true_positives = int(np.sum((expected == 1) & predicted))
        false_positives = int(np.sum((expected == 0) & predicted))
        false_negatives = int(np.sum((expected == 1) & ~predicted))
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0
        candidate = (f1, recall, -float(threshold))
        if candidate > best:
            best = candidate
    return -best[2]


@dataclass(slots=True)
class PolynomialMessagePassingGNN:
    """One-hop GNN with an FHE-friendly degree-2 hidden activation."""

    self_weights: np.ndarray
    neighbour_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    threshold: float = 0.5
    activation_quadratic: float = 0.125

    @classmethod
    def initialize(cls, input_features: int, hidden_features: int, *, seed: int) -> "PolynomialMessagePassingGNN":
        if input_features < 1 or hidden_features < 1:
            raise ValueError("model dimensions must be positive")
        generator = np.random.default_rng(seed)
        scale = np.sqrt(1 / input_features)
        return cls(
            self_weights=generator.normal(0, scale, (input_features, hidden_features)),
            neighbour_weights=generator.normal(0, scale, (input_features, hidden_features)),
            hidden_bias=np.zeros(hidden_features),
            output_weights=generator.normal(0, np.sqrt(1 / hidden_features), hidden_features),
            output_bias=0.0,
            feature_mean=np.zeros(input_features),
            feature_scale=np.ones(input_features),
        )

    def _normalized_inputs(self, features: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("feature matrix does not match model input dimension")
        normalized = (values - self.feature_mean) / self.feature_scale
        return normalized, mean_neighbour_features(normalized, edges)

    def logits(self, features: np.ndarray, edges: np.ndarray) -> np.ndarray:
        values, neighbours = self._normalized_inputs(features, edges)
        preactivation = (
            values @ self.self_weights
            + neighbours @ self.neighbour_weights
            + self.hidden_bias
        )
        hidden = preactivation + self.activation_quadratic * preactivation**2
        return hidden @ self.output_weights + self.output_bias

    def scores(self, features: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(features, edges))

    def fit(
        self,
        features: np.ndarray,
        edges: np.ndarray,
        labels: np.ndarray,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        *,
        epochs: int = 200,
        learning_rate: float = 0.01,
        l2: float = 1e-4,
    ) -> None:
        """Fit frozen weights with deterministic full-batch Adam."""

        values = np.asarray(features, dtype=np.float64)
        expected = np.asarray(labels, dtype=np.float64).reshape(-1)
        train = np.asarray(train_indices, dtype=np.int64).reshape(-1)
        validation = np.asarray(validation_indices, dtype=np.int64).reshape(-1)
        if len(values) != len(expected) or not len(train) or not len(validation):
            raise ValueError("training arrays must be aligned and splits non-empty")
        if epochs < 1 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")

        self.feature_mean = values[train].mean(axis=0)
        self.feature_scale = values[train].std(axis=0)
        self.feature_scale[self.feature_scale < 1e-8] = 1.0
        normalized, neighbours = self._normalized_inputs(values, edges)
        train_labels = expected[train]
        positives = float(train_labels.sum())
        negatives = float(len(train_labels) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError("training split must contain both classes")
        sample_weights = np.where(train_labels == 1, negatives / positives, 1.0)
        sample_weights /= sample_weights.mean()

        parameters = [
            self.self_weights,
            self.neighbour_weights,
            self.hidden_bias,
            self.output_weights,
        ]
        first_moments = [np.zeros_like(parameter) for parameter in parameters]
        second_moments = [np.zeros_like(parameter) for parameter in parameters]
        output_first = 0.0
        output_second = 0.0
        beta1, beta2 = 0.9, 0.999

        for step in range(1, epochs + 1):
            train_values = normalized[train]
            train_neighbours = neighbours[train]
            preactivation = (
                train_values @ self.self_weights
                + train_neighbours @ self.neighbour_weights
                + self.hidden_bias
            )
            hidden = preactivation + self.activation_quadratic * preactivation**2
            probabilities = sigmoid(hidden @ self.output_weights + self.output_bias)
            derivative = (probabilities - train_labels) * sample_weights / len(train)
            gradients = [
                train_values.T @ ((derivative[:, None] * self.output_weights) * (1 + 2 * self.activation_quadratic * preactivation)) + l2 * self.self_weights,
                train_neighbours.T @ ((derivative[:, None] * self.output_weights) * (1 + 2 * self.activation_quadratic * preactivation)) + l2 * self.neighbour_weights,
                np.sum((derivative[:, None] * self.output_weights) * (1 + 2 * self.activation_quadratic * preactivation), axis=0),
                hidden.T @ derivative + l2 * self.output_weights,
            ]
            gradient_norm = np.sqrt(sum(float(np.sum(gradient**2)) for gradient in gradients))
            if gradient_norm > 5:
                gradients = [gradient * (5 / gradient_norm) for gradient in gradients]
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
                first_moments[index] = beta1 * first_moments[index] + (1 - beta1) * gradient
                second_moments[index] = beta2 * second_moments[index] + (1 - beta2) * gradient**2
                corrected_first = first_moments[index] / (1 - beta1**step)
                corrected_second = second_moments[index] / (1 - beta2**step)
                parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            bias_gradient = float(np.sum(derivative))
            output_first = beta1 * output_first + (1 - beta1) * bias_gradient
            output_second = beta2 * output_second + (1 - beta2) * bias_gradient**2
            self.output_bias -= learning_rate * (output_first / (1 - beta1**step)) / (
                np.sqrt(output_second / (1 - beta2**step)) + 1e-8
            )

        validation_scores = self.scores(values, edges)[validation]
        self.threshold = select_f1_threshold(expected[validation].astype(np.int8), validation_scores)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            self_weights=self.self_weights,
            neighbour_weights=self.neighbour_weights,
            hidden_bias=self.hidden_bias,
            output_weights=self.output_weights,
            output_bias=np.array(self.output_bias),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            threshold=np.array(self.threshold),
            activation_quadratic=np.array(self.activation_quadratic),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PolynomialMessagePassingGNN":
        with np.load(Path(path)) as values:
            return cls(
                self_weights=values["self_weights"],
                neighbour_weights=values["neighbour_weights"],
                hidden_bias=values["hidden_bias"],
                output_weights=values["output_weights"],
                output_bias=float(values["output_bias"]),
                feature_mean=values["feature_mean"],
                feature_scale=values["feature_scale"],
                threshold=float(values["threshold"]),
                activation_quadratic=float(values["activation_quadratic"]),
            )
