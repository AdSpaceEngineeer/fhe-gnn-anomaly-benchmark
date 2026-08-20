"""Command-line interface for benchmark preparation and execution."""

from __future__ import annotations

import argparse

from .baseline import evaluate_plaintext_baseline, train_plaintext_baseline
from .benchmark import run_benchmark
from .dataset import prepare_yelpchi
from .model_bundle import write_model_bundle
from .sweep import run_retention_sweep
from .validation import validate_comparison_files, validate_result_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fhe-gnn-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-yelpchi", help="prepare separated YelpChi artifacts")
    prepare.add_argument("source")
    prepare.add_argument("output")
    prepare.add_argument("--seed", type=int, default=2026)

    train = commands.add_parser("train-baseline", help="train and freeze the plaintext GNN")
    train.add_argument("dataset")
    train.add_argument("model")
    train.add_argument("--hidden-features", type=int, default=16)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=2026)

    evaluate = commands.add_parser(
        "evaluate-baseline", help="report full-test plaintext quality and Q(k)"
    )
    evaluate.add_argument("dataset")
    evaluate.add_argument("model")
    evaluate.add_argument("output")
    evaluate.add_argument(
        "--retentions", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0)
    )

    bundle = commands.add_parser(
        "bundle-model", help="register compatible trained weights for automatic evaluation"
    )
    bundle.add_argument("model")
    bundle.add_argument("manifest")
    bundle.add_argument("--name", required=True)
    bundle.add_argument("--recommended-baseline", action="store_true")

    run = commands.add_parser("run", help="run one batch/retention configuration")
    run.add_argument("submission_manifest")
    run.add_argument("dataset")
    run.add_argument("model")
    run.add_argument("output")
    run.add_argument("--retention", type=float, required=True)
    run.add_argument("--batch-size", type=int, required=True)
    run.add_argument("--num-runs", type=int, default=3)
    run.add_argument("--seed", type=int, default=2026)

    sweep = commands.add_parser(
        "sweep", help="run canonical truncations and generate Q(k) summary/plot"
    )
    sweep.add_argument("submission_manifest")
    sweep.add_argument("dataset")
    sweep.add_argument("model")
    sweep.add_argument("output")
    sweep.add_argument("--batch-size", type=int, required=True)
    sweep.add_argument("--num-runs", type=int, default=3)
    sweep.add_argument("--seed", type=int, default=2026)
    sweep.add_argument("--retentions", type=float, nargs="+")

    validate = commands.add_parser("validate", help="validate result semantics")
    validate.add_argument("result")

    compare = commands.add_parser(
        "validate-comparison", help="verify result files use one identical workload"
    )
    compare.add_argument("results", nargs="+")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "prepare-yelpchi":
        metadata = prepare_yelpchi(arguments.source, arguments.output, seed=arguments.seed)
        print(f"Prepared YelpChi: {metadata['node_count']} nodes, {metadata['edge_count']} edges")
    elif arguments.command == "train-baseline":
        metrics = train_plaintext_baseline(
            arguments.dataset,
            arguments.model,
            hidden_features=arguments.hidden_features,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
        )
        print(f"Test Recall={metrics.recall:.4f} F1={metrics.f1:.4f} Accuracy={metrics.accuracy:.4f}")
    elif arguments.command == "evaluate-baseline":
        report = evaluate_plaintext_baseline(
            arguments.dataset,
            arguments.model,
            arguments.output,
            retentions=arguments.retentions,
        )
        metrics = report["evaluation"]["full_identifier_metrics"]
        print(
            f"Full-ID test Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} Accuracy={metrics['accuracy']:.4f}"
        )
    elif arguments.command == "bundle-model":
        path = write_model_bundle(
            arguments.model,
            arguments.manifest,
            name=arguments.name,
            recommended_baseline=arguments.recommended_baseline,
        )
        print(f"Model bundle: {path}")
    elif arguments.command == "run":
        paths = run_benchmark(
            arguments.submission_manifest,
            arguments.dataset,
            arguments.model,
            arguments.output,
            retention=arguments.retention,
            batch_size=arguments.batch_size,
            num_runs=arguments.num_runs,
            seed=arguments.seed,
        )
        print("\n".join(str(path) for path in paths))
    elif arguments.command == "sweep":
        options = {}
        if arguments.retentions is not None:
            options["retentions"] = arguments.retentions
        summary, plot = run_retention_sweep(
            arguments.submission_manifest,
            arguments.dataset,
            arguments.model,
            arguments.output,
            batch_size=arguments.batch_size,
            num_runs=arguments.num_runs,
            seed=arguments.seed,
            **options,
        )
        print(f"Summary: {summary}\nPlot: {plot}")
    elif arguments.command == "validate":
        validate_result_file(arguments.result)
        print(f"Valid: {arguments.result}")
    else:
        validate_comparison_files(arguments.results)
        print(f"Comparable workload: {len(arguments.results)} result files")


if __name__ == "__main__":
    main()
