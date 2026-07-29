import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SOURCE_RUNNER = (
    ROOT
    / "evaluation"
    / "baselines"
    / "basic_tool_use"
    / "run_basic_tool_use_baseline.py"
)


def load_source_runner():
    spec = importlib.util.spec_from_file_location(
        "basic_tool_use_source_runner",
        SOURCE_RUNNER,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load {SOURCE_RUNNER}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--model",
        default=None,
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=6.0,
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runner = load_source_runner()

    runner.TEST_CASES = dataset_path
    runner.BASE = output_dir

    runner.OUT_JSONL = (
        output_dir
        / "basic_tool_use_results_100.jsonl"
    )

    runner.OUT_CSV = (
        output_dir
        / "basic_tool_use_results_100.csv"
    )

    runner.OUT_METRICS = (
        output_dir
        / "basic_tool_use_metrics.txt"
    )

    child_argv = [
        str(SOURCE_RUNNER),
        "--sleep",
        str(args.sleep),
    ]

    if args.model:
        child_argv.extend(
            [
                "--model",
                args.model,
            ]
        )

    original_argv = sys.argv

    try:
        sys.argv = child_argv
        await runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    asyncio.run(main())
