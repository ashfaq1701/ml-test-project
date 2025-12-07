"""Standalone entry point to run the linear classifier pipeline."""
from __future__ import annotations

from linear_classifier import run_linear_pipeline


def main() -> None:
    output_path = run_linear_pipeline()
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
