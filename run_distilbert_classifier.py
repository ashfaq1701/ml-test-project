from distilbert_classifier import run_distilbert_pipeline


def main() -> None:
    output_path = run_distilbert_pipeline()
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
