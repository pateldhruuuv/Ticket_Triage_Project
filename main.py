"""Main entry point for Support Ticket Triage Pipeline.

Executes the complete end-to-end triage flow:
tickets.csv -> preprocessing -> classifier -> triage -> results.csv
"""

import argparse
import sys
from pathlib import Path
from src.utils import setup_logger, print_triage_summary
from src.preprocessing import load_and_preprocess_tickets
from src.classifier import TicketClassifier
from src.triage import TriageEngine, DEFAULT_CONFIDENCE_THRESHOLD


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    preserve_tentative: bool = False,
) -> None:
    """Executes the complete ticket triage pipeline."""
    logger = setup_logger("triage_main")
    logger.info("Initializing Support Ticket Triage Pipeline...")

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file.resolve()}")
        sys.exit(1)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Preprocessing: Load, clean text, handle missing values
    logger.info(f"Stage 1: Preprocessing tickets from {input_file}...")
    preprocessed_df = load_and_preprocess_tickets(input_file)
    logger.info(f"Loaded and cleaned {len(preprocessed_df)} tickets.")

    # 2. Classification: Predict category, priority, and confidence
    logger.info("Stage 2: Running AI/ML classification and confidence scoring...")
    classifier = TicketClassifier()
    predictions_df = classifier.predict_dataframe(preprocessed_df)
    logger.info("Classification completed successfully.")

    # 3. Triage: Evaluate confidence, route between auto-triage and human review
    logger.info(f"Stage 3: Evaluating triage confidence (threshold={confidence_threshold})...")
    triage_engine = TriageEngine(
        confidence_threshold=confidence_threshold,
        preserve_tentative=preserve_tentative,
    )
    results_df, stats = triage_engine.triage_predictions(predictions_df)

    # 4. Export results.csv
    logger.info(f"Stage 4: Writing final results to {output_file}...")
    results_df.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"Results saved successfully to {output_file.resolve()}")

    # 5. Print comprehensive execution summary report
    print_triage_summary(stats)


def main():
    parser = argparse.ArgumentParser(
        description="Support Ticket Triage: AI-driven category & priority prediction with confidence-based human routing."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/tickets.csv",
        help="Path to input tickets CSV file (default: data/tickets.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/results.csv",
        help="Path to output results CSV file (default: output/results.csv)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for human review routing (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--preserve-tentative",
        action="store_true",
        help="Keep tentative predictions in results.csv even when needs_review is 'yes' (default: empty per spec)",
    )

    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        confidence_threshold=args.threshold,
        preserve_tentative=args.preserve_tentative,
    )


if __name__ == "__main__":
    main()
