import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Ensure console supports UTF-8 where possible without failing on Windows legacy charmaps
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Constants defined by problem specification
VALID_CATEGORIES = ["Billing", "Technical", "Account", "Shipping", "Other"]
VALID_PRIORITIES = ["Low", "Medium", "High", "Urgent"]
VALID_REVIEW_FLAGS = ["yes", "no"]
REQUIRED_INPUT_COLUMNS = ["id", "subject", "body"]
REQUIRED_OUTPUT_COLUMNS = ["id", "category", "priority", "needs_review"]


def setup_logger(name: str = "triage", log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a UTF-8 aware logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Stream handler (UTF-8 console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Optional file handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validates that input DataFrame satisfies requirements."""
    if df.empty:
        raise ValueError("Input dataset is empty.")

    missing_cols = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataset is missing required columns: {missing_cols}")


def validate_output_dataframe(df: pd.DataFrame) -> None:
    """Validates that output results DataFrame adheres strictly to required schema."""
    missing_cols = [col for col in REQUIRED_OUTPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Output dataset missing required columns: {missing_cols}")

    invalid_reviews = df[~df["needs_review"].isin(VALID_REVIEW_FLAGS)]
    if not invalid_reviews.empty:
        raise ValueError(f"Invalid needs_review values found: {invalid_reviews['needs_review'].unique().tolist()}")

    # For auto-triaged tickets (needs_review == 'no'), category and priority must be valid
    auto_triaged = df[df["needs_review"] == "no"]
    invalid_cats = auto_triaged[~auto_triaged["category"].isin(VALID_CATEGORIES)]
    if not invalid_cats.empty:
        raise ValueError(f"Invalid categories in auto-triaged tickets: {invalid_cats['category'].unique().tolist()}")

    invalid_priorities = auto_triaged[~auto_triaged["priority"].isin(VALID_PRIORITIES)]
    if not invalid_priorities.empty:
        raise ValueError(f"Invalid priorities in auto-triaged tickets: {invalid_priorities['priority'].unique().tolist()}")


def print_triage_summary(summary_stats: Dict[str, Any]) -> None:
    """Prints a beautiful and clear execution summary of how tickets were handled."""
    total = summary_stats.get("total_tickets", 0)
    auto = summary_stats.get("auto_triaged", 0)
    review = summary_stats.get("human_review", 0)
    auto_pct = (auto / total * 100) if total else 0
    review_pct = (review / total * 100) if total else 0

    print("\n" + "=" * 65)
    print("           SUPPORT TICKET TRIAGE EXECUTION REPORT")
    print("=" * 65)
    print(f" Total Tickets Processed : {total}")
    print(f" Auto-Triaged (Confident): {auto:4d} ({auto_pct:5.1f}%)")
    print(f" Sent to Human Review    : {review:4d} ({review_pct:5.1f}%)")
    print("-" * 65)

    print(" Breakdown by Predicted Category (Auto-Triaged):")
    cat_counts = summary_stats.get("category_distribution", {})
    for cat in VALID_CATEGORIES:
        count = cat_counts.get(cat, 0)
        pct = (count / auto * 100) if auto else 0
        bar = "#" * int(pct // 4)
        print(f"   - {cat:<12}: {count:3d} ({pct:5.1f}%)  {bar}")

    print("-" * 65)
    print(" Breakdown by Priority (Auto-Triaged):")
    priority_counts = summary_stats.get("priority_distribution", {})
    for prio in ["Urgent", "High", "Medium", "Low"]:
        count = priority_counts.get(prio, 0)
        pct = (count / auto * 100) if auto else 0
        bar = "#" * int(pct // 4)
        print(f"   - {prio:<12}: {count:3d} ({pct:5.1f}%)  {bar}")

    if "review_reasons" in summary_stats and summary_stats["review_reasons"]:
        print("-" * 65)
        print(" Reasons for Escalation to Human Review:")
        for reason, cnt in summary_stats["review_reasons"].items():
            print(f"   - {reason:<32}: {cnt:3d}")

    print("=" * 65 + "\n")
