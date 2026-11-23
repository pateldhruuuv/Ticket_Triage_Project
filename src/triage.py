"""Triage module for Support Ticket Triage.

Evaluates confidence metrics, applies safety guards, and makes automated routing
decisions between automated categorization and escalation to human review.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from .utils import VALID_CATEGORIES, VALID_PRIORITIES, validate_output_dataframe

DEFAULT_CONFIDENCE_THRESHOLD = 0.50
MIN_CATEGORY_CONFIDENCE = 0.40
MIN_CATEGORY_MARGIN = 0.08
MIN_PRIORITY_CONFIDENCE = 0.40


class TriageEngine:
    """Decision engine that determines whether to auto-triage or escalate to human review."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_cat_confidence: float = MIN_CATEGORY_CONFIDENCE,
        min_cat_margin: float = MIN_CATEGORY_MARGIN,
        min_pri_confidence: float = MIN_PRIORITY_CONFIDENCE,
        preserve_tentative: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_cat_confidence = min_cat_confidence
        self.min_cat_margin = min_cat_margin
        self.min_pri_confidence = min_pri_confidence
        self.preserve_tentative = preserve_tentative

    def evaluate_ticket(self, prediction: Dict[str, Any]) -> Tuple[str, str, str, str]:
        """Evaluates a single ticket prediction and returns:

        (category, priority, needs_review, review_reason)
        """
        is_degraded = prediction.get("is_degraded", False)
        degrade_reason = prediction.get("degrade_reason", "")
        cat_conf = prediction.get("category_confidence", 0.0)
        cat_margin = prediction.get("category_margin", 0.0)
        pri_conf = prediction.get("priority_confidence", 0.0)
        overall_conf = prediction.get("overall_confidence", 0.0)
        pred_cat = prediction.get("category", "Other")
        pred_pri = prediction.get("priority", "Low")

        # 1. Guard against degraded/empty/gibberish input
        if is_degraded:
            review_reason = f"Degraded text: {degrade_reason}"
            cat_out = pred_cat if self.preserve_tentative else ""
            pri_out = pred_pri if self.preserve_tentative else ""
            return cat_out, pri_out, "yes", review_reason

        # 2. Check category confidence and margin
        if cat_conf < self.min_cat_confidence:
            review_reason = f"Low category confidence ({cat_conf:.2f} < {self.min_cat_confidence:.2f})"
            cat_out = pred_cat if self.preserve_tentative else ""
            pri_out = pred_pri if self.preserve_tentative else ""
            return cat_out, pri_out, "yes", review_reason

        if cat_margin < self.min_cat_margin:
            review_reason = f"Ambiguous category margin ({cat_margin:.2f} < {self.min_cat_margin:.2f})"
            cat_out = pred_cat if self.preserve_tentative else ""
            pri_out = pred_pri if self.preserve_tentative else ""
            return cat_out, pri_out, "yes", review_reason

        # 3. Check priority confidence
        if pri_conf < self.min_pri_confidence:
            review_reason = f"Low priority confidence ({pri_conf:.2f} < {self.min_pri_confidence:.2f})"
            cat_out = pred_cat if self.preserve_tentative else ""
            pri_out = pred_pri if self.preserve_tentative else ""
            return cat_out, pri_out, "yes", review_reason

        # 4. Check composite overall confidence threshold
        if overall_conf < self.confidence_threshold:
            review_reason = f"Overall confidence below threshold ({overall_conf:.2f} < {self.confidence_threshold:.2f})"
            cat_out = pred_cat if self.preserve_tentative else ""
            pri_out = pred_pri if self.preserve_tentative else ""
            return cat_out, pri_out, "yes", review_reason

        # Confident pass
        return pred_cat, pred_pri, "no", ""

    def triage_predictions(self, predictions_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Processes predictions DataFrame and returns:

        (results_df, summary_statistics)
        """
        results = []
        review_reasons = {}
        category_counts = {cat: 0 for cat in VALID_CATEGORIES}
        priority_counts = {pri: 0 for pri in VALID_PRIORITIES}
        auto_count = 0
        review_count = 0

        for _, row in predictions_df.iterrows():
            row_dict = row.to_dict()
            cat, pri, needs_review, reason = self.evaluate_ticket(row_dict)

            results.append({
                "id": row_dict["id"],
                "category": cat,
                "priority": pri,
                "needs_review": needs_review,
            })

            if needs_review == "no":
                auto_count += 1
                category_counts[cat] = category_counts.get(cat, 0) + 1
                priority_counts[pri] = priority_counts.get(pri, 0) + 1
            else:
                review_count += 1
                # Group reason key
                main_reason = reason.split(":")[0].split("(")[0].strip()
                review_reasons[main_reason] = review_reasons.get(main_reason, 0) + 1

        results_df = pd.DataFrame(results)
        validate_output_dataframe(results_df)

        stats = {
            "total_tickets": len(results_df),
            "auto_triaged": auto_count,
            "human_review": review_count,
            "category_distribution": category_counts,
            "priority_distribution": priority_counts,
            "review_reasons": review_reasons,
        }

        return results_df, stats
