"""Preprocessing module for support ticket triage.

Handles loading raw ticket CSV, cleaning subject and body text,
imputing missing values, and detecting degraded or uninformative inputs.
"""

import re
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from .utils import validate_input_dataframe


# Common email boilerplate patterns to strip for cleaner intent extraction
SALUTATION_PATTERNS = [
    r"^(?:hello|hi|hey|dear)(?:\s+support|\s+team|\s+there|\s+all)?[,.:!\s]*",
    r"^(?:to\s+whom\s+it\s+may\s+concern)[,.:!\s]*",
    r"^(?:good\s+morning|good\s+afternoon|good\s+evening)[,.:!\s]*",
    r"^(?:guten\s+tag|bonjour|hola|ol[áa]|ciao|buongiorno|नमस्ते)[,.:!\s]*",
]

SIGNATURE_PATTERNS = [
    r"(?:kind\s+regards|best\s+regards|regards|thanks\s+in\s+advance|thank\s+you|thanks|cheers|danke|merci|gracias|obrigado)[\s\S]*$",
    r"(?:sent\s+from\s+my\s+(?:iphone|phone|android|mobile))[\s\S]*$",
]

# Patterns for obvious gibberish or uninformative input
GIBBERISH_PATTERN = re.compile(r"^(?:asdf|test|xyz|abc|qwerty|\d+|\?+|\.+|\!+|\s+)+$", re.IGNORECASE)
PLACEHOLDER_WORDS = {"asdf", "test", "xyz", "abc", "123", "qwerty"}


def clean_text_field(text: Any) -> str:
    """Normalizes and cleans a single text field."""
    if pd.isna(text) or text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    # Normalize line breaks and remove control characters
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    # Normalize excessive spaces and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def strip_email_boilerplate(text: str) -> str:
    """Removes standard greetings and signatures while preserving the core complaint."""
    if not text:
        return ""

    cleaned = text
    # Strip leading salutation
    for pat in SALUTATION_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing signature if it's separated by newline or punctuation
    for pat in SIGNATURE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # If stripping removed everything (e.g. ticket was literally just "Thank you"), revert
    return cleaned if cleaned else text


def detect_text_degraded(subject: str, body: str) -> Tuple[bool, str]:
    """Detects whether ticket text is degraded, gibberish, or insufficient to classify."""
    comb = f"{subject} {body}".strip()
    if not comb:
        return True, "Empty subject and body"

    if subject.lower() in ["(no subject)", "no subject", ""] and not body:
        return True, "Missing body and subject"

    if GIBBERISH_PATTERN.match(comb):
        return True, "Gibberish or test text"

    words = re.findall(r"\w+", comb.lower())
    if words and set(words).issubset(PLACEHOLDER_WORDS):
        return True, "Gibberish or test text"

    unique_words = set(words)
    if unique_words.issubset({"help", "test", "issue", "problem", "pls", "please"}):
        return True, f"Vague/insufficient context ('{comb}')"

    if len(words) < 2 and len(comb) < 8:
        return True, f"Extremely short input ('{comb}')"

    return False, ""


def preprocess_ticket(row: pd.Series) -> Dict[str, Any]:
    """Preprocesses a single ticket row."""
    raw_subj = clean_text_field(row.get("subject", ""))
    raw_body = clean_text_field(row.get("body", ""))

    clean_subj = raw_subj
    if clean_subj.lower() in ["(no subject)", "no subject"]:
        clean_subj = ""

    clean_body = strip_email_boilerplate(raw_body)

    # Detect if text is degraded
    is_degraded, degrade_reason = detect_text_degraded(clean_subj, clean_body)

    # Build full text representation for classification
    # Weight subject by placing it prominently at the front
    if clean_subj and clean_body:
        full_text = f"{clean_subj}. {clean_body}"
    elif clean_subj:
        full_text = clean_subj
    else:
        full_text = clean_body

    return {
        "id": row.get("id"),
        "customer": row.get("customer", ""),
        "received_at": row.get("received_at", ""),
        "raw_subject": raw_subj,
        "raw_body": raw_body,
        "clean_subject": clean_subj,
        "clean_body": clean_body,
        "full_text": full_text,
        "is_degraded": is_degraded,
        "degrade_reason": degrade_reason,
        "word_count": len(re.findall(r"\w+", full_text)),
    }


def load_and_preprocess_tickets(file_path: str | Path) -> pd.DataFrame:
    """Loads a tickets CSV and returns a fully preprocessed DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tickets file not found at {path}")

    # Read UTF-8 CSV handling multiline quoted records
    df = pd.read_csv(path, encoding="utf-8")
    validate_input_dataframe(df)

    processed_rows = [preprocess_ticket(row) for _, row in df.iterrows()]
    return pd.DataFrame(processed_rows)
