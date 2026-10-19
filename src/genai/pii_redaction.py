"""Detect and redact personally identifiable information (PII)."""

import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


def redact_pii(text: str) -> str:
    """Redact detected PII before text enters the RAG pipeline."""
    if not text:
        return text

    results = analyzer.analyze(text=text, language="en")
    redacted = anonymizer.anonymize(text=text, analyzer_results=results)
    text = redacted.text

    # Additional masking for common contact details.
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
        text,
    )
    text = re.sub(
        r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
        "[REDACTED_PHONE]",
        text,
    )

    return text


if __name__ == "__main__":
    sample = (
        "Contact John Smith at john.smith@example.com "
        "or call 9876543210 about the claim."
    )
    print("Original:", sample)
    print("Redacted:", redact_pii(sample))