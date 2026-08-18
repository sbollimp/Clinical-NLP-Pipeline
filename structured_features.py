"""Structured features derived from the extraction stage.

The readmission-risk label is not really a property of the *words* in a note --
it is a property of the patient's clinical picture: how many chronic problems,
how many results out of range, how many recent admissions, where they were
discharged to.  A bag-of-n-grams can only reach those quantities indirectly.

So the risk classifier is **stacked on top of the NER stage**: it sees TF-IDF
*and* a handful of indicator features computed from the entities the pipeline
just extracted.  This is the point of doing extraction at all -- the structured
output is not merely a deliverable, it is a better representation for the next
model.

Two deliberate choices:

* Features are **indicators, not raw counts** (``n_chronic=3`` rather than
  ``n_chronic: 3.0``).  They stay on the same scale as TF-IDF values, so no
  feature standardisation is needed, and each one reads as a plain statement in
  an explanation.
* At training time these features are computed from **predicted** entities, not
  gold ones, so the classifier learns to tolerate the extraction stage's real
  error profile instead of being surprised by it at inference.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from clinical_nlp.data import lexicons as lex
from clinical_nlp.data.schema import ClinicalDocument, Span

_PRIOR_ADMISSIONS = re.compile(r"had\s+(\d+)\s+hospital\s+admission", re.IGNORECASE)
_ABNORMAL_MARK = re.compile(r"\(\s*(?:H|L|ABNORMAL|CRITICAL)\s*\)")

DISPOSITION_PATTERNS = [
    ("icu_transfer", re.compile(r"transferred to icu", re.IGNORECASE)),
    ("skilled_nursing", re.compile(r"skilled nursing facility", re.IGNORECASE)),
    ("home_health", re.compile(r"home with home health", re.IGNORECASE)),
    ("home", re.compile(r"discharged home", re.IGNORECASE)),
]

_CHRONIC_CANONICAL = {c[1] for c in lex.CONDITIONS if c[4]}
_VITAL_RANGES = {name: normal for name, _unit, normal, _bad in lex.VITAL_TEMPLATES}

AFFIRMED = {"present", "historical"}


def _bucket(value: int, cap: int) -> str:
    return f"{min(value, cap)}{'+' if value >= cap else ''}"


def _value_after(doc_text: str, anchor: Span, spans: Sequence[Span]) -> Optional[Span]:
    line_end = doc_text.find("\n", anchor.end)
    line_end = line_end if line_end != -1 else len(doc_text)
    for span in spans:
        if span.label == "LAB_VALUE" and anchor.end <= span.start < line_end:
            return span
    return None


def _abnormal_vitals(doc: ClinicalDocument) -> int:
    spans = sorted(doc.pred_spans, key=lambda s: s.start)
    count = 0
    for span in spans:
        if span.label != "VITAL":
            continue
        name = (span.normalized or span.text).strip()
        bounds = _VITAL_RANGES.get(name)
        if bounds is None:
            continue
        value_span = _value_after(doc.text, span, spans)
        if value_span is None:
            continue
        raw = value_span.text.split("/")[0]
        try:
            value = float(raw)
        except ValueError:
            continue
        if value < bounds[0] or value > bounds[1]:
            count += 1
    return count


def structured_features(doc: ClinicalDocument) -> Dict[str, float]:
    """Indicator features describing the clinical picture extracted from a note."""
    spans = doc.pred_spans
    conditions = [s for s in spans if s.label == "CONDITION"]
    affirmed = [s for s in conditions if s.assertion in AFFIRMED]
    chronic = [s for s in affirmed if (s.normalized or "") in _CHRONIC_CANONICAL]
    negated = [s for s in conditions if s.assertion == "negated"]
    medications = [s for s in spans if s.label == "MEDICATION" and s.assertion != "negated"]
    labs = [s for s in spans if s.label == "LAB"]
    abnormal_labs = len(_ABNORMAL_MARK.findall(doc.text))
    abnormal_vitals = _abnormal_vitals(doc)

    prior_match = _PRIOR_ADMISSIONS.search(doc.text)
    prior = int(prior_match.group(1)) if prior_match else 0

    disposition = "unspecified"
    for name, pattern in DISPOSITION_PATTERNS:
        if pattern.search(doc.text):
            disposition = name
            break

    feats = {
        f"struct:n_chronic={_bucket(len(chronic), 5)}": 1.0,
        f"struct:n_affirmed_conditions={_bucket(len(affirmed), 8)}": 1.0,
        f"struct:n_negated_conditions={_bucket(len(negated), 3)}": 1.0,
        f"struct:n_medications={_bucket(len(medications), 6)}": 1.0,
        f"struct:n_labs={_bucket(len(labs), 8)}": 1.0,
        f"struct:abnormal_labs={_bucket(abnormal_labs, 6)}": 1.0,
        f"struct:abnormal_vitals={_bucket(abnormal_vitals, 5)}": 1.0,
        f"struct:prior_admissions={_bucket(prior, 3)}": 1.0,
        f"struct:disposition={disposition}": 1.0,
        f"struct:n_entities_bucket={_bucket(len(spans) // 10, 6)}": 1.0,
    }
    return feats


def batch_structured_features(docs: Sequence[ClinicalDocument]) -> List[Dict[str, float]]:
    return [structured_features(doc) for doc in docs]


def describe(features: Dict[str, float]) -> Dict[str, Any]:
    """Human-readable view of the indicator set, used in explanation payloads."""
    return {k.replace("struct:", ""): v for k, v in sorted(features.items())}
