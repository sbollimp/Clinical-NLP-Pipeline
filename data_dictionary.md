# Data dictionary

Every field of the structured output, typed and described. The canonical definitions live in
[`src/clinical_nlp/data/schema.py`](../src/clinical_nlp/data/schema.py); this document is what an
analyst consuming the data should read.

Two files carry the output:

- **[`outputs/structured_records.jsonl`](../outputs/structured_records.jsonl)** — one JSON object per
  note, nested, complete. This is the source of truth.
- **[`outputs/structured_records.csv`](../outputs/structured_records.csv)** — one flat row per note,
  lossy, for loading straight into a warehouse or spreadsheet.

---

## 1. Record — top level

| Field | Type | Null? | Description |
| --- | --- | --- | --- |
| `doc_id` | string | no | Stable identifier for the note. Carried from the source filename, id column, or generated corpus id. |
| `patient_id` | string | may be `""` | Patient identifier from source metadata. Empty when the source provided none. |
| `encounter_date` | string | yes | Date of service. Taken from source metadata if present, otherwise the first `DATE` entity in the note. Format follows the source; not parsed to a date type. |
| `note_type` | enum | yes | `discharge_summary` \| `progress_note` \| `ed_note` \| `consult_note`. Predicted. |
| `readmission_risk` | enum | yes | `low` \| `medium` \| `high`. Predicted. **Demonstration label only** — see the [model card](model_card.md). |
| `problem_flags` | array\<enum\> | no (may be `[]`) | Condition-group flags above threshold: `diabetes`, `heart_failure`, `copd`, `ckd`, `sepsis`, `atrial_fibrillation`, `pneumonia`, `hypertension`. |
| `problems` | array\<Problem\> | no | Affirmed conditions. Excludes negated / hypothetical / family-history mentions. |
| `medications` | array\<Medication\> | no | Medications with attached dose, route and frequency. |
| `labs` | array\<Lab\> | no | Laboratory tests with value, unit and abnormal flag. |
| `vitals` | array\<Vital\> | no | Vital-sign measurements. |
| `procedures` | array\<Mention\> | no | Procedures and imaging studies. |
| `allergies` | array\<Allergy\> | no | Allergens with reaction where stated. |
| `negated_mentions` | array\<Problem\> | no | Conditions found but deliberately excluded from `problems`, with the reason in `assertion`. |
| `explanations` | object | no | Explanation digest — see §8. |
| `quality` | object | no | Row-level QC signals — see §7. |

> **Read `negated_mentions` before you aggregate.** A condition appearing there means the note
> discussed it and the pipeline judged it *not* an active problem. Silently unioning it into
> `problems` reintroduces exactly the error this pipeline exists to prevent.

---

## 2. `problems[]` and `negated_mentions[]`

| Field | Type | Description |
| --- | --- | --- |
| `text` | string | The mention exactly as it appears in the note, including any typo or casing. |
| `concept` | string | Canonical concept name, e.g. `Community acquired pneumonia`. Falls back to `text` when unmapped. |
| `code` | string \| null | Illustrative ICD-10-style code. **Demonstration values — not an authoritative mapping.** |
| `assertion` | enum | `present` \| `historical` \| `negated` \| `hypothetical` \| `family_history`. In `problems[]` only the first two occur. |
| `section` | string | Enclosing clinical section: `HEADER`, `CHIEF_COMPLAINT`, `HPI`, `PMH`, `ALLERGIES`, `MEDICATIONS`, `VITALS`, `LABS`, `PROCEDURES`, `HOSPITAL_COURSE`, `ASSESSMENT`, `DISPOSITION`, `OTHER`. |
| `start`, `end` | int | Character offsets into the original note text. Half-open: `text == note[start:end]`. |
| `confidence` | float | 0–1. Mean token probability for statistical spans; a fixed rule/dictionary score otherwise. |
| `source` | enum | `hybrid` (≥2 recognisers agreed) \| `statistical` \| `rule`. |
| `mention_count` | int | Present only when the concept appeared more than once; the record keeps the highest-confidence mention. |

## 3. `medications[]`

| Field | Type | Description |
| --- | --- | --- |
| `name` | string | Canonical drug name. |
| `text` | string | Surface form in the note. |
| `code` | string \| null | Illustrative RxNorm-style code. |
| `dose` | string \| null | e.g. `40 mg`, `3.375 g`, `0.05 mcg/kg/min`. Line-scoped to this drug. |
| `route` | string \| null | `PO`, `IV`, `IM`, `SC`, `INH`, `NEB`, `PR`, `SL`, `TD`. |
| `frequency` | string \| null | `daily`, `BID`, `TID`, `q6h`, `PRN`, `nightly`, `continuous`, `once`. |
| `status` | enum | `active` \| `discontinued` (set when the mention is negated, e.g. "…discontinued this admission"). |
| `assertion` | enum | As above. |
| `section`, `start`, `end`, `confidence`, `mention_count` | | As above. |

A `null` dose means none was found **on the same line**, not that none exists. Paragraph-style
medication narrative is the known weak case; `quality.medications_missing_dose` lists these.

## 4. `labs[]`

| Field | Type | Description |
| --- | --- | --- |
| `test` | string | Canonical test name, e.g. `Hemoglobin A1C`. |
| `text` | string | Surface form. |
| `value` | string \| null | Value as written. |
| `value_numeric` | float \| null | Parsed value. `null` for non-numeric or composite values. |
| `unit` | string \| null | Unit token immediately following the value, e.g. `mg/dL`. |
| `abnormal_flag` | enum \| null | `H` \| `L` \| `ABNORMAL` \| `CRITICAL`, from a flag adjacent to the value. |
| `section`, `start`, `end`, `confidence` | | As above. |

## 5. `vitals[]`

| Field | Type | Description |
| --- | --- | --- |
| `measure` | string | `BP`, `HR`, `RR`, `Temp`, `SpO2`. |
| `value` | string \| null | As written. Blood pressure stays composite (`148/92`). |
| `unit` | string \| null | e.g. `mmHg`, `bpm`, `%`. |
| `start`, `end`, `confidence` | | As above. |

## 6. `allergies[]`

| Field | Type | Description |
| --- | --- | --- |
| `substance` | string | Canonical allergen. |
| `reaction` | string \| null | Parenthetical reaction where stated, e.g. `anaphylaxis`. |
| `section`, `start`, `end`, `confidence` | | As above. |

---

## 7. `quality` — row-level QC

Computed per note so consumers can triage without re-deriving it.

| Field | Type | Description |
| --- | --- | --- |
| `n_entities` | int | Total entities extracted. |
| `n_low_confidence` | int | Entities scoring below 0.6. |
| `low_confidence_examples` | array\<string\> | Up to 5 examples. |
| `n_unmapped_concepts` | int | Mentions with no dictionary match, exact or fuzzy. |
| `unmapped_examples` | array\<string\> | Up to 5 examples. |
| `medications_missing_dose` | array\<string\> | Drugs with no dose attached. |
| `labs_missing_value` | array\<string\> | Tests with no value attached. |
| `completeness_score` | float | 0–1. Share of medications and labs whose attributes were filled. |
| `needs_review` | bool | True when any extraction-side signal above fired, **or** when the classifier's confidence/margin was low. The single authoritative review flag — a record and its HTML report can never disagree. **The suggested filter for human review.** |
| `review_reasons` | array\<string\> | Why `needs_review` is true, e.g. `["2 unmapped concepts", "low classification confidence or margin"]`. Empty when it is false. |

## 8. `explanations` — digest

Records carry a digest; the full explanation objects (signed contributions for every active feature)
live in [`outputs/explanations/explanations.jsonl`](../outputs/explanations/) and in the HTML reports.

| Field | Type | Description |
| --- | --- | --- |
| `summary` | object | Entity counts by source and assertion, mean confidence, `review_recommended`. |
| `note_type` | object | `prediction`, `confidence`, top supporting terms with signed contributions. |
| `readmission_risk` | object | Same, including `struct:*` features from the extraction stage. |
| `problem_flags` | object | Predicted set plus per-label probabilities. |
| `rule_based_problem_flags` | array\<enum\> | Flags derived from affirmed `CONDITION` mentions instead of the classifier. Compare against `problem_flags` as a consistency check. |
| `structured_features_used_by_risk_model` | array\<string\> | Which indicator features were active for this note. |
| `entities[]` | array | Per entity: `text`, `label`, `assertion`, `concept`, `confidence`, and `why` — one plain-language sentence. |

---

## 9. CSV view

[`outputs/structured_records.csv`](../outputs/structured_records.csv) flattens each record to one
row. Nested lists become `; `-joined strings, so it is lossy — use the JSONL for anything beyond
inspection.

| Column | Description |
| --- | --- |
| `doc_id`, `patient_id`, `encounter_date`, `note_type`, `readmission_risk` | As above |
| `problem_flags` | Pipe-joined flags |
| `n_problems`, `problems` | Count, then `; `-joined canonical concepts |
| `n_medications`, `medications` | Count, then `; `-joined `name dose route frequency` |
| `n_labs`, `abnormal_labs` | Count, then `; `-joined `test=value unit` for flagged results only |
| `allergies`, `procedures`, `negated_mentions` | `; `-joined |
| `completeness_score`, `needs_review` | From `quality` |

---

## 10. Entity types (intermediate representation)

Present on `Span` objects and in the HTML reports; the record schema above groups them into fields.

| Type | Examples | Where it lands |
| --- | --- | --- |
| `CONDITION` | `sepsis`, `type 2 diabetes mellitus` | `problems` / `negated_mentions` |
| `MEDICATION` | `furosemide`, `piperacillin-tazobactam` | `medications.name` |
| `DOSAGE` | `40 mg`, `24 units` | `medications.dose` |
| `LAB` | `creatinine`, `hemoglobin A1c` | `labs.test` |
| `LAB_VALUE` | `2.41`, `148/92` | `labs.value`, `vitals.value` |
| `VITAL` | `BP`, `SpO2` | `vitals.measure` |
| `PROCEDURE` | `chest X-ray`, `hemodialysis` | `procedures` |
| `ALLERGY` | `penicillin`, `latex` | `allergies.substance` |
| `DATE` | `10/02/2024` | `encounter_date` |

`FREQUENCY` and `ROUTE` are extracted as *attributes* rather than scored entity types; they populate
medication fields and are excluded from NER metrics.
