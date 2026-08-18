# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026

First complete release.

### Added

**Ingestion and preprocessing**
- Loaders for text directories, CSV/TSV exports, JSONL, and the project's annotated corpus format,
  with ingestion QC (duplicate ids, empty and short notes).
- Length-preserving PHI scrubbing with an auditable redaction report.
- Length-preserving text normalisation, clinical section detection with alias mapping, and an
  offset-preserving clinical tokenizer.

**Extraction**
- Hybrid NER: a windowed logistic-regression BIO tagger, a longest-match gazetteer, and ten named
  regex extractors, merged precision-first with per-span recogniser provenance.
- Assertion detection over five classes (`present`, `negated`, `hypothetical`, `historical`,
  `family_history`) in the NegEx / ConText tradition, reporting the deciding trigger and its offset.
- Concept normalisation with an exact dictionary pass and a conservative fuzzy fallback.

**Classification**
- Note type (single-label), problem flags (multi-label, one-vs-rest), readmission risk (three-class).
- Risk classifier stacked on the extraction stage via structured indicator features, trained on
  *predicted* entities to avoid the stacked-model leak.

**Structuring**
- Entity-to-field assembler producing problems, medications (dose/route/frequency/status), labs
  (value/unit/abnormal flag), vitals, procedures and allergies, with line-scoped attribute
  attachment and assertion-driven inclusion.
- Row-level QC block: low-confidence spans, unmapped concepts, missing attributes, completeness
  score and a `needs_review` flag.

**Explainability**
- Exact linear attribution for entity spans and document labels.
- Trigger provenance for assertion decisions; rule id and description for regex-sourced spans.
- Feature-family occlusion as a counterfactual check.
- Global model weights exported every run.
- Self-contained HTML reports: annotated note, contribution bars, entity table with plain-language
  rationales, review banner.

**Evaluation**
- Strict and partial span metrics overall and per type, assertion accuracy over matched spans,
  per-class classification metrics, multi-label micro/macro/exact-set/Jaccard, confidence
  calibration with ECE, recogniser ablation, per-stage throughput.
- Markdown and JSON reports, plus a run manifest recording versions, platform and resolved config.

**Data**
- Deterministic synthetic corpus generator with character-perfect gold annotations at entity,
  assertion and document level, including deliberately held-out vocabulary and surface noise.

**Project**
- CLI (`generate` / `train` / `evaluate` / `predict` / `run`), YAML configuration, Makefile,
  57-test suite, architecture doc and diagram, model card, data dictionary, explainability doc.

### Changed during development

- **Tagger solver `liblinear` → `lbfgs`.** scikit-learn 1.8 removed multiclass support from
  `liblinear`; `lbfgs` fits a multinomial model whose per-class logits decompose just as exactly.
- **Risk classifier stacked on extracted structure.** TF-IDF alone reached 0.53 macro-F1 because the
  label depends on counts a bag of n-grams represents poorly; adding indicator features derived from
  the NER output moved it to 0.74.
- **Classifier input de-identified.** The global model explanation showed attending-physician names
  among the most heavily weighted risk features (`dr delacroix` at rank 22 of 5,494 for the `low`
  class). Scrubbing identifiers before vectorising removed the shortcut, at a cost of 0.013 macro-F1.
- **Records carry an explanation digest rather than the full object.** Embedding every signed feature
  contribution inline produced a 6.5 MB output file; the digest plus a separate
  `explanations/explanations.jsonl` audit file is 1.6 MB and more useful.
- **Generator hardened twice.** The first version was solved outright by dictionary lookup
  (F1 1.000 from the gazetteer alone). Held-out vocabulary, surface noise, prose-style medication and
  lab rendering, a free-text `HOSPITAL COURSE` section and section-header aliases brought
  dictionary-only performance down to 0.889 and made the statistical tagger's contribution measurable.
