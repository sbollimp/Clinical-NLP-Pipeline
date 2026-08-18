# Model card — Clinical Document NLP Pipeline

Following the model-card format of Mitchell et al. (2019), adapted for a multi-model pipeline.
Every number here is reproduced by `python -m clinical_nlp.cli --config configs/default.yaml run`
and stored in [`../outputs/metrics.json`](../outputs/metrics.json).

---

## 1. Model details

| | |
| --- | --- |
| **Name** | Clinical Document NLP Pipeline |
| **Version** | 1.0.0 |
| **Type** | Composite pipeline: hybrid sequence tagger + rule-based assertion detector + three linear text classifiers |
| **Date** | 2026 |
| **License** | MIT |
| **Framework** | scikit-learn 1.x, NumPy, SciPy. No pretrained weights, no downloads, no GPU. |

### Components

| Component | Model | Parameters / size |
| --- | --- | --- |
| Entity tagger | `DictVectorizer` → multinomial `LogisticRegression`, windowed features, greedy decode + BIO repair | ~47.4k features × 16 BIO tags |
| Gazetteer | Longest-match dictionary over open, hand-written clinical lexicons | 83 entries (20 further surfaces deliberately held out) |
| Rule extractors | 10 named regular expressions, each with an id and description | — |
| Assertion detector | Scoped trigger matcher (NegEx / ConText tradition) | ~45 cue phrases + 13 terminators |
| Note-type classifier | TF-IDF (1–2 grams) → `LogisticRegression` | 4 classes |
| Problem-flag classifier | TF-IDF → one-vs-rest `LogisticRegression`, threshold 0.5 | 8 labels |
| Risk classifier | TF-IDF **+ structured indicator features from the extraction stage** → `LogisticRegression` | 3 classes |

---

## 2. Intended use

**Intended.** A reference implementation and teaching artefact for clinical information extraction:
a demonstration of hybrid NER, assertion detection, entity-to-field structuring, and
explanation-per-prediction; a starting skeleton for a real pipeline whose components you intend to
replace; a benchmark harness for comparing extraction approaches on a reproducible corpus.

**Out of scope.**

- ❌ Clinical decision-making of any kind
- ❌ Any use on real patient data without a validated de-identification system upstream, institutional
  review, and clinical validation on your own data
- ❌ Automated coding, billing, or quality reporting
- ❌ Risk stratification affecting patient care or resource allocation
- ❌ Any regulated use — this is not a medical device and has no regulatory clearance

**Users.** Data scientists and engineers evaluating or building clinical NLP. Not clinicians, and not
patients.

---

## 3. Training data

**Synthetic, generated in-repository.** No real patient data was used, and none is redistributed.
[`src/clinical_nlp/data/generate.py`](../src/clinical_nlp/data/generate.py) assembles notes span by
span from templates plus randomised clinical content, so gold annotations are exact by construction.

| | |
| --- | --- |
| Corpus size | 600 notes (420 train / 90 dev / 90 test) |
| Mean length | ~1,620 characters |
| Entities | 30,086 total, ~50 per note |
| Note types | discharge summary (213), progress note (149), ED note (129), consult note (109) |
| Risk labels | medium (267), high (246), low (87) |
| Assertion mix | present 80.2%, historical 11.5%, negated 4.1%, hypothetical 3.2%, family history 1.1% |
| Seed | 13 — regenerates identically |

### Difficulty injected on purpose

A generator emitting only clean, in-dictionary text would let a dictionary lookup solve the task and
would make every metric here meaningless. Two mechanisms prevent that:

- **Held-out vocabulary** — ~20 surface forms (`sevelamer`, `septic shock`, `paracentesis`,
  `CKD stage 3`, …) are written into notes but withheld from the gazetteer, standing in for the
  20–40% of real-world mentions that fall outside any curated terminology.
- **Surface noise** — 15% of entity mentions receive a casing change or a typo
  (`metformin` → `metfomin`), defeating exact string matching.

Roughly a third of each note is free-text narrative (`HOSPITAL COURSE`) where entities appear in
running prose rather than in labelled fields, and section headers vary across aliases
(`MEDS` / `MEDICATION LIST` / `DISCHARGE MEDICATIONS`).

### What the synthetic data still does not capture

This is the most important limitation in this document. Real clinical notes have:

- **Copy-forward duplication** — the same paragraph repeated across days, sometimes stale
- **Dictation artefacts** — speech-recognition errors, run-on sentences, missing punctuation
- **Inconsistent structure** — missing sections, nested headers, tables pasted as ASCII, template
  boilerplate interleaved with narrative
- **Far richer abbreviation and shorthand use**, much of it institution-specific and ambiguous
  (`MS` = mitral stenosis / multiple sclerosis / morphine sulfate / mental status)
- **Temporal complexity** — relative dates, event ordering, "since last admission"
- **Genuine ambiguity** that human annotators disagree about, giving a real ceiling below 1.0

Consequently **all metrics below are an upper bound.** Expect materially lower numbers on real notes,
particularly for entity boundaries, assertion status, and anything depending on section detection.

---

## 4. Evaluation

Held-out test split: 90 notes, 4,523 gold entity spans. Seed 13. Full report:
[`../outputs/metrics.md`](../outputs/metrics.md).

### Entity recognition

| Criterion | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Strict (exact offsets + type) | 0.995 | 0.995 | **0.995** |
| Partial (overlap + type) | 0.999 | 0.999 | **0.999** |

Per type (strict F1): ALLERGY 1.000 · DATE 1.000 · DOSAGE 1.000 · LAB_VALUE 1.000 · VITAL 1.000 ·
LAB 0.997 · MEDICATION 0.996 · CONDITION 0.989 · PROCEDURE 0.974.

Recogniser ablation:

| Configuration | P | R | F1 |
| --- | --- | --- | --- |
| dictionary + rules only | 0.986 | 0.810 | 0.889 |
| statistical tagger only | 0.995 | 0.995 | 0.995 |
| hybrid | 0.995 | 0.995 | 0.995 |

Read this as: the dictionary is precise and blind, and the statistical tagger carries recall. The
hybrid layer's value is not a higher F1 — it is that spans corroborated by two recognisers are
99.97% precise versus 97.6% for tagger-only spans, which is what makes selective human review
possible.

### Assertion status

Accuracy on strictly matched spans: **0.989** (n = 4,502). The residual errors are concentrated in
`present` ↔ `historical` and in narrative negations phrased without a cue in the trigger list.

### Document classification

| Task | Metric | Score |
| --- | --- | --- |
| Note type | accuracy / macro-F1 | 1.000 / 1.000 |
| Problem flags | micro-F1 / macro-F1 / exact-set accuracy | 0.933 / 0.929 / 0.756 |
| Readmission risk | accuracy / macro-F1 | 0.733 / 0.743 |

Note type is effectively a giveaway in this corpus — the type is the first line of the note. Treat
1.000 as evidence that the plumbing works, not as evidence that note-type classification is easy.

Problem flags derived by rolling up affirmed `CONDITION` mentions from the NER stage reach micro-F1
**0.998**, well ahead of the 0.933 classifier. Both routes are computed every run and reported side
by side; disagreement between them is a standing consistency check.

### Calibration

Expected calibration error **0.097** over entity confidences. Spans above 0.8 confidence are correct
99.7% of the time; the sparse 0.4–0.8 band is correct roughly half the time. That separation is what
`quality.needs_review` is built on.

### Throughput

56.5 ms per note end to end on a single CPU core, of which 32.6 ms is explanation generation.

---

## 5. Known failure modes

| Failure | Cause | Mitigation in place | Residual risk |
| --- | --- | --- | --- |
| Missed entity outside the dictionary | Held-out / novel vocabulary | Statistical tagger uses affix, shape and context features | Higher on real notes with institution-specific shorthand |
| Wrong assertion on a narrative negation | Cue phrase not in the trigger list | Cue list is data-driven and easy to extend; every decision reports its trigger | Real notes negate in ways no fixed cue list covers |
| Concept normalised to the wrong canonical form | Fuzzy match on a typo'd mention | Cutoff fixed at 0.88, restricted to the same entity type; unmapped mentions are flagged rather than guessed | A wrong normalisation silently corrupts downstream counts |
| Dose bound to the wrong drug | Prose medication lists with several drugs on one line | Line-scoped attachment | Fails on paragraph-style medication narrative |
| Section mis-detected | Unrecognised header format | Unknown headers become `OTHER`, never dropped | Section-conditioned features degrade |
| Risk over/under-called | Small training set, three coarse classes | Stacked structured features, `class_weight="balanced"` | 0.743 macro-F1 — the weakest component, and labelled as such |
| Spurious shortcut features | Model latching onto template artefacts | Classifier input is de-identified; global weights are exported every run for inspection | Only catches shortcuts someone looks for |

### A shortcut that was actually caught

The first version vectorised raw note text for the classifiers. `global_model_explanation.json`
showed **attending-physician names among the most heavily weighted risk features** — `dr delacroix`
ranked 22nd of 5,494 features for the `low` class and `dr castellanos` 57th for `medium`. The model
had partly learned which clinician wrote the note. De-identifying classifier input removed that
pathway, at a cost of 0.013 macro-F1 (0.756 → 0.743).

Two things are worth taking from this. First, exporting global model weights on every run is cheap
and it works. Second, per-prediction explanations would *not* have surfaced this — a single note's
explanation looked perfectly reasonable. Instance-level and model-level explanation answer different
questions and you need both.

---

## 6. Ethical considerations

**Privacy.** The bundled de-identifier removes field-level direct identifiers only. It is a second
line of defence, not a compliance control, and it does not constitute HIPAA Safe Harbor or Expert
Determination de-identification. Use a validated system (Philter, MIST, or a vendor solution)
upstream of this pipeline for any real data.

**Bias.** The synthetic corpus encodes the generator author's assumptions about which conditions,
drugs and phrasings are common. It is not demographically representative, and the risk label is a
deterministic function of a hand-written scoring formula rather than an observed outcome. **Nothing
about risk performance here transfers to real readmission prediction.** Deployed clinical risk models
have well-documented histories of encoding access-to-care disparities as if they were clinical
severity; any real version of this component needs subgroup evaluation before it is used for
anything.

**Automation bias.** Structured output looks authoritative in a way free text does not. That is why
every record carries confidence scores, a `needs_review` flag, and a completeness score, and why the
HTML reports lead with the note rather than with the extraction. The intended posture is
human-in-the-loop review, not autonomous extraction.

**Assertion errors are asymmetric.** Extracting a negated condition as present adds a false diagnosis
to a patient's derived record. Missing a present one omits a real diagnosis. These are not equally
costly and the appropriate operating point depends on the downstream use — tune
`multilabel_threshold` and `min_span_score` accordingly rather than accepting the defaults.

**Terminology codes are illustrative.** The ICD-10 / RxNorm-style identifiers in the lexicons were
written for demonstration. Do not treat them as authoritative mappings.

---

## 7. Reproducing these numbers

```bash
pip install -r requirements.txt
python -m clinical_nlp.cli --config configs/default.yaml run
```

Same config plus same seed reproduces every figure exactly. The run writes
[`../outputs/run_manifest.json`](../outputs/run_manifest.json) recording package version, Python
version, platform, the fully resolved config, and the headline metrics.

## 8. References

The approaches implemented here follow well-established clinical NLP work: NegEx (Chapman et al.,
2001) and ConText (Harkema et al., 2009) for assertion scoping; the hybrid dictionary/statistical
design familiar from cTAKES, MetaMap and CLAMP; the model-card format from Mitchell et al. (2019);
and the section-detection and structuring conventions common to i2b2/n2c2 shared-task systems. The
implementations are original and use no licensed terminology content.
