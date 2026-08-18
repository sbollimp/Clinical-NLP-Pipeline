# Explainability

What each explanation in this pipeline actually claims, how it is computed, and — the part usually
left out — what it does **not** prove.

---

## The design constraint

An extraction pipeline nobody trusts is an extraction pipeline nobody uses. In a clinical setting the
reviewer is usually not the person who built the model, so "the model said so, here is a confidence
score" is not an answer. The question is always some form of *why this, in this note*.

That constraint drove the model choice. A clinical BERT would score higher on entity F1. It would
also only be explainable through post-hoc approximation. Here the models are linear, so:

```
logit(class) = intercept + Σ (feature value × feature weight)
```

The explanation is the largest terms of that sum. It is not a model of the model — it is the
arithmetic the model performed, restated. Nothing is sampled, nothing is fitted, and the reported
contributions add up exactly to the logit, which the HTML report prints so you can check.

That is a real trade-off, not a free win. It is stated plainly in the [model card](model_card.md),
and the tagger interface is shaped so a transformer backend can be dropped in — at which point LIME
or kernel SHAP become the right tools and slot into the same `explanation` dict.

---

## The four methods

### 1. Linear weight attribution — entity spans

`SequenceTagger.explain_span`

For each token of a span, the weight vector of the predicted BIO tag is dotted with the token's
active features. Because every feature is a 0/1 indicator, a feature's contribution *is* its weight.
Contributions are summed across the span's tokens and the top six are reported by absolute value.

```jsonc
{
  "method": "linear_weight_attribution",
  "confidence": 0.999,
  "top_features": [
    {"feature": "gaz=B-VITAL",             "contribution": 3.2611, "direction": "supports"},
    {"feature": "gaz_sec=B-VITAL|VITALS",  "contribution": 3.2611, "direction": "supports"},
    {"feature": "rule[1]=B-LAB_VALUE",     "contribution": 3.1872, "direction": "supports"},
    {"feature": "sec=VITALS",              "contribution": 3.1027, "direction": "supports"}
  ],
  "gazetteer_tag": "B-VITAL", "rule_tag": "O", "section": "VITALS"
}
```

Read: this token is in the dictionary as a vital, sits in the `VITALS` section, and is followed by
something the rules recognise as a value.

**What it does not prove.** Correlated features share credit arbitrarily — `gaz=B-VITAL` and
`gaz_sec=B-VITAL|VITALS` fire together and split the attribution between them, so neither number is
that feature's *causal* importance. Use the occlusion test (below) when the question is causal.

### 2. Linear contribution — document labels

`_BaseTextModel._attribute`

Same idea over the TF-IDF vector (plus structured indicators for the risk model): contribution =
`tfidf(term) × weight(term)`. Reported as ranked supporting and opposing terms, with the class
probability distribution and the margin over the runner-up.

```jsonc
{
  "task": "readmission_risk", "prediction": "medium", "confidence": 0.777,
  "method": "linear_contribution(tfidf+structured)",
  "logit": 2.1235, "intercept": 2.0298, "sum_term_contributions": 0.0937,
  "supporting_terms": [
    {"term": "struct:prior_admissions=1", "contribution": 0.4931},
    {"term": "struct:n_medications=4",    "contribution": 0.1557}
  ],
  "opposing_terms": [
    {"term": "struct:disposition=unspecified",  "contribution": -0.5355},
    {"term": "struct:n_negated_conditions=2",   "contribution": -0.5231}
  ]
}
```

Structured features read as plain statements by design — `prior_admissions=1` needs no glossary,
whereas a raw count of 1.0 with a weight of 0.49 does.

**What it does not prove.** A high contribution means the term moved *this* decision, given
everything else in the note. It does not mean the term is globally important, and it does not mean
the relationship is causal. See §"Instance-level is not enough".

### 3. Trigger provenance — assertion status

`AssertionDetector.classify`

Assertion is rule-based, so its explanation is complete rather than approximate: the literal cue, its
character offset, and the scope it opened.

```jsonc
{
  "assertion": "negated",
  "trigger": "no evidence of",
  "trigger_offset": 334,
  "scope_text": "No evidence of cellulitis",
  "rationale": "trigger 'no evidence of' opened a negated scope with no terminator before the concept"
}
```

A reviewer can verify this by reading the note. That is the whole point.

**What it does not prove.** The cue list is finite. A negation phrased outside it produces a silent
`present` with the rationale "no assertion trigger in scope" — which is honest, but is an absence of
evidence, not evidence of absence. Real notes negate in ways no fixed cue list covers.

### 4. Feature-family occlusion — counterfactual check

`SequenceTagger.occlusion_test`

Re-scores a span with each *family* of features removed in turn, reporting how far the probability
falls. Ablating whole families rather than single features sidesteps the shared-credit problem in §1.

Two real examples from [`explanations.jsonl`](../outputs/explanations/). First, a clean
in-dictionary mention:

```jsonc
{"span": "sepsis", "baseline_probability": 1.0,
 "probability_drop_when_removed": {"gaz": 0.002, "w": 0.0, "shape": 0.0, "sec": 0.0}}
```

Nothing moves it. The model is confident for redundant reasons — dictionary, section and context all
agree — which is comfortable but tells you little.

Now the same word, typo'd, in a note where it is not in the dictionary at all:

```jsonc
{"span": "srtoke", "baseline_probability": 0.9465,
 "probability_drop_when_removed": {"shape": 0.9419, "w": 0.9018, "suf": 0.0659,
                                   "rule": 0.0201, "pre": 0.0041, "gaz": -0.0254}}
```

Removing orthographic shape or the surrounding word window collapses the prediction; removing the
gazetteer features slightly *helps*, because the dictionary is actively voting against a mention it
does not recognise. This is the tagger doing the job the dictionary cannot — and it is visible only
because the counterfactual was run.

The diagnostic to watch for is the inverse: a large `gaz` drop with near-zero everywhere else means
the model is effectively reading the dictionary and will collapse on vocabulary the dictionary lacks.
That is the failure a clean overall F1 hides.

**What it does not prove.** Removing a feature moves the input off the training distribution, so the
resulting probability is an extrapolation. Treat the ordering as informative and the magnitudes as
approximate.

---

## Instance-level is not enough

The pipeline exports **global** model weights on every run, to
[`outputs/global_model_explanation.json`](../outputs/global_model_explanation.json): the top-weighted
terms per class for all three classifiers, and the top features per BIO tag for the tagger.

This is not redundant with per-prediction explanations — it catches a different class of problem, as
this repository demonstrates on itself.

The first version fed raw note text to the classifiers. Every individual explanation looked
reasonable. The global export did not:

```
readmission_risk / "low"    → "dr delacroix"    rank 22 of 5,494 features
readmission_risk / "medium" → "dr castellanos"  rank 57 of 5,494 features
```

Attending-physician names were among the most heavily weighted features — the model had partly
learned *who wrote the note*. No single note's explanation would reliably have surfaced that: in any
one note the name is one modest term among hundreds, and it looks unremarkable. It is only visible in
aggregate.

Classifier input now runs through the de-identifier, and the top risk terms are clinical:

```
struct:prior_admissions=3+ · struct:disposition=icu_transfer ·
struct:abnormal_labs=5 · struct:abnormal_vitals=5+ · struct:n_chronic=5+
```

Cost: 0.013 macro-F1. Worth it.

**The generalisable lesson:** instance-level explanations answer "why this prediction"; global
weights answer "what did this model learn". Shortcut learning lives almost entirely in the second
question, and exporting global weights on every run costs nothing.

---

## Confidence, and whether to believe it

A confidence score is only useful if it is calibrated, so calibration is measured every run:

| Confidence bin | n | accuracy |
| --- | --- | --- |
| 0.4–0.6 | 4 | 0.250 |
| 0.6–0.8 | 11 | 0.545 |
| 0.8–1.0 | 4,509 | 0.997 |

Expected calibration error **0.097**. The distribution is heavily concentrated at the top — which is
what makes selective review workable: spans above 0.8 are right 99.7% of the time, and the small
uncertain tail is where a reviewer's attention should go. `quality.needs_review` is built on exactly
this separation.

If that concentration ever flattens on your data, `needs_review` stops being a useful filter and the
threshold needs re-deriving. Re-check calibration whenever the input distribution changes.

---

## The HTML reports

[`outputs/explanations/index.html`](../outputs/explanations/) — one self-contained file per note.
Inline CSS, no scripts, no network requests, no build step, so a file can be attached to a validation
package and opened years later.

Each report contains:

- **The note, highlighted in place.** Entities colour-coded by type, badged with assertion status
  when it is not `present`, hoverable for type / assertion / confidence / source / canonical concept.
  Because offsets are preserved end to end, the highlights land on the clinician's own characters.
- **A card per document label** with contribution bars for supporting and opposing terms, the class
  distribution, and the logit arithmetic printed out.
- **An entity table** with a plain-language rationale per row, for readers who do not want weights:
  > Tagged CONDITION because it is a known term in the clinical dictionary (confidence 1.00). Marked
  > historical because the phrase `'history of'` appears in scope before it.
- **A counterfactual card** when occlusion was run.
- **A review banner** driven by the pipeline's own confidence signals.
- **The full structured record**, collapsed, so the extraction can be checked against the source.

---

## Adding a method

The explanation contract is a dict on each `Span` and each classifier result. To plug in LIME, SHAP,
integrated gradients or anything else:

1. Compute it wherever the prediction is made (`ner/tagger.py`, `classify/model.py`).
2. Return a dict with a `method` key and whatever payload the method produces.
3. Extend `explain/attributions.py` if it should appear in the digest, and `explain/report.py` if it
   should render.

Nothing downstream inspects the internals of an explanation dict, so a new method does not break
existing consumers.
