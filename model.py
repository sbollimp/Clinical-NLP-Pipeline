"""Document-level text classification.

Three tasks run over a shared representation:

``note_type``          single-label -- what kind of note is this?
``problem_flags``      multi-label  -- which conditions is this encounter about?
``readmission_risk``   ordinal      -- low / medium / high

All three use ``TfidfVectorizer`` -> ``LogisticRegression``.  As with the
tagger, the linear form is chosen so that a prediction can be decomposed
*exactly* into the features that produced it: the logit is
``b + sum_f x_f * w_f``, so the per-term contribution reported to a reviewer is
the arithmetic the model actually performed, not a surrogate fitted to it.

Three clinical-domain details matter here:

* The token pattern keeps single characters, because ``(H)`` and ``(L)``
  abnormal-result flags are among the most informative tokens in a note.
* ``problem_flags`` is genuinely multi-label -- a patient can be septic *and*
  diabetic -- so it is trained one-vs-rest with a per-label threshold rather
  than being forced into a single class.
* ``readmission_risk`` is **stacked on the extraction stage**: it consumes the
  indicator features in :mod:`clinical_nlp.classify.structured_features`
  alongside TF-IDF, because risk depends on counts (chronic problems, results
  out of range, prior admissions) that bag-of-n-grams represents poorly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from clinical_nlp.data.schema import ClinicalDocument
from clinical_nlp.utils.logging import get_logger

LOGGER = get_logger(__name__)

FeatureDict = Dict[str, float]


@dataclass
class ClassifierConfig:
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_features: int = 60000
    sublinear_tf: bool = True
    C: float = 4.0
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"
    top_terms: int = 8
    multilabel_threshold: float = 0.5
    token_pattern: str = r"(?u)\b\w+\b"
    random_state: int = 13


class _BaseTextModel:
    """Shared TF-IDF (+ optional structured) plumbing and exact attribution."""

    def __init__(self, config: Optional[ClassifierConfig] = None, use_structured: bool = False) -> None:
        self.config = config or ClassifierConfig()
        self.use_structured = use_structured
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_features=self.config.max_features,
            sublinear_tf=self.config.sublinear_tf,
            token_pattern=self.config.token_pattern,
            lowercase=True,
        )
        self.struct_vectorizer: Optional[DictVectorizer] = (
            DictVectorizer(sparse=True) if use_structured else None
        )
        self.feature_names_: np.ndarray = np.array([])

    # -- vectorisation ----------------------------------------------------

    def _fit_vectorizer(
        self, texts: Sequence[str], structured: Optional[Sequence[FeatureDict]] = None
    ):
        X = self.vectorizer.fit_transform(texts)
        names = list(self.vectorizer.get_feature_names_out())
        if self.use_structured:
            if structured is None:
                raise ValueError("structured features required but not provided")
            S = self.struct_vectorizer.fit_transform(list(structured))  # type: ignore[union-attr]
            names += list(self.struct_vectorizer.get_feature_names_out())  # type: ignore[union-attr]
            X = sp.hstack([X, S]).tocsr()
        self.feature_names_ = np.array(names, dtype=object)
        LOGGER.info(
            "%s: %d docs x %d features%s",
            type(self).__name__,
            X.shape[0],
            X.shape[1],
            " (tf-idf + structured)" if self.use_structured else "",
        )
        return X

    def _transform(
        self, texts: Sequence[str], structured: Optional[Sequence[FeatureDict]] = None
    ):
        X = self.vectorizer.transform(list(texts))
        if self.use_structured:
            S = self.struct_vectorizer.transform(list(structured or [{}] * len(list(texts))))  # type: ignore[union-attr]
            X = sp.hstack([X, S]).tocsr()
        return X

    # -- attribution ------------------------------------------------------

    def _attribute(
        self,
        text: str,
        structured: Optional[FeatureDict],
        coef: np.ndarray,
        intercept: float,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Exact decomposition of one linear decision for one document."""
        k = k or self.config.top_terms
        x = self._transform([text], [structured] if structured is not None else None)
        contributions = x.multiply(coef).tocoo()
        pairs = [
            (str(self.feature_names_[j]), float(v))
            for j, v in zip(contributions.col, contributions.data)
        ]
        pairs.sort(key=lambda kv: -abs(kv[1]))
        supporting = [p for p in pairs if p[1] > 0][:k]
        opposing = [p for p in pairs if p[1] < 0][:k]
        total = float(sum(v for _t, v in pairs))
        return {
            "method": "linear_contribution(tfidf" + ("+structured" if self.use_structured else "") + ")",
            "logit": round(total + float(intercept), 4),
            "intercept": round(float(intercept), 4),
            "sum_term_contributions": round(total, 4),
            "supporting_terms": [{"term": t, "contribution": round(v, 4)} for t, v in supporting],
            "opposing_terms": [{"term": t, "contribution": round(v, 4)} for t, v in opposing],
            "n_active_terms": len(pairs),
        }


class SingleLabelClassifier(_BaseTextModel):
    """Multinomial text classifier (used for note type and risk level)."""

    def __init__(
        self,
        name: str,
        config: Optional[ClassifierConfig] = None,
        use_structured: bool = False,
    ) -> None:
        super().__init__(config, use_structured)
        self.name = name
        self.model = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
        )
        self.classes_: List[str] = []

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        structured: Optional[Sequence[FeatureDict]] = None,
    ) -> "SingleLabelClassifier":
        X = self._fit_vectorizer(texts, structured)
        self.model.fit(X, list(labels))
        self.classes_ = list(self.model.classes_)
        return self

    def predict(
        self, text: str, structured: Optional[FeatureDict] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        X = self._transform([text], [structured] if structured is not None else None)
        proba = self.model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        dist = {cls: round(float(p), 4) for cls, p in zip(self.classes_, proba)}
        return self.classes_[idx], float(proba[idx]), dist

    def explain(
        self,
        text: str,
        structured: Optional[FeatureDict] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        pred, confidence, dist = self.predict(text, structured)
        label = label or pred
        i = self.classes_.index(label)
        coef = self.model.coef_[i] if self.model.coef_.shape[0] > 1 else self.model.coef_[0]
        intercept = (
            self.model.intercept_[i] if len(self.model.intercept_) > 1 else self.model.intercept_[0]
        )
        attribution = self._attribute(text, structured, coef, float(intercept))
        runner_up = sorted(dist.items(), key=lambda kv: -kv[1])[1:2]
        return {
            "task": self.name,
            "prediction": pred,
            "explained_label": label,
            "confidence": round(confidence, 4),
            "margin_over_runner_up": round(confidence - (runner_up[0][1] if runner_up else 0.0), 4),
            "class_distribution": dist,
            **attribution,
        }

    def global_top_terms(self, k: int = 12) -> Dict[str, List[Tuple[str, float]]]:
        out: Dict[str, List[Tuple[str, float]]] = {}
        for i, cls in enumerate(self.classes_):
            coef = self.model.coef_[i] if self.model.coef_.shape[0] > 1 else self.model.coef_[0]
            order = np.argsort(-coef)[:k]
            out[cls] = [(str(self.feature_names_[j]), round(float(coef[j]), 4)) for j in order]
        return out


class MultiLabelClassifier(_BaseTextModel):
    """One-vs-rest text classifier for the problem-flag task."""

    def __init__(
        self, name: str, labels: Sequence[str], config: Optional[ClassifierConfig] = None
    ) -> None:
        super().__init__(config, use_structured=False)
        self.name = name
        self.labels = list(labels)
        self.models: Dict[str, LogisticRegression] = {}
        self.trained_labels: List[str] = []

    def fit(self, texts: Sequence[str], label_sets: Sequence[Sequence[str]]) -> "MultiLabelClassifier":
        X = self._fit_vectorizer(texts)
        for label in self.labels:
            y = [1 if label in labels else 0 for labels in label_sets]
            if len(set(y)) < 2:
                LOGGER.warning("label %r has a single class in training data; skipping", label)
                continue
            clf = LogisticRegression(
                C=self.config.C,
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
            )
            clf.fit(X, y)
            self.models[label] = clf
            self.trained_labels.append(label)
        return self

    def predict(self, text: str) -> Tuple[List[str], Dict[str, float]]:
        X = self._transform([text])
        scores = {label: float(clf.predict_proba(X)[0, 1]) for label, clf in self.models.items()}
        positive = sorted(l for l, p in scores.items() if p >= self.config.multilabel_threshold)
        return positive, {k: round(v, 4) for k, v in scores.items()}

    def explain(self, text: str, labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        predicted, scores = self.predict(text)
        targets = list(labels) if labels is not None else predicted
        per_label = {}
        for label in targets:
            clf = self.models.get(label)
            if clf is None:
                continue
            per_label[label] = {
                "probability": scores.get(label),
                **self._attribute(text, None, clf.coef_[0], float(clf.intercept_[0])),
            }
        return {
            "task": self.name,
            "prediction": predicted,
            "threshold": self.config.multilabel_threshold,
            "label_probabilities": scores,
            "per_label": per_label,
        }

    def global_top_terms(self, k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        out: Dict[str, List[Tuple[str, float]]] = {}
        for label, clf in self.models.items():
            order = np.argsort(-clf.coef_[0])[:k]
            out[label] = [(str(self.feature_names_[j]), round(float(clf.coef_[0][j]), 4)) for j in order]
        return out


@dataclass
class DocumentClassifierBundle:
    """The three document-level models, trained and applied together."""

    note_type: SingleLabelClassifier
    problem_flags: MultiLabelClassifier
    risk: SingleLabelClassifier
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls, problem_labels: Sequence[str], config: Optional[ClassifierConfig] = None
    ) -> "DocumentClassifierBundle":
        return cls(
            note_type=SingleLabelClassifier("note_type", config),
            problem_flags=MultiLabelClassifier("problem_flags", problem_labels, config),
            risk=SingleLabelClassifier("readmission_risk", config, use_structured=True),
        )

    def fit(
        self,
        docs: Sequence[ClinicalDocument],
        structured: Optional[Sequence[FeatureDict]] = None,
        texts: Optional[Sequence[str]] = None,
    ) -> "DocumentClassifierBundle":
        # ``texts`` lets the caller feed de-identified text to the classifiers
        # while the NER stage keeps working on the original. See
        # ClinicalNLPPipeline._classifier_text for why that matters.
        texts = list(texts) if texts is not None else [d.text for d in docs]
        self.note_type.fit(texts, [d.gold_labels.note_type for d in docs])
        self.problem_flags.fit(texts, [d.gold_labels.problem_flags for d in docs])
        self.risk.fit(texts, [d.gold_labels.readmission_risk for d in docs], structured)
        self.metadata = {
            "n_train_docs": len(docs),
            "risk_uses_structured_features": self.risk.use_structured,
        }
        return self

    def predict(
        self,
        doc: ClinicalDocument,
        structured: Optional[FeatureDict] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        text = doc.text if text is None else text
        note_type, nt_conf, nt_dist = self.note_type.predict(text)
        flags, flag_scores = self.problem_flags.predict(text)
        risk, risk_conf, risk_dist = self.risk.predict(text, structured)
        return {
            "note_type": note_type,
            "note_type_confidence": round(nt_conf, 4),
            "note_type_distribution": nt_dist,
            "problem_flags": flags,
            "problem_flag_scores": flag_scores,
            "readmission_risk": risk,
            "readmission_risk_confidence": round(risk_conf, 4),
            "readmission_risk_distribution": risk_dist,
        }
