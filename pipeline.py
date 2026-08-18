"""End-to-end orchestration: raw note in, structured record + explanation out.

    ingest -> de-id -> normalise -> sectionise -> tokenise
           -> hybrid NER -> assertion -> concept normalisation
           -> document classification
           -> structuring -> explainability -> outputs

The class below is the only object a caller needs.  ``fit`` trains every model
from an annotated corpus, ``process`` runs one document all the way through,
and ``save``/``load`` persist the whole thing as a single artefact so that
inference does not depend on retraining.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib

from clinical_nlp import __version__
from clinical_nlp.classify.model import ClassifierConfig, DocumentClassifierBundle
from clinical_nlp.classify.structured_features import describe, structured_features
from clinical_nlp.config import PipelineConfig
from clinical_nlp.data.schema import PROBLEM_FLAGS, ClinicalDocument, StructuredRecord
from clinical_nlp.explain.attributions import document_explanation, global_model_explanation
from clinical_nlp.ingest.deid import scrub
from clinical_nlp.ner.pipeline import HybridConfig, HybridNER
from clinical_nlp.ner.tagger import TaggerConfig
from clinical_nlp.preprocess.normalize import normalize_text
from clinical_nlp.preprocess.sectionizer import assign_sections, detect_sections
from clinical_nlp.preprocess.tokenize import tokenize
from clinical_nlp.structure.assembler import assemble
from clinical_nlp.utils.logging import get_logger, timed

LOGGER = get_logger(__name__)


@dataclass
class ProcessedNote:
    document: ClinicalDocument
    record: StructuredRecord
    explanation: Dict[str, Any]
    timings_ms: Dict[str, float]


class ClinicalNLPPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.ner = HybridNER(
            HybridConfig(
                use_statistical=self.config.ner.use_statistical,
                use_gazetteer=self.config.ner.use_gazetteer,
                use_rules=self.config.ner.use_rules,
                agreement_bonus=self.config.ner.agreement_bonus,
                fuzzy_cutoff=self.config.ner.fuzzy_cutoff,
                tagger=TaggerConfig(
                    C=self.config.ner.C,
                    max_iter=self.config.ner.max_iter,
                    solver=self.config.ner.solver,
                    min_span_score=self.config.ner.min_span_score,
                    random_state=self.config.data.seed,
                ),
            )
        )
        self.classifiers = DocumentClassifierBundle.build(
            PROBLEM_FLAGS,
            ClassifierConfig(
                ngram_range=(1, self.config.classify.ngram_max),
                min_df=self.config.classify.min_df,
                max_features=self.config.classify.max_features,
                C=self.config.classify.C,
                class_weight=self.config.classify.class_weight,
                multilabel_threshold=self.config.classify.multilabel_threshold,
                random_state=self.config.data.seed,
            ),
        )
        self.trained_at: Optional[str] = None

    # -- training ---------------------------------------------------------

    def fit(self, docs: Sequence[ClinicalDocument]) -> "ClinicalNLPPipeline":
        with timed(LOGGER, f"training hybrid NER on {len(docs)} notes"):
            self.ner.fit(docs)

        # The risk model is stacked on the extraction stage, so its training
        # features come from *predicted* entities on the training notes -- not
        # from gold spans. Training on gold and serving on predictions is the
        # classic stacked-model leak; this avoids it.
        with timed(LOGGER, "extracting structured features from training notes"):
            structured = []
            for doc in docs:
                self.preprocess(doc)
                self.ner.predict(doc)
                structured.append(structured_features(doc))

        with timed(LOGGER, "training document classifiers"):
            self.classifiers.fit(
                docs, structured, texts=[self._classifier_text(d) for d in docs]
            )
        self.trained_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        return self

    # -- inference --------------------------------------------------------

    @staticmethod
    def _classifier_text(doc: ClinicalDocument) -> str:
        """The text view the document classifiers see: de-identified.

        This is not decoration. The first version of this pipeline fed raw note
        text to the classifiers, and the global explanation
        (``outputs/global_model_explanation.json``) showed attending-physician
        names among the top-weighted terms for readmission risk -- the model was
        learning *which doctor wrote the note*, a textbook shortcut that would
        transfer to nothing and would encode a protected-ish attribute besides.
        Scrubbing identifiers before vectorising removes that pathway. The
        scrubber is length-preserving, so entity offsets stay valid and the NER
        stage continues to read the original text.
        """
        return scrub(doc.text)[0]

    def preprocess(self, doc: ClinicalDocument, apply_deid: bool = False) -> ClinicalDocument:
        text = normalize_text(doc.text)
        if apply_deid:
            text, report = scrub(text)
            doc.metadata["deid"] = {"n_redactions": report.n_redactions, "by_type": report.by_type}
        doc.text = text
        doc.sections = detect_sections(text)
        doc.tokens = tokenize(text)
        assign_sections(doc.tokens, doc.sections)
        return doc

    def process(self, doc: ClinicalDocument, apply_deid: bool = False) -> ProcessedNote:
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        self.preprocess(doc, apply_deid=apply_deid)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self.ner.predict(doc)
        timings["ner"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        struct = structured_features(doc)
        doc.metadata["structured_features"] = describe(struct)
        classifier_text = self._classifier_text(doc)
        predictions = self.classifiers.predict(doc, struct, text=classifier_text)
        doc.pred_labels.note_type = predictions["note_type"]
        doc.pred_labels.problem_flags = predictions["problem_flags"]
        doc.pred_labels.readmission_risk = predictions["readmission_risk"]
        timings["classify"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        explanation = document_explanation(
            doc,
            self.classifiers,
            structured=struct,
            classifier_text=classifier_text,
            max_entities=self.config.output.max_entities_explained,
            occlusion_ner=self.ner if self.ner._fitted else None,
        )
        explanation["rule_based_problem_flags"] = self.ner.problem_flags_from_spans(doc.pred_spans)
        doc.explanations = explanation
        timings["explain"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        record = assemble(
            doc,
            predictions,
            include_explanations=self.config.output.include_explanations_in_records,
        )
        timings["structure"] = (time.perf_counter() - t0) * 1000
        timings["total"] = sum(timings.values())

        return ProcessedNote(doc, record, explanation, {k: round(v, 2) for k, v in timings.items()})

    def process_many(
        self, docs: Sequence[ClinicalDocument], apply_deid: bool = False
    ) -> List[ProcessedNote]:
        return [self.process(doc, apply_deid=apply_deid) for doc in docs]

    # -- introspection ----------------------------------------------------

    def global_explanation(self) -> Dict[str, Any]:
        return global_model_explanation(self.ner, self.classifiers)

    def model_summary(self) -> Dict[str, Any]:
        return {
            "package_version": __version__,
            "trained_at": self.trained_at,
            "ner": {
                "tagger_classes": self.ner.tagger.classes_,
                "n_tagger_features": len(self.ner.tagger._feature_names),
                "gazetteer_entries": len(self.ner.gazetteer.entries),
                "components": {
                    "statistical": self.config.ner.use_statistical,
                    "gazetteer": self.config.ner.use_gazetteer,
                    "rules": self.config.ner.use_rules,
                },
            },
            "classifiers": {
                "note_type_classes": self.classifiers.note_type.classes_,
                "risk_classes": self.classifiers.risk.classes_,
                "problem_flags": self.classifiers.problem_flags.trained_labels,
                "tfidf_features": int(self.classifiers.note_type.feature_names_.shape[0]),
            },
            "config": self.config.to_dict(),
        }

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"ner": self.ner, "classifiers": self.classifiers, "config": self.config,
             "trained_at": self.trained_at, "version": __version__},
            path,
            compress=3,
        )
        LOGGER.info("saved pipeline -> %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @classmethod
    def load(cls, path: Path) -> "ClinicalNLPPipeline":
        payload = joblib.load(Path(path))
        obj = cls(payload["config"])
        obj.ner = payload["ner"]
        obj.classifiers = payload["classifiers"]
        obj.trained_at = payload.get("trained_at")
        return obj
