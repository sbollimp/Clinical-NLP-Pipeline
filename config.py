"""Configuration.

One YAML file drives the whole run so that an experiment is reproducible from
a single artefact: same config + same seed = same numbers.  Every field has a
default here, so the YAML only needs to state what it changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # PyYAML is optional -- JSON configs work without it
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class DataConfig:
    n_notes: int = 600
    seed: int = 13
    corpus_dir: str = "data/corpus"
    train_ratio: float = 0.7
    dev_ratio: float = 0.15
    noise_rate: float = 0.15


@dataclass
class NERConfig:
    use_statistical: bool = True
    use_gazetteer: bool = True
    use_rules: bool = True
    C: float = 4.0
    max_iter: int = 400
    solver: str = "lbfgs"
    min_span_score: float = 0.30
    fuzzy_cutoff: float = 0.88
    agreement_bonus: float = 0.05


@dataclass
class ClassifyConfig:
    ngram_max: int = 2
    min_df: int = 2
    max_features: int = 60000
    C: float = 4.0
    class_weight: Optional[str] = "balanced"
    multilabel_threshold: float = 0.5


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    model_dir: str = "models"
    n_reports: int = 6
    write_csv: bool = True
    write_jsonl: bool = True
    include_explanations_in_records: bool = True
    max_entities_explained: Optional[int] = None


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    log_level: str = "INFO"
    run_ablation: bool = True

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PipelineConfig":
        if path is None:
            return cls()
        raw = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to read YAML configs; pip install pyyaml")
            payload: Dict[str, Any] = yaml.safe_load(raw) or {}
        else:
            import json

            payload = json.loads(raw)
        return cls(
            data=DataConfig(**payload.get("data", {})),
            ner=NERConfig(**payload.get("ner", {})),
            classify=ClassifyConfig(**payload.get("classify", {})),
            output=OutputConfig(**payload.get("output", {})),
            log_level=payload.get("log_level", "INFO"),
            run_ablation=payload.get("run_ablation", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> List[str]:
        return [f"{k}: {v}" for k, v in self.to_dict().items()]
