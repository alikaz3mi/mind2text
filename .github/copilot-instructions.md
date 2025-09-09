# Copilot Custom Instructions

*These guidelines apply to every suggestion Copilot makes in this repository.*

Repository layout:

```
.
├── data/
│   ├── ds004148/
│   └── storage/
├── docs/
│   └── datasets.md
├── LICENSE
├── mind2text/
│   ├── entities/                 # ← NEW: Pydantic domain models
│   │   ├── __init__.py
│   │   ├── common.py             # public (shared) entities
│   │   ├── dataset/              # feature-scoped entities
│   │   │   └── __init__.py
│   │   ├── features/             # feature vectors, tokens
│   │   │   └── __init__.py
│   │   ├── modeling/             # configs, runs, predictions
│   │   │   └── __init__.py
│   │   └── reports/
│   │       └── __init__.py
│   ├── preprocessing/
│   ├── algorithm/
│   ├── models/
│   └── postprocessing/
├── requirements.txt
└── setup.py
```

> Folder rule: **feature-scoped** entities go under `mind2text/entities/<feature-name>/`.
> **Public** entities (shared across layers/features) live directly under `mind2text/entities/` (e.g., `common.py`).

---

## 1) Architecture & dependency direction

**Do not violate this order:**

`entities` → `preprocessing` → `algorithm` → `models` → `postprocessing` → (optional) `frameworks`

* **entities**: canonical data contracts (Pydantic `BaseModel`), strict typing, validation, and serialization.
* **preprocessing**: IO + transforms return **entities** (e.g., `Trial`, `FeatureVector`, `TokenSequence`).
* **algorithm**: consumes `FeatureVector` and emits `TokenSequence` entities.
* **models**: consumes `TokenSequence`, emits `Prediction`, `Rationale`, `TrainingRun` entities.
* **postprocessing**: consumes `Prediction` (+ logits) and emits calibrated `PredictionCalibrated`, `Report` entities.

---

## 2) Coding style

* PEP8, full type hints, absolute imports, **one NumPy-style docstring per function/method**, no inline comments.
* `pathlib.Path`, TZ-aware `datetime` if used.
* `logging` only (`LOGGER = logging.getLogger(__name__)`).

---

## 3) Entities (Pydantic rules)

**All entities MUST:**

* Inherit from `pydantic.BaseModel`.
* Use `Field(..., description="...", example=...)` for **every field**.
* Be immutable where logical: `model_config = ConfigDict(frozen=True)` (v2) or `class Config: frozen=True` (v1).
* Include a `version: str = Field(..., ...)` when schema stability matters.
* Provide `from_*`/`to_*` helpers only if mapping is non-trivial (keep pure).

### Suggested core entities (public)

```python
# mind2text/entities/common.py
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

Band = Literal["delta", "theta", "alpha", "beta", "gamma"]

class ChannelInfo(BaseModel):
    """EEG channel metadata."""
    model_config = ConfigDict(frozen=True)
    name: str = Field(..., description="Channel name (10-20)", example="C3")
    index: int = Field(..., description="Zero-based channel index", example=12)

class Subject(BaseModel):
    """Dataset subject descriptor."""
    model_config = ConfigDict(frozen=True)
    subject_id: str = Field(..., description="Unique subject code", example="S087")
    age: Optional[int] = Field(None, description="Subject age (years)", example=25)
    handedness: Optional[Literal["L","R","Ambi"]] = Field("R", description="Dominant hand", example="R")

class SplitTag(BaseModel):
    """Data split label."""
    model_config = ConfigDict(frozen=True)
    name: Literal["train","val","test"] = Field(..., description="Split type", example="train")
    fold: Optional[int] = Field(None, description="CV fold index if applicable", example=0)

class Trial(BaseModel):
    """Single motor imagery trial metadata."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Unique trial identifier", example="S087_T034")
    subject_id: str = Field(..., description="Subject identifier", example="S087")
    label: Literal["left_hand","right_hand","feet","tongue"] = Field(..., description="Ground-truth class", example="left_hand")
    tmin: float = Field(..., description="Trial start time (s)", example=0.0)
    tmax: float = Field(..., description="Trial end time (s)", example=4.0)
    sfreq: float = Field(..., description="Sampling rate (Hz)", example=1000.0)
    channels: List[ChannelInfo] = Field(..., description="Ordered channel list", example=[{"name":"C3","index":12},{"name":"C4","index":28}])
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class FeatureVector(BaseModel):
    """Band-power features per channel."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Origin trial id", example="S087_T034")
    bands: List[Band] = Field(..., description="Band order", example=["alpha","beta"])
    values: List[List[float]] = Field(..., description="Shape (n_channels x n_bands)", example=[[2.1, 0.9],[1.7,1.2]])
    channel_names: List[str] = Field(..., description="Channel order aligned to values rows", example=["C3","C4"])
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class TokenSequence(BaseModel):
    """Symbolic tokenization of a trial."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Origin trial id", example="S087_T034")
    tokens: List[str] = Field(..., description="Ordered token list", example=["ALPHA_HIGH_C3","BETA_LOW_C4"])
    vocab_version: str = Field(..., description="Vocab version used for IDs", example="vocab_2025-09-10")
    version: str = Field("1.0", description="Entity schema version", example="1.0")
```

### Modeling entities

```python
# mind2text/entities/modeling/__init__.py
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class ModelConfig(BaseModel):
    """Training configuration for LLM or baseline."""
    model_config = ConfigDict(frozen=True)
    base_model: Literal["distilgpt2","llama2-7b","cnn","svm"] = Field(..., description="Backbone or baseline type", example="distilgpt2")
    lora_r: Optional[int] = Field(8, description="LoRA rank (LLM only)", example=8)
    lora_alpha: Optional[int] = Field(16, description="LoRA alpha (LLM only)", example=16)
    lr: float = Field(2e-4, description="Learning rate", example=0.0002)
    batch_size: int = Field(16, description="Global batch size", example=16)
    epochs: int = Field(5, description="Number of epochs", example=5)
    seed: int = Field(42, description="Random seed for reproducibility", example=42)
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class TrainingRun(BaseModel):
    """Immutable record of a training run."""
    model_config = ConfigDict(frozen=True)
    run_id: str = Field(..., description="Unique run identifier", example="run_2025-09-10_12-00")
    commit: str = Field(..., description="Git commit hash", example="a1b2c3d")
    dataset_hash: str = Field(..., description="Hash of split/indices", example="ds004148_split_v1")
    metrics: Dict[str, float] = Field(..., description="Key metrics (val accuracy, F1)", example={"val_acc":0.78,"val_f1":0.76})
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class Prediction(BaseModel):
    """Raw model prediction (pre-calibration)."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Trial id", example="S087_T034")
    logits: List[float] = Field(..., description="Raw logits for 4 classes", example=[1.2,-0.3,0.1,0.0])
    probs: List[float] = Field(..., description="Softmax probabilities", example=[0.54,0.12,0.19,0.15])
    pred_class: Literal["left_hand","right_hand","feet","tongue"] = Field(..., description="Argmax class", example="left_hand")
    rationale_text: Optional[str] = Field(None, description="Generated short rationale", example="Predicted left hand — beta suppression near C3.")
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class CalibrationParams(BaseModel):
    """Parameters learned for probability calibration."""
    model_config = ConfigDict(frozen=True)
    method: Literal["temperature","isotonic"] = Field(..., description="Calibration method", example="temperature")
    temperature: Optional[float] = Field(1.0, description="Temperature scaling factor", example=1.2)
    per_class_params: Optional[Dict[str, float]] = Field(None, description="Optional per-class params", example={"left_hand":1.1})
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class PredictionCalibrated(BaseModel):
    """Post-calibrated prediction."""
    model_config = ConfigDict(frozen=True)
    trial_id: str = Field(..., description="Trial id", example="S087_T034")
    probs_calibrated: List[float] = Field(..., description="Calibrated probabilities", example=[0.51,0.14,0.20,0.15])
    pred_class: Literal["left_hand","right_hand","feet","tongue"] = Field(..., description="Class after calibration/thresholds", example="left_hand")
    abstained: bool = Field(False, description="True if low-confidence abstention triggered", example=False)
    version: str = Field("1.0", description="Entity schema version", example="1.0")
```

### Reporting entities

```python
# mind2text/entities/reports/__init__.py
from typing import Dict, List
from pydantic import BaseModel, Field, ConfigDict

class ConfusionMatrix(BaseModel):
    """Dense confusion matrix with label order."""
    model_config = ConfigDict(frozen=True)
    labels: List[str] = Field(..., description="Class order", example=["left_hand","right_hand","feet","tongue"])
    matrix: List[List[int]] = Field(..., description="Counts (len=labels x labels)", example=[[50,3,2,1],[4,48,5,2],[3,4,45,6],[2,1,5,49]])
    version: str = Field("1.0", description="Entity schema version", example="1.0")

class Report(BaseModel):
    """Evaluation report with core metrics."""
    model_config = ConfigDict(frozen=True)
    run_id: str = Field(..., description="Training/eval run id", example="run_2025-09-10_12-00")
    accuracy: float = Field(..., description="Overall accuracy", example=0.78)
    macro_f1: float = Field(..., description="Macro F1-score", example=0.76)
    ece: float = Field(..., description="Expected calibration error", example=0.04)
    nll: float = Field(..., description="Negative log-likelihood", example=0.72)
    confusion: ConfusionMatrix = Field(..., description="Confusion matrix entity")
    version: str = Field("1.0", description="Entity schema version", example="1.0")
```

> You can add more feature-specific entities in `entities/dataset/` or `entities/features/` as needed (e.g., `BinningRule`, `VocabInfo`, etc.), following the same rules.

---

## 4) Configuration

* No DI framework. Prefer explicit parameters.
* Optional: `mind2text/settings.py` with `pydantic_settings.BaseSettings` for environment overrides (paths, model names).

---

## 5) Tests

* `unittest`, ≥90% coverage, mirror layout.
* **Entities tests**: validation, (de)serialization, immutability, example-driven roundtrips.
* **Pipelines**: CPU-only smoke tests for preprocessing/algorithm; deterministic seeds for models.

---

## 6) Preprocessing rules (returns entities)

* `eeg_loader.py` → returns `Subject` list + raw handles; metadata as `Trial`.
* `trial_segmenter.py` → returns `Trial` per segment.
* `signal_processor.py` → applies filters; returns arrays + validated metadata.
* `feature_extractor.py` → returns `FeatureVector`.

Persist intermediates under `data/storage/` when `save=True`.

---

## 7) Symbolization & tokenization (entities in/out)

* Discretize band powers with saved bin edges (serialized as an entity, e.g., `BinningRule`).
* `symbolic_encoder.py` → `TokenSequence` (keeps provenance).
* `tokenizer.py` → manages `vocab.json`; expose entity `VocabInfo`.

---

## 8) Modeling & training (entities in/out)

* LLM (LoRA/PEFT) + baselines share splits; consume `TokenSequence`, emit `Prediction`.
* Save `TrainingRun` snapshot (`run.json`) and checkpoints under `data/storage/checkpoints/<run_id>/`.

---

## 9) Post-processing (entities in/out)

* `calibrator.py` → fit `CalibrationParams`, apply to `Prediction` → `PredictionCalibrated`.
* `aggregator.py` → subject-level aggregation entities (e.g., `SubjectSummary`).
* `thresholds.py` → abstention logic (config as an entity).
* `rationale_refiner.py` → improves `rationale_text` using token provenance; always traceable.
* `exporters.py` → emit `Report` + JSONL rows of predictions (typed via entities).

---

## 10) Error handling

* Raise specific exceptions with actionable context; re-raise with `from e` when adding context.

---

## 11) Function template

```python
def example(name: str) -> str:
    """Return a polite greeting.

    Parameters
    ----------
    name : str
        Person’s display name.

    Returns
    -------
    str
        A greeting string.
    """
    return f"Hello, {name}!"
```

---

## 12) E2E demos (optional)

* **Streamlit** pages show: inputs → tokens → raw `Prediction` → `PredictionCalibrated` → `Report`.
* **FastAPI** only if needed; do not import frameworks inside core layers.

---

## 13) Documentation

* Docstrings on all public APIs with minimal examples.
* `docs/datasets.md`: dataset root, channels, sampling, splits.
* `docs/pipeline.md`: end-to-end including **post-processing**.
* `docs/experiments/<run_id>.md`: metrics, calibration plots, qualitative examples.

---

## 14) Things to avoid

* No `print` (use `LOGGER`).
* No magic constants—promote to `constants.py` or Enums (or entity fields).
* Functions >30 lines → refactor.
* No framework imports in `entities`, `preprocessing`, `algorithm`, `models`, `postprocessing`.
* Don’t commit big binaries in `data/`.

---

**Remember:** Copilot must propose code that **creates/uses entities end-to-end** (with `Field` descriptions & examples), respects deps (`entities` at the base), returns/accepts **entities in every layer**, and never sneaks in DI.
