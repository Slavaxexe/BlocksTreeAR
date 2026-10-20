"""Stable JSON envelopes; task names and algorithms are deliberately open-ended."""

from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
NodeID = Annotated[str, Field(pattern=r"^node_[0-9]{3}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


class Sourced(Contract):
    evidence: dict[str, list[str]] = Field(description="Flat lists of exact Markdown quotes for every populated field, including mapping fields")

    @model_validator(mode='before')
    @classmethod
    def normalize_evidence(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get('evidence'), dict):
            return value
        evidence = {}
        for field, quotes in value['evidence'].items():
            if field in cls.model_fields and value.get(field) in (None, '', [], {}):
                continue
            if isinstance(quotes, dict):
                quotes = [quote for group in quotes.values() for quote in (group if isinstance(group, list) else [group])]
            quotes = [quotes] if isinstance(quotes, str) else quotes
            evidence[field] = list(dict.fromkeys(quotes)) if isinstance(quotes, list) and all(isinstance(q, str) for q in quotes) else quotes
        return {**value, 'evidence': evidence}

    @model_validator(mode="after")
    def check_evidence(self, info: ValidationInfo):
        values = self.model_dump(exclude={"evidence"})
        populated = {k for k, v in values.items() if v is not None and v != [] and v != {} and v != ""}
        if set(self.evidence) != populated:
            raise ValueError(f"Evidence keys: missing {sorted(populated - set(self.evidence))}; "
                             f"unexpected {sorted(set(self.evidence) - populated)}")
        source = (info.context or {}).get("source")
        normalize = lambda text: " ".join(text.split())
        for field, quotes in self.evidence.items():
            if not quotes or any(not q.strip() for q in quotes):
                raise ValueError(f"{field}: expected nonempty source quotes")
            if source is not None and any(normalize(q) not in normalize(source) for q in quotes):
                raise ValueError(f"{field}: quote does not occur in the original Markdown")
            if field.endswith('_file') or field in {'target_column', 'target_columns', 'id_column', 'columns', 'class_labels', 'warnings'}:
                items = values[field] if isinstance(values[field], list) else [values[field]]
                if any(not any(normalize(str(item)) in normalize(q) for q in quotes) for item in items):
                    raise ValueError(f'{field}: literal identifiers must occur in their evidence')
        return self


class ProblemFacts(Sourced):
    task_type: str | None = None
    target_column: str | None = None
    target_columns: list[str] = Field(default_factory=list)
    target_unit: str | None = None
    class_labels: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, description="Short verbatim caveats from the source, not recommendations")


class DataFacts(Sourced):
    train_file: str | None = None
    test_file: str | None = None
    sample_submission_file: str | None = None
    dictionary_file: str | None = None
    id_column: str | None = None
    files: dict[str, str] = Field(default_factory=dict, description="Other named inputs: role to path")
    formats: dict[str, str] = Field(default_factory=dict, description="Explicitly stated file formats")
    encoding: str | None = None
    delimiter: str | None = None
    decimal: str | None = None


class MetricFacts(Sourced):
    name: str | None = None
    direction: Literal["minimize", "maximize"] | None = None
    definition: str | None = Field(default=None, description="Definition, formula or supplied code reference")
    secondary_metrics: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OutputFacts(Sourced):
    submission_file: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    format: str | None = None
    columns: list[str] = Field(default_factory=list)
    prediction_type: str | None = None
    preserve_test_order: bool | None = None
    unique_ids: bool | None = None
    description: str | None = None


class Extraction(Contract):
    schema_version: Literal[2] = 2
    problem: ProblemFacts
    data: DataFacts
    metric: MetricFacts
    output: OutputFacts


class Metric(Contract):
    name: str | None = None
    direction: Literal["minimize", "maximize"] | None = None
    definition: str | None = None
    status: Literal["ready", "needs_implementation"] = "needs_implementation"


class DataBinding(Contract):
    files: dict[str, str] = Field(description="Input roles to paths relative to the task directory")
    target_columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)


class OutputBinding(Contract):
    filename: str = Field(min_length=1)
    format: str | None = None
    columns: list[str] = Field(default_factory=list)


class PreparationDraft(Contract):
    rationale: str = Field(min_length=1)
    data: DataBinding
    validation: dict[str, Any]
    metric: Metric
    output: OutputBinding
    code: str = Field(min_length=1, description="Complete prepare.py source implementing the requested Python interface")


class Preparation(Contract):
    schema_version: Literal[2] = 2
    status: Literal["pending", "ready", "partial", "blocked"] = "pending"
    extraction: str = "extraction.json"
    task_directory: str
    rationale: str
    data: DataBinding
    validation: dict[str, Any]
    metric: Metric
    output: OutputBinding
    profile: dict[str, Any] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    code_sha256: SHA256
    issues: list[str] = Field(default_factory=list)


class Pipeline(Contract):
    data: dict[str, Any]
    preprocessing: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]


class Hypothesis(Contract):
    rationale: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    pipeline: Pipeline


class Program(Contract):
    code: str = Field(default="", description="Complete experiment.py implementing the saved start, or empty when blocked")
    notes: list[str] = Field(default_factory=list, description="Implementation details and limitations; never measured scores")
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def code_or_reason(self):
        if not self.code.strip() and not (self.blocked_reason or '').strip():
            raise ValueError('Return implementation code or explain why the saved plan cannot be implemented')
        return self


class Start(Contract):
    schema_version: Literal[3] = 3
    node_id: NodeID = "node_000"
    parent_id: NodeID | None = None
    source: Literal["qwen"] = "qwen"
    preparation: str = "../../preparation.json"
    preparation_sha256: SHA256
    context: dict[str, Any] = Field(default_factory=dict, description="Source references, hashes and installed libraries; shared data is not duplicated")
    hypothesis: Hypothesis
    implementation: str = "experiment.py"
    outputs: dict[str, str]
    timeout_seconds: float = Field(gt=0)


class Finish(Contract):
    schema_version: Literal[3] = 3
    node_id: NodeID = "node_000"
    status: Literal["completed", "failed", "timeout"]
    stage: Literal["implementation", "check", "execution"] = "execution"
    metric: Metric
    value: float | None = None
    evaluation_status: Literal["completed", "unavailable", "failed"] = "unavailable"
    duration_seconds: float = Field(ge=0)
    preparation_sha256: SHA256
    start_sha256: SHA256
    code_sha256: SHA256 | None = None
    notes: list[str] = Field(default_factory=list)
    validation_file: str | None = None
    submission_file: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def consistent_result(self):
        if self.status == 'completed' and (not self.submission_file or not self.validation_file):
            raise ValueError('Completed execution requires prediction files')
        if self.status == 'completed' and (self.stage != 'execution' or self.code_sha256 is None):
            raise ValueError('Completed execution requires the executed source hash and execution stage')
        if (self.evaluation_status == 'completed') != (self.value is not None):
            raise ValueError('Only completed evaluation can have a score')
        if self.status != 'completed' and (self.value is not None or not self.error):
            raise ValueError('Failed execution requires an error and cannot have a score')
        return self


class LLMCall(Contract):
    role: str
    attempt: int
    status: Literal["completed", "invalid_json", "api_error"]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class RunReport(Contract):
    schema_version: Literal[2] = 2
    status: Literal["running", "ready", "partial", "failed"]
    stage: Literal["extraction", "preparation", "baseline", "complete"] = "extraction"
    mode: Literal["qwen"] = "qwen"
    task_sha256: SHA256
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)


SCHEMAS = (Extraction, Preparation, Start, Finish, RunReport)
