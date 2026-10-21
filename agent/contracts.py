"""Stable JSON envelopes; task names and algorithms are deliberately open-ended."""

from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
NodeID = Annotated[str, Field(pattern=r"^node_[0-9]{3,}$")]
INITIAL_STATE = frozenset({'fit', 'validation', 'test', 'preparation', 'auxiliary'})


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


class Block(Contract):
    name: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    source_node: NodeID | None = None
    description: str = Field(min_length=1, description='Behavior and exact input/output types, shapes, column conventions and object APIs')
    requires: list[str] = Field(default_factory=list)
    provides: list[str] = Field(min_length=1, description='Required non-None state outputs; nullable settings belong inside output objects')


class Assembly(Contract):
    blocks: list[Block] = Field(min_length=1)
    write_blocks: list[str] = Field(min_length=1)

    @model_validator(mode='before')
    @classmethod
    def legacy_edit_block(cls, value):
        if isinstance(value, dict) and 'edit_block' in value and 'write_blocks' not in value:
            return {**{k: v for k, v in value.items() if k != 'edit_block'},
                    'write_blocks': [value['edit_block']]}
        return value

    @model_validator(mode='after')
    def connected(self):
        names = [block.name for block in self.blocks]
        if len(names) != len(set(names)) or len(self.write_blocks) != len(set(self.write_blocks)):
            raise ValueError('Block names and write_blocks must be unique')
        if not set(self.write_blocks) <= set(names):
            raise ValueError('Every write_blocks entry must name an assembly block')
        available = set(INITIAL_STATE)
        for block in self.blocks:
            if block.name not in self.write_blocks and block.source_node is None:
                raise ValueError('Every new block must be listed in write_blocks')
            missing = set(block.requires) - available
            if missing:
                raise ValueError(f'{block.name}: inputs unavailable at this position: {sorted(missing)}')
            if set(block.provides) & INITIAL_STATE:
                raise ValueError(f'{block.name}: shared prepared inputs cannot be replaced; return new state keys')
            available.update(block.provides)
        if not {'validation_predictions', 'test_predictions'} <= available:
            raise ValueError('Assembly must produce validation_predictions and test_predictions')
        return self


class Proposal(Contract):
    purpose: Literal['baseline', 'improve', 'explore']
    parent_id: NodeID | None = None
    hypothesis: Hypothesis
    assembly: Assembly


class Idea(Contract):
    purpose: Literal['baseline', 'improve', 'explore']
    parent_id: NodeID | None = None
    change: str = Field(min_length=1, description='One concrete change, including intended algorithm or parameters')
    change_blocks: list[str] = Field(default_factory=list, description='Existing or proposed block names to change')
    expected_effect: str = Field(min_length=1)


class Ideas(Contract):
    ideas: list[Idea] = Field(min_length=1)

    @model_validator(mode='after')
    def requested_ideas(self, info: ValidationInfo):
        context = info.context or {}
        if 'purposes' not in context:
            return self
        if [idea.purpose for idea in self.ideas] != context['purposes']:
            raise ValueError('Return exactly the requested purposes in their given order')
        for idea in self.ideas:
            if idea.purpose == 'baseline':
                if idea.parent_id is not None:
                    raise ValueError('Baseline must have no parent')
            elif idea.parent_id not in context['parents']:
                raise ValueError('Choose an available parent')
            elif idea.purpose == 'improve' and idea.parent_id != context['leader']:
                raise ValueError('Improve must extend the current leader')
        return self


class Proposals(Contract):
    proposals: list[Proposal] = Field(min_length=1)

    @model_validator(mode='after')
    def valid_plans(self, info: ValidationInfo):
        context = info.context or {}
        count = context.get('count')
        if count is not None and len(self.proposals) != count:
            raise ValueError(f'Return exactly {count} proposals')
        for index, proposal in enumerate(self.proposals):
            expected = 'baseline' if context.get('baseline') else context.get('purposes', [p.purpose for p in self.proposals])[index]
            if count is not None and proposal.purpose != expected:
                raise ValueError('Return proposals in the requested purposes order')
            if context.get('baseline') and proposal.parent_id is not None:
                raise ValueError('Baseline must have no parent')
            if 'parents' in context and proposal.parent_id not in context['parents']:
                raise ValueError('Choose an available parent')
            if proposal.purpose == 'improve' and proposal.parent_id != context.get('leader', proposal.parent_id):
                raise ValueError('Improve must extend the selected leader')
            catalog = context.get('catalog')
            for block in proposal.assembly.blocks:
                if catalog is not None and block.source_node is not None:
                    source = next((b for b in catalog.get(block.source_node, []) if b['name'] == block.name), None)
                    if source is None:
                        raise ValueError(f'Unknown saved block: {block.source_node}/{block.name}')
                    if block.name not in proposal.assembly.write_blocks and block.model_dump(exclude={'source_node'}) != {
                            k: source[k] for k in ('name', 'description', 'requires', 'provides')}:
                        raise ValueError('Keep the declared interface of reused blocks unchanged')
        return self


class BlockReference(Contract):
    name: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    source_node: NodeID


class NodePlan(Contract):
    hypothesis: Hypothesis
    order: list[str] = Field(min_length=1, description='Block names in execution order')
    reuse: list[BlockReference] = Field(default_factory=list, description='References only; Python copies the saved interfaces')
    write: list[Block] = Field(min_length=1, description='Complete interfaces of only the new or changed blocks')

    def proposal(self, idea: Idea, catalog: dict) -> Proposal:
        blocks = {block.name: block for block in self.write}
        for reference in self.reuse:
            source = next((block for block in catalog.get(reference.source_node, [])
                           if block['name'] == reference.name), None)
            if source is None:
                raise ValueError(f'Unknown saved block: {reference.source_node}/{reference.name}')
            blocks[reference.name] = Block(source_node=reference.source_node,
                **{key: source[key] for key in ('name', 'description', 'requires', 'provides')})
        return Proposal(purpose=idea.purpose, parent_id=idea.parent_id, hypothesis=self.hypothesis,
                        assembly=Assembly(blocks=[blocks[name] for name in self.order],
                                          write_blocks=[block.name for block in self.write]))

    @model_validator(mode='after')
    def assembled_plan(self, info: ValidationInfo):
        names = [block.name for block in self.write] + [block.name for block in self.reuse]
        if len(names) != len(set(names)) or len(self.order) != len(set(self.order)) or set(names) != set(self.order):
            raise ValueError('order must contain every written or reused block exactly once')
        context = info.context or {}
        if 'idea' in context:
            proposal = self.proposal(context['idea'], context['catalog'])
            Proposals.model_validate({'proposals': [proposal]}, context={**context, 'count': 1,
                'baseline': proposal.purpose == 'baseline', 'purposes': [proposal.purpose]})
        return self


class Analysis(Contract):
    summary: str = Field(min_length=1)
    findings: list[str] = Field(default_factory=list)
    next_ideas: list[str] = Field(default_factory=list)


class Program(Contract):
    code: str = Field(default='', description='Full source of ONLY the requested block, defining run(state)')
    notes: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    @model_validator(mode='after')
    def code_or_reason(self):
        if not self.code.strip() and not (self.blocked_reason or '').strip():
            raise ValueError('Return block code or blocked_reason')
        return self


class Start(Contract):
    schema_version: Literal[4, 5] = 5
    node_id: NodeID = "node_000"
    parent_id: NodeID | None = None
    source: Literal["qwen"] = "qwen"
    preparation: str = "../../preparation.json"
    preparation_sha256: SHA256
    context: dict[str, Any] = Field(default_factory=dict)
    purpose: Literal['baseline', 'improve', 'explore']
    hypothesis: Hypothesis
    assembly: Assembly
    implementation: str = 'experiment.py'
    outputs: dict[str, str]
    timeout_seconds: float | None = Field(default=None, gt=0)


class Finish(Contract):
    schema_version: Literal[4] = 4
    node_id: NodeID = "node_000"
    status: Literal["completed", "failed", "timeout"]
    stage: Literal['implementation', 'assembly', 'check', 'execution'] = 'execution'
    failed_block: str | None = None
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
        if self.status == 'completed' and (self.code_sha256 is None or self.stage != 'execution'):
            raise ValueError('Completed execution requires an executed assembly hash')
        if (self.evaluation_status == 'completed') != (self.value is not None):
            raise ValueError('Only completed evaluation can have a score')
        if self.status != 'completed' and (self.value is not None or not self.error):
            raise ValueError('Failed execution requires an error and cannot have a score')
        return self


class Change(Contract):
    path: str
    before: Any = None
    after: Any = None


class Comparison(Contract):
    reference_id: NodeID
    comparable: bool
    reason: str | None = None
    improvement: float | None = None
    changes: list[Change] = Field(default_factory=list)


class NodeDelta(Contract):
    node_id: NodeID
    parent_id: NodeID | None
    hypothesis: Hypothesis
    assembly: Assembly
    result: Finish
    analysis: Analysis | None = None
    analysis_status: Literal['pending', 'running', 'completed', 'failed', 'skipped'] = 'pending'
    comparisons: list[Comparison] = Field(default_factory=list)


class Delta(Contract):
    preparation_sha256: SHA256
    metric: Metric
    nodes: list[NodeDelta] = Field(default_factory=list)
    best_node_id: NodeID | None = None


class LLMCall(Contract):
    role: str
    node_id: NodeID | None = None
    block: str | None = None
    attempt: int
    status: Literal["completed", "invalid_json", "api_error"]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class RunReport(Contract):
    schema_version: Literal[2] = 2
    status: Literal["running", "ready", "partial", "failed"]
    stage: Literal["extraction", "preparation", "baseline", "search", "complete"] = "extraction"
    mode: Literal["qwen"] = "qwen"
    task_sha256: SHA256
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    experiments: int = 0
    best_node: NodeID | None = None
    stop_reason: str | None = None
    running_nodes: list[NodeID] = Field(default_factory=list)
    queued_nodes: list[NodeID] = Field(default_factory=list)
    planning_nodes: list[NodeID] = Field(default_factory=list)
    queued_ideas: list[dict[str, Any]] = Field(default_factory=list)
    analyzing_nodes: list[NodeID] = Field(default_factory=list)
    planning: bool = False
    max_parallel_nodes: int = 3
    max_parallel_agents: int = 8
    active_agents: list[dict[str, Any]] = Field(default_factory=list)
    node_states: dict[str, dict[str, Any]] = Field(default_factory=dict)


SCHEMAS = (Extraction, Preparation, Start, Finish, RunReport, Proposals, Ideas, NodePlan, Analysis, Assembly, NodeDelta, Delta)
