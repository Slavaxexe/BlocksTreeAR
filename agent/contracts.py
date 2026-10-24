"""Stable JSON envelopes; task names and algorithms are deliberately open-ended."""

from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .plans import AutoMLPlan, AnalysisRequest, ToolCall, EnsemblePlan, SplitPlan
from .validation import metadata

SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
NodeID = Annotated[str, Field(pattern=r"^node_[0-9]{3}$")]
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

    normalize_validation = field_validator('validation')(metadata)

    @model_validator(mode='after')
    def deployment_matches_validation(self, info: ValidationInfo):
        from .validation import split_plan, strategy
        split_plan(self.validation, self.data.target_columns)
        context = info.context or {}
        scenario = self.validation.get('deployment_scenario')
        if context.get('require_deployment_assessment'):
            if scenario not in {'future_period', 'unseen_groups', 'exchangeable', 'uncertain'}:
                raise ValueError('Assess deployment_scenario from task semantics and ordered_query_ranges before fixing validation')
            if not str(self.validation.get('deployment_reason', '')).strip():
                raise ValueError('deployment_reason must explain the measured separation and the chosen prediction scenario')
        if scenario == 'future_period' and (not self.validation.get('time_column') or strategy(self.validation) == 'random'):
            raise ValueError('future_period needs an ordered time_column and a non-random outer split')
        if scenario == 'unseen_groups' and not self.validation.get('group_columns'):
            raise ValueError('unseen_groups needs explicit group_columns')
        return self


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

    normalize_validation = field_validator('validation')(metadata)


class Pipeline(Contract):
    data: dict[str, Any]
    preprocessing: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]


class ModelConfiguration(Contract):
    name: str = Field(min_length=1, pattern=r'\S', description='Actual algorithm/API name or ensemble name, not a block name')
    configuration: str = Field(min_length=1, pattern=r'\S', description='Concrete settings of every model; state defaults explicitly')


class PipelinePlan(Contract):
    preprocessing: str = Field(min_length=1, pattern=r'\S', description='Actual transformations, or explicitly no preprocessing')
    model: ModelConfiguration
    training: str = Field(min_length=1, pattern=r'\S', description='Fitting, tuning, internal validation and ensemble procedure')

    def configuration(self) -> Pipeline:
        return Pipeline(data={}, preprocessing={'description': self.preprocessing},
                        model=self.model.model_dump(), training={'description': self.training})


class Hypothesis(Contract):
    rationale: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    pipeline: Pipeline
    research_ids: list[str] = Field(default_factory=list)


class Block(Contract):
    operation: ToolCall | None = None
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
    automl: AutoMLPlan | None = Field(default=None, description='Executable AutoML configuration; null for composed trusted operations')
    ensemble: EnsemblePlan | None = Field(default=None, description='Combine completed prediction sources; compiled without a node-planner call')
    feature_candidate_ids: list[str] = Field(default_factory=list, description='Exact measured feature candidates selected for this experiment; planner must include their recipes')
    priority: int = Field(default=50, ge=0, le=100, description='Queue priority; higher runs first, not an expected score')
    research_ids: list[str] = Field(default_factory=list, description='IDs of only the research insights needed for this change; empty when unrelated')


class IdeaUpdate(Contract):
    idea_id: str
    action: Literal['prioritize', 'remove']
    priority: int = Field(default=50, ge=0, le=100)
    reason: str = Field(min_length=1)


class FeatureDecision(Contract):
    candidate_id: str
    action: Literal['test', 'defer', 'reject']
    reason: str = Field(min_length=1)


class Ideas(Contract):
    ideas: list[Idea] = Field(default_factory=list)
    updates: list[IdeaUpdate] = Field(default_factory=list)
    feature_decisions: list[FeatureDecision] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def select_requested(cls, value, info: ValidationInfo):
        context = info.context or {}
        if 'max_new_ideas' not in context or not isinstance(value, dict) or not isinstance(value.get('ideas'), list):
            return value
        limit = context['max_new_ideas']
        if len(value['ideas']) <= limit:
            return value
        available = []
        for row in value['ideas']:
            try:
                idea = Idea.model_validate(row)
            except ValueError:
                continue
            if (idea.purpose in context['purposes']
                    and (idea.purpose != 'baseline' or idea.parent_id is None)
                    and (idea.purpose != 'improve' or idea.parent_id == context['leader'])
                    and (idea.purpose == 'baseline' or idea.parent_id in context['parents'])):
                available.append(idea)
        if len(available) < limit:
            return value  # Actual invalid ideas still require a corrected model response.
        available.sort(key=lambda idea: -idea.priority)
        return {**value, 'ideas': available[:limit]}

    @model_validator(mode='after')
    def requested_ideas(self, info: ValidationInfo):
        context = info.context or {}
        if 'purposes' not in context:
            return self
        if len(self.ideas) > context.get('max_new_ideas', len(context['purposes'])):
            raise ValueError('Too many new ideas for this request')
        candidates = context.get('feature_candidates', [])
        from .feature_trials import candidate_id, selected_candidates
        for idea in self.ideas:
            selected_candidates(candidates, idea.feature_candidate_ids)
            if idea.feature_candidate_ids and idea.ensemble is not None:
                raise ValueError('Feature candidates require a model experiment, not a saved-prediction ensemble')
            if idea.ensemble is not None:
                if idea.purpose == 'baseline' or idea.automl is not None:
                    raise ValueError('An ensemble is a separate non-baseline hypothesis')
                sources = {item['node_id']: item for item in context.get('prediction_sources', [])}
                if 'prediction_sources' in context:
                    if not set(idea.ensemble.source_nodes) <= sources.keys():
                        raise ValueError('Select completed sources from prediction_sources')
                    selected = [sources[node] for node in idea.ensemble.source_nodes]
                    if (idea.ensemble.method in {'stacking', 'gating'} or idea.ensemble.optimize_weights) and not all(item.get('meta_ready', item.get('oof_recipe', False)) or item.get('meta_available', False) for item in selected):
                        raise ValueError('Learned ensembles need meta_ready sources with saved held-out predictions; sources are not retrained')
                    if (idea.ensemble.method == 'probabilities' or idea.ensemble.combine == 'probabilities') and not all(
                            item['probabilities'] for item in selected):
                        raise ValueError('These sources have no saved probabilities; use vote/mean')
            if idea.automl is not None:
                if idea.purpose == 'baseline':
                    raise ValueError('The initial baseline must remain a cheap predictor')
                package = 'flaml' if idea.automl.engine == 'flaml' else 'autogluon.tabular'
                if 'installed_libraries' in context and package not in context['installed_libraries']:
                    raise ValueError(f'AutoML engine is not installed: {package}')
            if 'research_ids' in context and not set(idea.research_ids).issubset(context['research_ids']):
                raise ValueError('Reference only supplied research insight IDs')
            if idea.purpose not in context['purposes']:
                raise ValueError('Choose an allowed idea purpose')
            if idea.purpose == 'baseline':
                if idea.parent_id is not None:
                    raise ValueError('Baseline must have no parent')
            elif idea.parent_id not in context['parents']:
                raise ValueError('Choose an available parent')
            elif idea.purpose == 'improve' and idea.parent_id != context['leader']:
                raise ValueError('Improve must extend the current leader')
        if 'feature_candidates' in context:
            pending = {candidate_id(item) for item in candidates if item.get('trial_status') == 'untested'}
            addressed = {identity for idea in self.ideas for identity in idea.feature_candidate_ids}
            for decision in self.feature_decisions:
                if decision.action == 'test' and decision.candidate_id not in addressed:
                    raise ValueError('A test decision needs an idea selecting that feature_candidate_id')
            addressed.update(decision.candidate_id for decision in self.feature_decisions)
            if pending - addressed:
                raise ValueError(f'Decide test/defer/reject for untested feature candidates: {sorted(pending - addressed)}')
        ids = [update.idea_id for update in self.updates]
        if len(ids) != len(set(ids)) or not set(ids).issubset(context.get('idea_ids', [])):
            raise ValueError('Update each supplied idea ID at most once')
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
                            **{k: source[k] for k in ('name', 'description', 'requires', 'provides')},
                            'operation': source.get('operation')}:
                        raise ValueError('Keep the declared interface of reused blocks unchanged')
        return self


class BlockReference(Contract):
    name: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    source_node: NodeID


class NodePlan(Contract):
    pipeline: PipelinePlan
    order: list[str] = Field(min_length=1, description='Block names in execution order')
    reuse: list[BlockReference] = Field(default_factory=list, description='References only; Python copies the saved interfaces')
    write: list[Block] = Field(min_length=1, description='Complete interfaces of only the new or changed blocks')

    @model_validator(mode='before')
    @classmethod
    def legacy_hypothesis(cls, value):
        if isinstance(value, dict):
            written = []
            for block in value.get('write', []):
                if isinstance(block, dict) and block.get('operation'):
                    call = ToolCall.model_validate(block['operation'])
                    required, provided = call.interface()
                    block = {**block, 'requires': required, 'provides': provided}
                written.append(block)
            value = {**value, 'write': written}
        if isinstance(value, dict) and 'hypothesis' in value and 'pipeline' not in value:
            hypothesis = value['hypothesis']
            pipeline = hypothesis.pipeline if isinstance(hypothesis, Hypothesis) else hypothesis['pipeline']
            return {**{key: item for key, item in value.items() if key != 'hypothesis'}, 'pipeline': pipeline}
        return value

    def proposal(self, idea: Idea, catalog: dict) -> Proposal:
        blocks = {block.name: block for block in self.write}
        for reference in self.reuse:
            source = next((block for block in catalog.get(reference.source_node, [])
                           if block['name'] == reference.name), None)
            if source is None:
                raise ValueError(f'Unknown saved block: {reference.source_node}/{reference.name}')
            blocks[reference.name] = Block(source_node=reference.source_node,
                **{key: source[key] for key in ('name', 'description', 'requires', 'provides')},
                operation=source.get('operation'))
        hypothesis = Hypothesis(rationale=idea.change, expected_effect=idea.expected_effect,
                                pipeline=self.pipeline.configuration(), research_ids=idea.research_ids)
        return Proposal(purpose=idea.purpose, parent_id=idea.parent_id, hypothesis=hypothesis,
                        assembly=Assembly(blocks=[blocks[name] for name in self.order],
                                          write_blocks=[block.name for block in self.write]))

    @model_validator(mode='after')
    def assembled_plan(self, info: ValidationInfo):
        if any(block.operation is None for block in self.write):
            raise ValueError('Every written block must select a trusted operation; Python generation is disabled')
        from .operations import CATALOG
        for block in self.write:
            call = block.operation
            entry = CATALOG[call.kind]
            required = {call.inputs.get(key, key) for key in entry['requires']}
            if call.kind == 'ensemble':
                required.update(key for pair in call.parameters.get('sources', []) for key in pair)
            provided = {call.outputs.get(key, key) for key in entry['provides']}
            if not required <= set(block.requires) or not provided <= set(block.provides):
                raise ValueError(f'{block.name}: interface must include the operation inputs/outputs after aliases')
        names = [block.name for block in self.write] + [block.name for block in self.reuse]
        if len(names) != len(set(names)) or len(self.order) != len(set(self.order)) or set(names) != set(self.order):
            raise ValueError('order must contain every written or reused block exactly once')
        context = info.context or {}
        if 'idea' in context:
            missing = set(context['idea'].change_blocks) - {block.name for block in self.write}
            if missing:
                raise ValueError(f'Write every block requested by the idea: {sorted(missing)}')
            proposal = self.proposal(context['idea'], context['catalog'])
            from .feature_trials import check_recipes
            check_recipes(proposal, context.get('selected_feature_candidates', []))
            Proposals.model_validate({'proposals': [proposal]}, context={**context, 'count': 1,
                'baseline': proposal.purpose == 'baseline', 'purposes': [proposal.purpose]})
        return self


class Analysis(Contract):
    summary: str = Field(min_length=1)
    findings: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    next_ideas: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)


class Program(Contract):
    code: str = Field(default='', description='Full source of ONLY the requested block, defining run(state)')
    notes: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    @model_validator(mode='after')
    def code_or_reason(self):
        if not self.code.strip() and not (self.blocked_reason or '').strip():
            raise ValueError('Return block code or blocked_reason')
        return self


ResearchID = Annotated[str, Field(pattern=r'^research_[0-9]{3}$')]


class ResearchPlan(Contract):
    parent_id: ResearchID | None = None
    question: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    method: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    analyses: list[AnalysisRequest] = Field(min_length=1, description='Executable feature-analysis requests; no generated Python')
    alternative_reason: str | None = Field(default=None, description='Required if the assigned research direction cannot be tested: explain the data limitation and why the alternative is new')

    @model_validator(mode='after')
    def unique_requests(self):
        names = [item.name for item in self.analyses]
        if len(names) != len(set(names)):
            raise ValueError('Analysis result names must be unique')
        return self


class DataInsight(Contract):
    statement: str = Field(min_length=1)
    kind: Literal['observation', 'interpretation']
    evidence_keys: list[str] = Field(min_length=1)
    implication: str = Field(min_length=1)
    importance: Literal['low', 'medium', 'high', 'critical'] = Field(default='medium', description='Decision impact: critical invalidates task/validation; high changes a major modeling decision; medium is a useful refinement; low is incidental')
    confidence: Literal['low', 'medium', 'high'] = Field(default='medium', description='Strength of measured support, not subjective certainty; a proxy gain alone does not prove outer/test improvement')
    topic: str = Field(default='', description='Short stable topic including the relevant feature/population; reuse for the same claim across studies')


class ResearchReview(Contract):
    outcome: Literal['supported', 'refuted', 'inconclusive']
    summary: str = Field(min_length=1)
    insights: list[DataInsight] = Field(default_factory=list)
    open_questions: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    resolves: list[str] = Field(default_factory=list, description='Previously open questions answered by this research')
    supersedes: list[str] = Field(default_factory=list, description='Existing insight IDs contradicted or refined by these measurements')

    @model_validator(mode='after')
    def grounded(self, info: ValidationInfo):
        context = info.context or {}
        if 'measurements' in context:
            for insight in self.insights:
                if not set(insight.evidence_keys).issubset(context['measurements']):
                    raise ValueError('Insight evidence must name measured output keys')
                if any(isinstance(context['measurements'][key], dict) and
                       context['measurements'][key].get('status') in {'failed', 'unavailable'}
                       for key in insight.evidence_keys):
                    raise ValueError('Failed or unavailable measurements cannot support an insight')
        if 'insight_ids' in context and not set(self.supersedes).issubset(context['insight_ids']):
            raise ValueError('Only existing insights can be superseded')
        if self.supersedes and not self.insights:
            raise ValueError('Superseding an insight requires a replacement')
        if 'open_questions' in context:
            known = {' '.join(question.split()).casefold(): question for question in context['open_questions']}
            resolves = list(dict.fromkeys(known[' '.join(question.split()).casefold()]
                for question in self.resolves if ' '.join(question.split()).casefold() in known))
            if resolves != self.resolves:
                return self.model_copy(update={'resolves': resolves})
        return self

class Start(Contract):
    schema_version: Literal[4, 5] = 5
    node_id: NodeID = "node_000"
    parent_id: NodeID | None = None
    source: Literal["qwen", "builtin"] = "qwen"
    preparation: str = "../../preparation.json"
    preparation_sha256: SHA256
    context: dict[str, Any] = Field(default_factory=dict)
    purpose: Literal['baseline', 'improve', 'explore']
    hypothesis: Hypothesis
    assembly: Assembly
    implementation: str = 'experiment.py'
    outputs: dict[str, str]
    timeout_seconds: float = Field(gt=0)


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
    node_id: NodeID | ResearchID | None = None
    block: str | None = None
    attempt: int
    status: Literal["completed", "invalid_json", "api_error"]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class RunReport(Contract):
    schema_version: Literal[2] = 2
    status: Literal["running", "ready", "partial", "failed"]
    stage: Literal["extraction", "preparation", "data_research", "baseline", "search", "final_refit", "complete"] = "extraction"
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
    max_parallel_nodes: int = 6
    search_parallel_limit: int = 6
    api_budget: dict[str, Any] = Field(default_factory=dict)
    max_parallel_agents: int = 8
    active_agents: list[dict[str, Any]] = Field(default_factory=list)
    node_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    max_parallel_research: int = 0
    researching_nodes: list[str] = Field(default_factory=list)
    research_nodes: int = 0
    research_revision: int = 0
    research_insights: int = 0
    research_stop_reason: str | None = None


SCHEMAS = (Extraction, Preparation, Start, Finish, RunReport, Proposals, Ideas, NodePlan, Analysis, Assembly,
           NodeDelta, Delta, ResearchPlan, ResearchReview, SplitPlan)
