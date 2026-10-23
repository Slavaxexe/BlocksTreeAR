"""Small executable requests shared by Baker, AutoML and data research."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False, frozen=True)


class FeatureSpec(Plan):
    name: str = Field(min_length=1)
    operation: Literal['sum', 'mean', 'product', 'difference', 'ratio', 'signed_log',
                       'missing', 'frequency', 'clip', 'group_mean', 'datetime']
    columns: list[str] = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class SearchSpace(Plan):
    kind: Literal['int', 'float', 'categorical']
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: list[Any] = Field(default_factory=list)

    @model_validator(mode='after')
    def valid_domain(self):
        if self.kind == 'categorical':
            if not self.choices:
                raise ValueError('Categorical search needs choices')
        elif self.low is None or self.high is None or self.low >= self.high or (self.log and self.low <= 0):
            raise ValueError('Search needs low < high, with positive bounds for a log domain')
        return self


class AutoMLPlan(Plan):
    engine: Literal['flaml', 'autogluon']
    models: list[str] = Field(default_factory=list)
    model_parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    search_space: dict[str, dict[str, SearchSpace]] = Field(default_factory=dict)
    num_trials: int | None = Field(default=None, ge=1)
    validation: Literal['holdout', 'cv'] = 'holdout'
    folds: int = Field(default=3, ge=2)
    holdout_fraction: float = Field(default=0.2, gt=0, lt=1)
    bag_folds: int = Field(default=0, ge=0)
    stack_levels: int = Field(default=0, ge=0)
    ensemble: bool | None = None
    budget_fraction: float = Field(default=1.0, gt=0, le=1)
    columns: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    features: list[FeatureSpec] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def engine_model_names(cls, value):
        from .automl_catalog import normalize_models
        return normalize_models(value)

    @model_validator(mode='after')
    def supported_combination(self):
        if self.bag_folds == 1:
            raise ValueError('Bagging needs zero or at least two folds')
        if self.engine == 'flaml' and (self.bag_folds or self.stack_levels):
            raise ValueError('Use validation=cv and ensemble for FLAML; bag_folds/stack_levels are AutoGluon settings')
        if self.engine == 'autogluon' and self.stack_levels and not (self.bag_folds or self.validation == 'cv'):
            raise ValueError('AutoGluon stacking needs bagging: set bag_folds or validation=cv')
        return self


class RowFilter(Plan):
    column: str
    operation: Literal['eq', 'ne', 'lt', 'le', 'gt', 'ge', 'isin', 'isna', 'notna']
    value: Any = None


class BlendRule(Plan):
    conditions: list[RowFilter] = Field(min_length=1)
    weights: list[float] = Field(min_length=2)


class EnsemblePlan(Plan):
    source_nodes: list[str] = Field(min_length=2, description='Completed nodes from prediction_sources, on the same preparation')
    method: Literal['mean', 'vote', 'probabilities', 'conditional', 'stacking', 'gating']
    weights: list[float] = Field(default_factory=list)
    combine: Literal['mean', 'vote', 'probabilities'] | None = None
    rules: list[BlendRule] = Field(default_factory=list, description='First matching rule wins; default weights apply otherwise')
    folds: int = Field(default=3, ge=2)
    seed: int = 42
    estimator: str | None = Field(default=None, description='Installed sklearn-compatible class path; stacking defaults to LogisticRegression/Ridge, gating to shallow DecisionTreeClassifier with predict_proba')
    parameters: dict[str, Any] = Field(default_factory=dict, description='Combiner constructor kwargs, not source-model settings')
    feature_columns: list[str] = Field(default_factory=list, description='Raw features for gating or stacking passthrough; [] selects all allowed features')
    passthrough: bool = Field(default=False, description='Stacking also sees preprocessed raw features alongside source predictions; supports nonlinear dependencies')

    @model_validator(mode='after')
    def compatible_sources(self):
        import re
        if len(set(self.source_nodes)) != len(self.source_nodes) or any(
                not re.fullmatch(r'node_[0-9]{3}', node) for node in self.source_nodes):
            raise ValueError('Select distinct saved node IDs')
        for weights in [self.weights, *(rule.weights for rule in self.rules)]:
            if weights and (len(weights) != len(self.source_nodes) or min(weights) < 0 or sum(weights) <= 0):
                raise ValueError('Weights must match sources and be nonnegative with a positive sum')
        if self.method == 'conditional' and not self.rules:
            raise ValueError('Conditional blending needs at least one rule')
        if self.method != 'conditional' and self.rules:
            raise ValueError('Rules apply only to conditional blending')
        return self


class ToolCall(Plan):
    kind: Literal['constant', 'features', 'preprocess', 'fit', 'predict', 'automl', 'ensemble', 'saved_ensemble']
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict, description='Map canonical operation inputs to saved state keys')
    outputs: dict[str, str] = Field(default_factory=dict, description='Map canonical outputs to custom state keys')


    @model_validator(mode='before')
    @classmethod
    def canonical_parameters(cls, value):
        if not isinstance(value, dict):
            return value
        parameters = value.get('parameters', {})
        if value.get('kind') == 'automl':
            parameters = AutoMLPlan.model_validate(parameters).model_dump()
        elif value.get('kind') == 'saved_ensemble':
            parameters = EnsemblePlan.model_validate(parameters).model_dump()
        elif value.get('kind') == 'preprocess' and isinstance(parameters, dict) and 'columns' in parameters:
            parameters = dict(parameters)
            columns = parameters.pop('columns')
            if 'feature_columns' in parameters and parameters['feature_columns'] != columns:
                raise ValueError('Specify feature_columns once; columns is its legacy alias')
            parameters['feature_columns'] = columns
        return {**value, 'parameters': parameters}

    def interface(self):
        from .operations import CATALOG
        entry = CATALOG[self.kind]
        required = [self.inputs.get(key, key) for key in entry['requires']]
        if self.kind == 'ensemble':
            required += [key for pair in self.parameters.get('sources', []) for key in pair]
        provided = [self.outputs.get(key, key) for key in entry['provides']]
        return list(dict.fromkeys(required)), list(dict.fromkeys(provided))

    @model_validator(mode='after')
    def operation_parameters(self, info: ValidationInfo):
        fields = {
            'constant': set(), 'features': {'recipes'},
            'preprocess': {'feature_columns', 'drop_columns', 'numeric_imputer', 'encoding', 'scaling', 'pca_components'},
            'fit': {'estimator', 'parameters', 'search_space', 'num_trials', 'validation', 'group_by', 'split_column'},
            'predict': {'output'}, 'ensemble': {'sources', 'weights', 'method'},
        }
        if self.kind == 'automl':
            AutoMLPlan.model_validate(self.parameters)
        elif self.kind == 'saved_ensemble':
            EnsemblePlan.model_validate(self.parameters)
        else:
            unknown = set(self.parameters) - fields[self.kind]
            if unknown:
                raise ValueError(f'Unknown {self.kind} settings: {sorted(unknown)}')
        block = (info.context or {}).get('block')
        if block is not None:
            from .operations import CATALOG
            entry = CATALOG[self.kind]
            required = {self.inputs.get(key, key) for key in entry['requires']}
            if self.kind == 'ensemble':
                required.update(key for pair in self.parameters.get('sources', []) for key in pair)
            provided = {self.outputs.get(key, key) for key in entry['provides']}
            if not required <= set(block['requires']) or not set(block['provides']) <= provided:
                raise ValueError('A repaired operation must preserve the block input/output interface')
        return self


class AnalysisRequest(Plan):
    name: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    kind: Literal['quality', 'association', 'correlation', 'shift', 'groups', 'redundancy',
                  'missingness', 'interactions', 'predictive', 'permutation', 'ablation', 'feature', 'stability']
    columns: list[str] = Field(default_factory=list)
    target: str | None = None
    dataset: Literal['fit', 'validation', 'test'] = Field(default='fit', description='Actual table for descriptive analyses; supervised analyses use fit only')
    compare: Literal['validation', 'test'] = Field(default='test', description='Other table for shift only; does not select the table for other analyses')
    group_by: str | None = None
    baseline_columns: list[str] = Field(default_factory=list)
    features: list[FeatureSpec] = Field(default_factory=list)
    sample_rows: int | None = Field(default=5000, ge=20)
    top_k: int | None = Field(default=10, ge=1)
    seed: int = 42
    estimator: Literal['extra_trees', 'hist_gradient_boosting', 'linear'] = 'extra_trees'
    repeats: int = Field(default=3, ge=1)
    validation: Literal['auto', 'random', 'group', 'time'] = 'auto'
    validation_fraction: float = Field(default=0.25, gt=0, lt=1)
    split_column: str | None = None
    filters: list[RowFilter] = Field(default_factory=list)
    comparison_filters: list[RowFilter] = Field(default_factory=list)
