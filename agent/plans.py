"""Small executable requests shared by Baker, AutoML and data research."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False, frozen=True)


class FeatureSpec(Plan):
    name: str = Field(min_length=1)
    operation: Literal['sum', 'mean', 'product', 'difference', 'ratio', 'signed_log',
                       'missing', 'frequency', 'clip', 'group_mean', 'datetime', 'expression',
                       'history', 'target_encoding']
    columns: list[str] = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict, description='For expression use {expression: formula}; x0/x1 or col(quoted column) read declared columns. Compose arithmetic, conditions, log1p, abs, sqrt, where, clip, frequency and group_mean/group_std/group_count. Statistics learn only on fit.')

    @model_validator(mode='after')
    def supervised_parameters(self):
        if self.operation == 'history':
            HistorySettings.model_validate(self.parameters)
        elif self.operation == 'target_encoding':
            TargetEncodingSettings.model_validate(self.parameters)
        return self


class HistorySettings(Plan):
    time_column: str
    value_column: str
    aggregation: Literal['last', 'mean', 'sum', 'std', 'min', 'max', 'count', 'trend', 'age'] = 'last'
    window: int | None = Field(default=None, ge=1, description='Number of distinct earlier periods, not rows')
    lag: int = Field(default=1, ge=1, description='1 starts at the latest strictly earlier period')
    min_history: int = Field(default=1, ge=1)
    max_age: float | None = Field(default=None, gt=0, description='Time units for numeric time, days for timestamps')
    fallback: Literal['missing', 'global'] = 'missing'
    class_label: Any = Field(default=None, description='Encode this class as 1, others as 0; omit for numeric values')


class TargetEncodingSettings(Plan):
    target: str
    smoothing: float = Field(default=10., ge=0)
    folds: int = Field(default=5, ge=2)
    seed: int = 42
    validation: Literal['auto', 'random', 'group', 'time'] = 'auto'
    group_column: str | None = None
    time_column: str | None = None
    class_label: Any = None


class RowWeights(Plan):
    time_column: str
    half_life: float = Field(gt=0, description='Numeric time units, or days for timestamps; weight halves per half_life')


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
    split_column: str | None = Field(default=None, description='Optional explicit ordered column for the inner holdout; outer split stays fixed')
    group_by: str | None = Field(default=None, description='Optional explicit grouping column for the inner holdout')
    bag_folds: int = Field(default=0, ge=0)
    stack_levels: int = Field(default=0, ge=0)
    ensemble: bool | None = None
    budget_fraction: float = Field(default=1.0, gt=0, le=1)
    columns: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    features: list[FeatureSpec] = Field(default_factory=list)
    row_weights: RowWeights | None = None

    @model_validator(mode='before')
    @classmethod
    def engine_model_names(cls, value):
        from .automl_catalog import normalize_models
        return normalize_models(value)

    @model_validator(mode='after')
    def supported_combination(self):
        if (self.features or self.row_weights) and (self.validation == 'cv' or self.bag_folds or self.stack_levels):
            raise ValueError('Feature recipes require validation=holdout with bag_folds=stack_levels=0; the adapter rebuilds recipes on its explicit inner train')
        if self.engine == 'flaml' and (self.features or self.row_weights) and self.ensemble:
            raise ValueError('Use separate saved ensembles after feature-aware FLAML; internal stacking cannot rebuild recipes inside its folds')
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
    optimize_weights: bool = Field(default=False, description='Tune nonnegative blend weights on saved inner-holdout predictions, never outer validation')
    weight_trials: int = Field(default=64, ge=1, description='Maximum local weight candidates, no LLM calls')
    combine: Literal['mean', 'vote', 'probabilities'] | None = None
    rules: list[BlendRule] = Field(default_factory=list, description='First matching rule wins; default weights apply otherwise')
    estimator: str | None = Field(default=None, description='Installed sklearn-compatible class path; stacking defaults to LogisticRegression/Ridge, gating to shallow DecisionTreeClassifier with predict_proba')
    parameters: dict[str, Any] = Field(default_factory=dict, description='Combiner constructor kwargs, not source-model settings')
    feature_columns: list[str] = Field(default_factory=list, description='Raw features for gating or stacking passthrough; [] selects all allowed features')
    passthrough: bool = Field(default=False, description='Stacking also sees preprocessed raw features alongside source predictions; supports nonlinear dependencies')

    @model_validator(mode='before')
    @classmethod
    def legacy_oof_settings(cls, value):
        # Old saved plans may contain these; shared holdout no longer varies per ensemble.
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if key not in {'folds', 'seed'}}
        return value

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
        if self.optimize_weights and self.method not in {'mean', 'vote', 'probabilities'}:
            raise ValueError('Weight optimization applies to mean, vote or probability blending')
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
            'preprocess': {'feature_columns', 'drop_columns', 'numeric_imputer', 'encoding', 'native_backend', 'scaling', 'pca_components'},
            'fit': {'estimator', 'parameters', 'search_space', 'num_trials', 'validation', 'group_by', 'split_column', 'row_weights', 'seeds'},
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
        if self.kind == 'fit':
            if self.parameters.get('row_weights') is not None:
                RowWeights.model_validate(self.parameters['row_weights'])
            seeds = self.parameters.get('seeds', [])
            if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds):
                raise ValueError('seeds must be a list of integers')
        block = (info.context or {}).get('block')
        if block is not None:
            original = block.get('operation')
            if original:
                if self.kind != original['kind']:
                    raise ValueError('Repair must preserve the operation kind and hypothesis')
                fixed = ('method', 'source_nodes') if self.kind == 'saved_ensemble' else (
                    ('engine',) if self.kind == 'automl' else ())
                for key in fixed:
                    if self.parameters.get(key) != original.get('parameters', {}).get(key):
                        raise ValueError(f'Repair must preserve {key}; create a new hypothesis to change it')
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
                  'missingness', 'interactions', 'predictive', 'permutation', 'ablation', 'feature', 'stability', 'entities']
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
    key_candidates: list[list[str]] = Field(default_factory=list, description='For entities: proposed single/composite keys; [] uses columns as one key. No target access required.')
    filters: list[RowFilter] = Field(default_factory=list)
    comparison_filters: list[RowFilter] = Field(default_factory=list)
