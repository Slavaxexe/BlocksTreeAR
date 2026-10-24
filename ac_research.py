"""Classification AutoResearch: measured hypotheses, Qwen, and heuristic fallback."""
import json
import time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ac_experiment import ExperimentSpec, run_in_subprocess
from ac_models import DEFAULTS, PARAM_SPACE


class Proposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    hypothesis: str
    why: str
    parent: str | None = None
    spec: ExperimentSpec


class Analysis(BaseModel):
    model_config = ConfigDict(extra='forbid')
    experiment_id: str
    lesson: str


class Step(BaseModel):
    model_config = ConfigDict(extra='forbid')
    analysis: Analysis
    new_insights: list[str] = Field(default_factory=list, max_length=3)
    backlog_add: list[str] = Field(default_factory=list, max_length=5)
    decision: str = 'experiment'
    proposal: Proposal | None = None


METHOD = '''You are the classification AutoResearch scientist. Return one populated JSON DATA INSTANCE, never
the JSON Schema itself. Analyze the LAST measured experiment, record a short numerical lesson, and propose
ONE justified experiment relative to a parent, or stop. The code owns splits, metrics, execution, and timing.
The PRIMARY task metric is fixed: maximize accuracy/balanced_accuracy/F1/ROC-AUC, minimize log_loss.
Class labels are categorical, never numeric regression targets: do NOT average class codes, log-transform
the target, use Gamma/Poisson/MSE regression losses, or clip labels. Classifier loss is cross_entropy.
Available actions: choose a classifier and supported parameters from CATALOG; add/drop feature blocks;
use balanced class weights or recent sample weights when temporal; tune a binary threshold, never a
multiclass threshold. Native estimators and probability class alignment are implemented by code.
Temporal history consists of CLASS PROBABILITIES from strictly past labels, fitted separately inside each
outer fold. No-time target encoding is cross-fitted entirely inside outer-train. Validation/test labels
are unavailable to code generation. Never request them. All candidates share the same outer folds.
An apparent improvement on repeatedly inspected OOF folds is selection evidence, not an unbiased or
hidden-test score. Check per-class errors and imbalance; do not optimize accuracy when task asks Macro F1.
Prefer one change at a time; test feature history only if available. Avoid repeats and honor remaining time.
Voluntary decision=stop is allowed only after at least four successful experiments. Before that, propose
an inexpensive justified alternative even if the measured metric is perfect; the timer can stop sooner.
Optional spec.feature_code defines ONLY def add_features(df):, copies df, adds 1..32 numeric row-local
features and returns the copy with original columns unchanged. np and pd are supplied. No imports, IO,
private attributes, loops, global statistics, groupby, row reordering or labels. Use actual feature names.
Example: def add_features(df):\n    out = df.copy()\n    out['ratio'] = df['x'] / df['z'].replace(0, np.nan)\n    return out
The AST and subprocess errors will be returned for repair. Preserve original template APIs.
'''


class Lab:
    def __init__(self, mem, settings, budget, features, context, root):
        self.mem, self.settings, self.budget, self.features, self.context, self.root = mem, settings, budget, features, context, root
        self.custom_codes = set()

    def reserve(self):
        best = sorted(self.mem.ok_experiments(), key=lambda e: e['cv'], reverse=self.mem.direction == 'maximize')[:4]
        folds = max(1, len(self.context['fold_weights']))
        requested = min(180., self.budget.total_s*.15)+sum(1.7*e['seconds']/folds+10 for e in best)
        return min(self.budget.total_s*.45, max(self.budget.total_s*.12, requested))

    def left(self):
        return self.budget.deadline-time.monotonic()-self.reserve()

    def validate(self, spec):
        errors = []
        if set(spec.blocks)-set(self.features.available()):
            errors.append('Unavailable feature blocks; use exactly CATALOG.blocks')
        if spec.training.sample_weight == 'recent' and not self.context['time_column']:
            errors.append('Recent weights require a detected time column')
        if len(self.context['classes']) != 2 and spec.training.threshold != .5:
            errors.append('Threshold changes apply only to binary classification')
        if spec.key() in self.mem.spec_hashes():
            errors.append('Duplicate experiment; change its specification')
        if spec.feature_code and spec.feature_code not in self.custom_codes and len(self.custom_codes) >= 6:
            errors.append('Maximum six distinct generated feature functions per run')
        return errors

    def run(self, spec, hypothesis, why, parent, source):
        errors = self.validate(spec)
        if errors:
            raise ValueError('; '.join(errors))
        identifier = f'exp_{len(self.mem.state["experiments"])+1:03d}'
        previous = self.mem.best()
        context = {**self.context, 'mode': 'cv'}
        self.mem.state['active_experiment'] = {'id': identifier, 'spec': spec.model_dump()}
        self.mem.log('experiment_started', id=identifier, spec=spec.model_dump(), source=source)
        self.mem.save()
        seconds = min(self.settings.experiment_timeout_s, max(.1, self.left()))
        result = run_in_subprocess(spec, context, self.root/'experiments'/identifier, seconds)
        record = {'id': identifier, 'parent': parent, 'source': source, 'hypothesis': hypothesis,
                  'why': why, 'spec': spec.model_dump(), 'spec_hash': spec.key(), 'summary': spec.summary(), **result}
        if spec.feature_code:self.custom_codes.add(spec.feature_code)
        if result.get('status') == 'ok':
            sign = 1 if self.mem.direction == 'minimize' else -1
            gain = sign*(previous['cv']-result['cv'])/max(abs(previous['cv']), 1e-12) if previous else None
            record['delta_best_pct'] = 100*gain if gain is not None else None
            record['verdict'] = 'baseline' if previous is None else 'new_best' if gain >= .002 else 'no_gain' if abs(gain) < .002 else 'worse'
        else:
            record['verdict'] = 'failed'
        self.mem.add_experiment(record)
        self.mem.state['active_experiment'] = None
        self.mem.save()
        return record

    def heuristic(self):
        best = self.mem.best()
        if not best:return None
        original = best['spec']
        variants = []
        for block in ('entity_history', 'group_history', 'similar_history', 'missing', 'families', 'log_numeric', 'oof_target_encoding'):
            if block in self.features.available() and block not in original['blocks']:
                variants.append({**original, 'blocks': [*original['blocks'], block]})
        for model_id in PARAM_SPACE:
            for balance in (None, 'balanced'):
                parameters = {}
                if balance:
                    parameters['auto_class_weights' if model_id == 'catboost' else 'class_weight'] = 'Balanced' if model_id == 'catboost' else balance
                variants.append({**original, 'model': {'model_id': model_id, 'loss': 'cross_entropy', 'params': parameters}})
        family = original['model']['model_id']
        for parameters in ({'num_leaves': 15, 'reg_lambda': 5.}, {'num_leaves': 63, 'min_child_samples': 50}) if family == 'lightgbm' else ():
            variants.append({**original, 'model': {**original['model'], 'params': {**original['model']['params'], **parameters}}})
        if len(self.context['classes']) == 2:
            for threshold in (.35, .65):
                variants.append({**original, 'training': {**original['training'], 'threshold': threshold}})
        for value in variants:
            spec = ExperimentSpec.model_validate(value)
            if not self.validate(spec):
                return Proposal(hypothesis='One controlled feature/model/balancing change from measured winner',
                    why=f'Best measured {self.context["metric"]}={best["cv"]:.6f}; test complementary information or capacity.',
                    parent=best['id'], spec=spec)
        return None


def research_loop(lab, llm, log, on_baseline):
    record = lab.run(ExperimentSpec(), 'Baseline LightGBM classifier', 'Measured point of comparison', None, 'baseline')
    log(f'{record["id"]} baseline: {record.get("cv")} [{record["status"]}] {record.get("error", "")}')
    on_baseline(record)
    rejected = 0
    while len(lab.mem.state['experiments']) < lab.settings.max_experiments:
        if lab.left() < min(15., lab.budget.total_s*.04):return 'time_budget'
        completed = lab.mem.ok_experiments()
        if len(completed) >= 13:
            latest = max(i for i, e in enumerate(lab.mem.state['experiments']) if e.get('verdict') in {'baseline', 'new_best'})
            if len(lab.mem.state['experiments'])-latest-1 >= lab.settings.patience:return 'patience'
        proposal = None
        if llm.enabled and rejected < 3:
            last = lab.mem.state['experiments'][-1]['id']
            def validate(step):
                errors = []
                if step.analysis.experiment_id != last:errors.append('Analyze the LAST experiment id')
                if step.decision not in {'experiment', 'stop'}:errors.append('decision must be experiment or stop')
                if step.decision == 'stop':
                    if len(completed) < 4:errors.append('Complete at least four useful experiments before voluntary stop')
                elif step.proposal is None:errors.append('experiment requires a proposal')
                else:
                    errors += lab.validate(step.proposal.spec)
                    if step.proposal.parent and step.proposal.parent not in {e['id'] for e in lab.mem.state['experiments']}:
                        errors.append('Unknown parent experiment')
                return errors
            catalog = {'blocks': lab.features.available(), 'classifiers': PARAM_SPACE, 'defaults': DEFAULTS,
                       'classes': lab.context['classes'], 'primary_metric': lab.context['metric']}
            memory = lab.mem.render(lab.settings.memory_chars, {'research_seconds_left': int(lab.left()),
                    'final_reserve_seconds': lab.reserve(), 'experiments_done': len(lab.mem.state['experiments'])})
            log(f'Qwen analyzing {last} and planning the next classification hypothesis')
            step = llm.ask('classification_researcher', METHOD+'\nCATALOG:\n'+json.dumps(catalog, ensure_ascii=False)
                +'\nVALIDATION SCHEMA (return DATA values):\n'+json.dumps(Step.model_json_schema()),
                'MEASURED MEMORY:\n'+memory, Step, deadline=lab.budget.deadline-lab.reserve(), validator=validate)
            if step:
                rejected = 0
                lab.mem.annotate_experiment(last, analysis=step.analysis.lesson)
                for insight in step.new_insights:lab.mem.add_insight(insight, 'qwen')
                lab.mem.add_backlog(step.backlog_add, 'qwen')
                if step.decision == 'stop':return 'qwen_stop'
                proposal = step.proposal
            else:
                rejected += 1
        source = 'qwen' if proposal is not None else 'heuristic'
        proposal = proposal or lab.heuristic()
        if proposal is None:return 'ideas_exhausted'
        record = lab.run(proposal.spec, proposal.hypothesis, proposal.why, proposal.parent, source)
        log(f'{record["id"]} [{source}] {record["summary"]} cv={record.get("cv")} {record["verdict"]} {record.get("error", "")}')
    return 'max_experiments'
