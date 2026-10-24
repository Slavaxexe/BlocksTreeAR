"""Classification AutoResearch: measured hypotheses, Qwen, and heuristic fallback."""
import json
import time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ac_experiment import ExperimentSpec, run_in_subprocess
from ac_models import DEFAULTS, PARAM_SPACE
from ac_diagnostics import compact_monitor, prediction_monitor
from ac_eda import FeaturePolicy, compact_profile, confusion_evidence
from ac_strategy import required_trials, protocol_status


class Proposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    hypothesis: str
    why: str
    parent: str | None = None
    spec: ExperimentSpec
    eda_evidence: list[str] = Field(default_factory=list, max_length=5)


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
Class-balance monitoring is MEASURED, not a command to maximize rare-class recall at any cost.
Imbalance decision rules:
1. Keep an unweighted baseline and compare moderate clipped weights when the metric values minority classes.
2. If recall is low but precision is reasonable, consider a modest increase in weights or better class-separating features.
3. If a class is overpredicted and precision collapses, DECREASE its weighting strength; do not increase weights just because it is rare.
4. If both precision and recall are weak, inspect conflicting feature-identical labels, class confusions,
   categorical encodings and alternative model families. Weighting cannot recover missing information.
5. For log_loss/ROC-AUC do not optimize predicted class balance; compare probabilities using the primary metric.
6. After several failed weight/leaf/lambda tweaks, switch hypothesis family rather than another tiny numeric change.
SEARCH_PROTOCOL is enforced by the coordinator: applicable EDA encodings, feature-quality/tail probes and
model-family diversity are evaluated before open-ended parameter tuning. HISTORICAL_REFERENCE is only
a candidate rechecked on identical current data/folds, never a score that can be claimed without training.
A bounded model-family probe has a small iteration/time cap. Its result does not establish the best
possible score for the entire family; consider expanding a promising probe if the remaining budget permits.
Use CURRENT/BEST_DIAGNOSTICS precision, recall, F1, predicted counts and confusion pairs. Test soft
training.class_weight_power in [0,1] and class_weight_clip in [1,100] before assuming full balanced
weights help. Weights are fitted only on each training partition. Never combine soft and native class
weights. Keep changes only by PRIMARY metric, not by balancing predicted class frequencies to train.
Inspect EDA ambiguity, duplicate labels and potential categorical/absence indicators before inventing
features. A rare class may be underpredicted because features are ambiguous, not because it needs more weight.
For log_loss/ROC-AUC, probability quality matters; no automatic minority balancing is required.
The budget field search_seconds_available ALREADY EXCLUDES final_reserve_seconds. Use it directly,
never subtract that reserve twice. A non-improving experiment can still help a probability ensemble.
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
Use spec.preprocessing for explicit EDA experiments: exact drop_columns, constants/duplicate removal,
correlation filtering, missingness indicators, categorical overrides, rare-category merging, frequency
encoding, absence indicators, numeric arithmetic and category-pair frequency interactions, optional
quantile clipping/outlier flags. These are fitted on each FIT/TRAIN partition independently.
For any non-default preprocessing or generated function, cite measured observations in proposal.eda_evidence.
Treat correlation, skew, train/test shift, placeholder values and label ambiguity as hypotheses, not proof
that columns or rare rows must be deleted. Never remove labels/rows or force train/test class balances.
Change one justified feature hypothesis at a time and compare PRIMARY metric on the same full folds.
A timeout is a runtime observation, not a poor model score. Do not repeat an expensive timed-out spec.
Use fewer local boosting rounds via training.max_boost_rounds, simpler trees, fewer heavy feature blocks,
or a different local classifier. Qwen/API credentials never change during recovery.
'''


class Lab:
    def __init__(self, mem, settings, budget, features, context, root):
        self.mem, self.settings, self.budget, self.features, self.context, self.root = mem, settings, budget, features, context, root
        self.custom_codes = set()
        self._diagnostic_sample = None
        self.protocol_deferred = {}

    def coverage(self):
        profile = self.mem.get('audit.feature_profile', {})
        profile = {**profile, 'categorical': [row for row in profile.get('categorical', [])
            if row['feature'] != self.context.get('time_column')]}
        blocks = self.features.available() if self.features is not None else self.mem.get('feature_blocks', {})
        trials = required_trials(profile, self.mem.get('audit.dataset_profile', {}), blocks)
        if any(record.get('balance_monitor', {}).get('soft_weights_worth_testing') for record in self.mem.state['experiments']):
            trials.insert(0, {'id': 'imbalance_check', 'title': 'Compare moderate class weights by the primary metric',
                'balance_trial': True, 'evidence': ['Measured minority underprediction and imbalance justify a controlled weighting trial.']})
        enabled = self.budget.total_s >= 180 and self.settings.max_experiments >= 5
        return {'enabled': enabled, 'trials': protocol_status(self.mem.state['experiments'], trials, self.protocol_deferred)}, trials

    def history_proposal(self):
        if self.budget.total_s < 180 or self.settings.max_experiments < 5:return None
        reference = self.mem.get('search.previous_winner')
        if not reference or any(record.get('source') == 'history_recheck' for record in self.mem.state['experiments']):return None
        try:spec = ExperimentSpec.model_validate(reference['spec'])
        except ValueError:return None
        if self.validate(spec):return None
        if self.left() < self.estimate_seconds(spec):return None
        best = self.mem.best()
        return Proposal(hypothesis='Recheck the strongest saved candidate on identical current data and validation folds',
            why=f'Historical reference CV={reference["reference_cv"]:.6f}; this is not a current measured score until re-evaluated.',
            parent=best['id'] if best else None, spec=spec,
            eda_evidence=['Data, target classes, primary metric and fold arrays have an identical fingerprint.'])

    def protocol_proposal(self):
        coverage, trials = self.coverage()
        if not coverage['enabled']:return None
        best = self.mem.best()
        if not best:return None
        for status, trial in zip(coverage['trials'], trials):
            if status['status'] != 'pending':continue
            value = json.loads(json.dumps(best['spec']))
            if value['model']['model_id'] == 'prior':
                value = ExperimentSpec().model_dump()
            if trial.get('model_alternative'):
                family = trial.get('preferred_model', 'catboost')
                parameters = {'depth': 5, 'learning_rate': .08, 'l2_leaf_reg': 3.} if family == 'catboost' else {'n_estimators': 120, 'max_features': .8, 'min_samples_leaf': 2}
                value['model'] = {'model_id': family, 'loss': 'cross_entropy', 'params': parameters}
                value['training']['max_boost_rounds'] = 64
            if trial.get('balance_trial'):
                value['model']['params'].pop('class_weight', None)
                value['model']['params'].pop('auto_class_weights', None)
                value['training'].update(class_weight_power=.5, class_weight_clip=20.)
            if trial.get('blocks_add'):
                value['blocks'] = list(dict.fromkeys([*value['blocks'], *trial['blocks_add']]))
            if trial.get('preprocessing'):
                value['preprocessing'] = {**value.get('preprocessing', {}), **trial['preprocessing']}
            try:spec = ExperimentSpec.model_validate(value)
            except ValueError:
                self.protocol_deferred[trial['id']] = 'No valid compatible feature policy for the current candidate'
                continue
            if self.validate(spec):continue
            estimate = self.estimate_seconds(spec)
            if self.left() < estimate:
                self.protocol_deferred[trial['id']] = f'Needs approximately {estimate:.0f}s; {max(0., self.left()):.0f}s search budget available'
                continue
            coverage, _ = self.coverage();self.mem.fact('search.protocol', coverage);self.mem.save()
            return trial['id'], Proposal(hypothesis=trial['title'],
                why='Required measured search coverage; retain this change only if the primary validation metric supports it.',
                parent=best['id'], spec=spec, eda_evidence=trial['evidence'])
        coverage, _ = self.coverage();self.mem.fact('search.protocol', coverage);self.mem.save()
        return None

    def research_memory(self):
        """Classification diagnostics must survive prompt compaction; ar_* memory stays unchanged."""
        best = self.mem.best()
        recent = self.mem.state['experiments'][-1]
        def detail(record):
            result = {key: record.get(key) for key in ('id', 'status', 'cv', 'cv_std', 'folds',
                'train_cv', 'seconds', 'spec', 'top_features', 'error', 'analysis',
                'timeout_scope', 'progress', 'retryable', 'eda_evidence') if record.get(key) is not None}
            if record.get('preprocessing_fits'):
                fitted = dict(record['preprocessing_fits'][-1])
                fitted['clip_bounds'] = dict(list(fitted.get('clip_bounds', {}).items())[:16])
                result['last_training_partition_preprocessing'] = fitted
            return result
        profile = self.mem.get('balance.training', {})
        profile = {**profile, 'per_class': sorted(profile.get('per_class', []), key=lambda row: row['count'])[:32]}
        structure = dict(self.mem.get('dataset_interface', {}))
        names = structure.get('feature_columns', [])
        structure['feature_columns'] = names[:200]
        structure['omitted_features'] = max(0, len(names)-200)
        structure['dtypes'] = {key: value for key, value in structure.get('dtypes', {}).items() if key in set(names[:200])}
        structure['dictionary'] = [{key: str(value)[:160] for key, value in row.items()}
            for row in structure.get('dictionary', [])[:24]]
        return json.dumps({'BUDGET': {'total_run_seconds_remaining': max(0, int(self.budget.deadline-time.monotonic())),
            'search_seconds_available': max(0, int(self.left())), 'final_reserve_seconds': round(self.reserve()),
            'final_reserve_already_excluded': True}, 'TRAIN_CLASS_BALANCE': profile,
            'EDA': {'structure': structure, 'audit': self.mem.get('audit.dataset_profile'),
                    'feature_profile': compact_profile(self.mem.get('audit.feature_profile')),
                    'distribution_shift': self.mem.get('audit.adversarial'), 'validation': self.mem.get('validation')},
            'SEARCH_PROTOCOL': self.coverage()[0],
            'HISTORICAL_REFERENCE': {key: value for key, value in (self.mem.get('search.previous_winner') or {}).items() if key != 'spec'},
            'CONFUSION_FEATURE_EVIDENCE': recent.get('confusion_feature_evidence'),
            'RECENT_RUNTIME_LIMITS': [{key: record.get(key) for key in ('id', 'summary', 'timeout_scope', 'seconds', 'progress', 'error')}
                for record in self.mem.state['experiments'] if record.get('status') == 'timeout'][-3:],
            'RECENT_FAILED_EXPERIMENTS': [detail(record) for record in self.mem.state['experiments'] if record.get('status') == 'failed'][-3:],
            'CURRENT': detail(recent), 'BEST': detail(best) if best else None,
            'CURRENT_DIAGNOSTICS': compact_monitor(recent.get('balance_monitor')),
            'BEST_DIAGNOSTICS': compact_monitor(best.get('balance_monitor')) if best else None,
            'ALL_EXPERIMENTS': [{key: record.get(key) for key in ('id', 'summary', 'cv', 'status', 'verdict')}
                for record in self.mem.state['experiments']],
            'INSIGHTS': [item['text'] for item in self.mem.state['insights'][-10:]],
            'BACKLOG': [item['idea'] for item in self.mem.state['backlog'] if item['status'] == 'open'][-8:]},
            ensure_ascii=False)

    def balance_proposal(self):
        best = self.mem.best()
        if best and best['spec']['model']['model_id'] == 'prior':return None
        if not best or not best.get('balance_monitor', {}).get('soft_weights_worth_testing'):
            return None
        if any(record['spec']['training'].get('class_weight_power', 0) > 0 for record in self.mem.state['experiments']):
            return None
        if self.left() < max(15., best['seconds']*1.25):
            return None
        value = json.loads(json.dumps(best['spec']))
        value['model']['params'].pop('class_weight', None)
        value['model']['params'].pop('auto_class_weights', None)
        value['training'].update(class_weight_power=.5, class_weight_clip=20.)
        spec = ExperimentSpec.model_validate(value)
        if self.validate(spec):return None
        return Proposal(hypothesis='Soft clipped class weights may improve the primary metric for underpredicted minority classes',
            why='Measured imbalance AND poor minority recall/prediction counts triggered one controlled weighting trial; full balanced weights are not assumed better.',
            parent=best['id'], spec=spec)

    def reserve(self):
        best = sorted(self.mem.ok_experiments(), key=lambda e: e['cv'], reverse=self.mem.direction == 'maximize')[:4]
        folds = max(1, len(self.context['fold_weights']))
        requested = min(180., self.budget.total_s*.15)+sum(1.7*e['seconds']/folds+10 for e in best)
        return min(self.budget.total_s*.45, max(self.budget.total_s*.12, requested))

    def left(self):
        return self.budget.deadline-time.monotonic()-self.reserve()

    def estimate_seconds(self, spec):
        records = self.mem.ok_experiments()
        same = [record['seconds'] for record in records
            if record['spec']['model']['model_id'] == spec.model.model_id
            and set(record['spec']['blocks']) == set(spec.blocks)
            and bool(record['spec'].get('feature_code')) == bool(spec.feature_code)]
        if not same:
            same = [record['seconds'] for record in records if record['spec']['model']['model_id'] == spec.model.model_id]
        if not same:
            same = [record['seconds'] for record in records]
        estimate = (float(np.median(same[-5:])) if same else 60.)*1.25+3.
        if spec.model.model_id == 'prior':return 8.
        if spec.training.max_boost_rounds < 2000:
            iterations = [np.mean(record.get('best_iters') or [2000]) for record in records
                if record['spec']['model']['model_id'] == spec.model.model_id]
            reference = float(np.median(iterations)) if iterations else 2000.
            estimate *= max(.2, min(1., spec.training.max_boost_rounds/max(1., reference)))
        return max(15., estimate)

    def recovery_proposal(self):
        records = self.mem.state['experiments']
        if not records or records[-1].get('status') != 'timeout':return None
        attempt = sum(record.get('source') == 'timeout_recovery' for record in records)
        if attempt >= 3 or (attempt >= 2 and self.mem.best() is not None):return None
        variants = [
            {'model': {'model_id': 'lightgbm', 'params': {'num_leaves': 15, 'min_child_samples': 30}},
                'training': {'max_boost_rounds': 128}},
            {'model': {'model_id': 'extra_trees', 'params': {'n_estimators': 60, 'max_depth': 12, 'max_features': .6}}},
            {'model': {'model_id': 'prior'}}]
        for index in range(attempt, len(variants)):
            if index == 2 and self.mem.best() is not None:continue
            spec = ExperimentSpec.model_validate(variants[index])
            if self.validate(spec):continue
            return Proposal(hypothesis='Recover from a timed-out experiment using a bounded local ML workload',
                why='Previous experiment exceeded a runtime limit; preserve all successful models and use full unchanged validation folds.',
                parent=records[-1]['id'], spec=spec)
        return None

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
        for record in self.mem.ok_experiments():
            original = ExperimentSpec.model_validate(record['spec'])
            old, new = original.model_dump(), spec.model_dump()
            old['training'].pop('max_boost_rounds', None);new['training'].pop('max_boost_rounds', None)
            if old == new and original.training.max_boost_rounds != spec.training.max_boost_rounds:
                measured = record.get('best_iters') or []
                if measured and all(value is not None for value in measured) and min(original.training.max_boost_rounds, spec.training.max_boost_rounds) > max(measured)*1.2:
                    errors.append('Changing a non-binding iteration ceiling above all measured early-stop iterations is not a useful new experiment')
                    break
        if spec.feature_code and spec.feature_code not in self.custom_codes and len(self.custom_codes) >= 6:
            errors.append('Maximum six distinct generated feature functions per run')
        return errors

    def run(self, spec, hypothesis, why, parent, source, eda_evidence=None, protocol_id=None):
        errors = self.validate(spec)
        if errors:
            raise ValueError('; '.join(errors))
        identifier = f'exp_{len(self.mem.state["experiments"])+1:03d}'
        previous = self.mem.best()
        available = max(.1, self.left())
        context = {**self.context, 'mode': 'cv',
            'timeout_scope': 'experiment' if self.settings.experiment_timeout_s < available else 'search_budget'}
        self.mem.state['active_experiment'] = {'id': identifier, 'spec': spec.model_dump()}
        self.mem.log('experiment_started', id=identifier, spec=spec.model_dump(), source=source)
        self.mem.save()
        seconds = min(self.settings.experiment_timeout_s, max(.1, self.left()))
        if protocol_id == 'model_family':
            cap = max(30., min(180., self.budget.total_s*.15))
            if cap < seconds:context['timeout_scope'] = 'protocol_probe'
            seconds = min(seconds, cap)
        result = run_in_subprocess(spec, context, self.root/'experiments'/identifier, seconds)
        record = {'id': identifier, 'parent': parent, 'source': source, 'hypothesis': hypothesis,
                  'why': why, 'protocol_id': protocol_id, 'eda_evidence': eda_evidence or [], 'spec': spec.model_dump(), 'spec_hash': spec.key(), 'summary': spec.summary(), **result}
        if spec.feature_code:self.custom_codes.add(spec.feature_code)
        if result.get('status') == 'ok':
            oof = np.load(self.root/'experiments'/identifier/'oof.npy', allow_pickle=False)
            valid = np.isfinite(oof).all(axis=1)
            labels = np.load(self.context['labels_path'], allow_pickle=False)
            record['balance_monitor'] = prediction_monitor(labels[valid], oof[valid], self.context['classes'],
                self.context['metric'], self.mem.get('balance.training'), spec.training.threshold, self.context['positive_index'])
            self.mem.fact('balance.latest', {'experiment_id': identifier, **record['balance_monitor']})
            if self._diagnostic_sample is None:
                from ar_validation import load_folds
                import pandas as pd
                source_rows = load_folds(self.root/'folds.npz')[0]['fit']
                source_rows = np.random.default_rng(42).choice(source_rows, min(5000, len(source_rows)), replace=False)
                frame = pd.read_parquet(self.context['features_path']).iloc[source_rows]
                frame = frame.drop(columns=self.context['id_cols'], errors='ignore').reset_index(drop=True)
                self._diagnostic_sample = frame, labels[source_rows]
            record['confusion_feature_evidence'] = confusion_evidence(*self._diagnostic_sample,
                self.context['classes'], record['balance_monitor']['top_confusions'])
            (self.root/'experiments'/identifier/'balance.json').write_text(
                json.dumps(record['balance_monitor'], ensure_ascii=False, indent=2), encoding='utf-8')
            sign = 1 if self.mem.direction == 'minimize' else -1
            gain = sign*(previous['cv']-result['cv'])/max(abs(previous['cv']), 1e-12) if previous else None
            record['delta_best_pct'] = 100*gain if gain is not None else None
            record['verdict'] = 'baseline' if previous is None else 'new_best' if gain >= .002 else 'no_gain' if abs(gain) < .002 else 'worse'
        else:
            record['verdict'] = 'time_limit' if result.get('status') == 'timeout' else 'failed'
        self.mem.add_experiment(record)
        self.mem.fact('search.protocol', self.coverage()[0])
        self.mem.state['active_experiment'] = None
        self.mem.save()
        return record

    def heuristic(self):
        best = self.mem.best()
        if not best:return None
        original = best['spec']
        variants = []
        profile = self.mem.get('audit.feature_profile', {})
        if profile.get('constants') or profile.get('duplicate_columns'):
            variants.append({**original, 'preprocessing': {**original.get('preprocessing', {}),
                'drop_constants': True, 'drop_duplicates': True}})
        if profile.get('correlated_pairs'):
            variants.append({**original, 'preprocessing': {**original.get('preprocessing', {}), 'correlation_threshold': .995}})
        if any((row.get('skew') or 0) > 3 for row in profile.get('numeric', [])):
            variants.append({**original, 'preprocessing': {**original.get('preprocessing', {}), 'clip_quantile': .01, 'outlier_flags': True}})
        cats = [row['feature'] for row in profile.get('categorical', []) if row.get('unique', 0) > 2][:4]
        if cats:
            variants.append({**original, 'preprocessing': {**original.get('preprocessing', {}), 'frequency_columns': cats}})
        if any(row.get('rare_mass_below_1pct', 0) > .02 for row in profile.get('categorical', [])):
            variants.append({**original, 'preprocessing': {**original.get('preprocessing', {}), 'rare_category_min_fraction': .005}})
        for block in ('entity_history', 'group_history', 'similar_history', 'missing', 'families', 'log_numeric', 'oof_target_encoding'):
            if block in self.features.available() and block not in original['blocks']:
                variants.append({**original, 'blocks': [*original['blocks'], block]})
        for model_id in PARAM_SPACE:
            if model_id == 'prior':continue
            for balance in (None, 'balanced'):
                parameters = {}
                if balance:
                    parameters['auto_class_weights' if model_id == 'catboost' else 'class_weight'] = 'Balanced' if model_id == 'catboost' else balance
                variants.append({**original, 'model': {'model_id': model_id, 'loss': 'cross_entropy', 'params': parameters},
                    'training': {**original['training'], 'class_weight_power': 0. if balance else original['training'].get('class_weight_power', 0.)}})
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
        proposal = lab.recovery_proposal()
        recovering = proposal is not None
        if proposal is None:proposal = lab.balance_proposal()
        monitored_proposal = proposal is not None and not recovering
        historical = False
        if proposal is None:
            proposal = lab.history_proposal();historical = proposal is not None
        protocol_id = None
        if proposal is None:
            required = lab.protocol_proposal()
            if required is not None:protocol_id, proposal = required
        if llm.enabled and rejected < 3 and proposal is None:
            last = lab.mem.state['experiments'][-1]['id']
            def validate(step):
                errors = []
                if step.analysis.experiment_id != last:errors.append('Analyze the LAST experiment id')
                if step.decision not in {'experiment', 'stop'}:errors.append('decision must be experiment or stop')
                if step.decision == 'stop':
                    if len(completed) < 4:errors.append('Complete at least four useful experiments before voluntary stop')
                    coverage, _ = lab.coverage()
                    if coverage['enabled'] and any(item['status'] == 'pending' for item in coverage['trials']):
                        errors.append('Required EDA/model coverage is still pending; do not stop before the protocol is completed or explicitly deferred by the coordinator')
                elif step.proposal is None:errors.append('experiment requires a proposal')
                else:
                    errors += lab.validate(step.proposal.spec)
                    if (step.proposal.spec.preprocessing != FeaturePolicy() or step.proposal.spec.feature_code) and not step.proposal.eda_evidence:
                        errors.append('Cite observed EDA/diagnostic facts in proposal.eda_evidence for feature changes')
                    if step.proposal.parent and step.proposal.parent not in {e['id'] for e in lab.mem.state['experiments']}:
                        errors.append('Unknown parent experiment')
                return errors
            catalog = {'blocks': lab.features.available(), 'classifiers': PARAM_SPACE, 'defaults': DEFAULTS,
                       'classes': lab.context['classes'], 'primary_metric': lab.context['metric']}
            memory = lab.research_memory()
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
        source = 'timeout_recovery' if recovering else 'balance_monitor' if monitored_proposal else 'history_recheck' if historical else 'eda_protocol' if protocol_id else 'qwen' if proposal is not None else 'heuristic'
        proposal = proposal or lab.heuristic()
        if proposal is None:return 'ideas_exhausted'
        estimate = lab.estimate_seconds(proposal.spec)
        if lab.left() < estimate:
            lab.mem.log('experiment_skipped', reason='insufficient_measured_research_budget',
                estimated_seconds=estimate, search_seconds_left=max(0., lab.left()), spec=proposal.spec.model_dump())
            lab.mem.save()
            log(f'Stop search: next experiment needs about {estimate:.0f}s; available {max(0., lab.left()):.0f}s after final reserve')
            return 'time_budget'
        record = lab.run(proposal.spec, proposal.hypothesis, proposal.why, proposal.parent, source,
            eda_evidence=proposal.eda_evidence, protocol_id=protocol_id)
        if record.get('status') == 'ok' and len(lab.mem.ok_experiments()) == 1:
            on_baseline(record)  # First successful recovery also gets a full-data safety submission.
        log(f'{record["id"]} [{source}] {record["summary"]} cv={record.get("cv")} {record["verdict"]} {record.get("error", "")}')
    return 'max_experiments'
