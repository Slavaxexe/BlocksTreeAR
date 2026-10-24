"""Sequential, recorded regression decisions passed to each following Qwen role."""
import json

from max_contracts import RegressionSplit, RegressionValidation, RegressionRegularization
from max_storage import save


async def validation_steps(agents, root, context, preparation):
    facts = {**context, 'measured_profile': preparation.profile,
             'provisional_validation': preparation.validation}
    split = await agents.ask(RegressionSplit,
        'Regression decision 1: choose training/validation size and random/group/time split using task '
        'requirements and measured profile. No model has been compared yet. A date column alone does '
        'not imply forecasting. Do not invent observed distributions. These choices will be implemented '
        'by a final preparation pass, then frozen for all experiments.\n' + json.dumps(facts, ensure_ascii=False),
        'regression_split')
    columns = preparation.profile.get('train', {}).get('columns', {})
    if split.strategy == 'group' and (not split.group_columns or set(split.group_columns) - set(columns)):
        raise ValueError('Group split requires actual measured group columns')
    if split.strategy == 'time' and split.time_column not in columns:
        raise ValueError('Time split requires an actual measured time column')
    steps = [{'step': 'split_size', **split.model_dump()}]
    save(root / 'regression_steps.json', {'steps': steps})
    validation = await agents.ask(RegressionValidation,
        'Regression decision 2: based on step 1 choose holdout, internal cross-validation or engine-managed '
        'validation for training/hyperparameter search. Outer validation is the frozen scoring holdout; '
        'all internal folds must remain within fit. Respect groups/time when needed. FLAML supports '
        'holdout/CV, AutoGluon and Optuna adapters use holdout. Do not request unsupported CV from them; '
        'a custom generated model can implement extra folds.\n' +
        json.dumps({**facts, 'previous_steps': steps}, ensure_ascii=False), 'regression_validation')
    steps.append({'step': 'inner_validation', **validation.model_dump()})
    save(root / 'regression_steps.json', {'steps': steps})
    return steps


async def regularization_step(agents, root, shared):
    decision = await agents.ask(RegressionRegularization,
        'Regression decision 4: given previous split, validation and model choices, describe regularization '
        'and bounded hyperparameter search. AutoML chooses actual strengths internally; Optuna adapters '
        'use their built-in search ranges. Do not claim a suggested value was already applied. '
        'Any specific change outside those adapters must become a generated experiment. '
        'SGD, Adam, AdaGrad, momentum/Nesterov are only relevant for compatible differentiable models, '
        'not tree-boosting optimizers: explicitly set optimizer=not_applicable for tree models. '
        'Record the intended hyperparameter_search and parameters_to_tune as well as regularization policy. '
        'Decide what is justified by the available evidence and budget.\n' +
        json.dumps(shared, ensure_ascii=False), 'regression_regularization')
    steps = [*shared.get('regression_steps', []),
             {'step': 'training_approaches', **shared['training_selection']},
             {'step': 'regularization_and_search', **decision.model_dump()}]
    shared['regression_steps'] = steps
    save(root / 'regression_steps.json', {'steps': steps,
         'next_step': 'Analyzer reviews actual experiments and proposes justified parameter changes'})
