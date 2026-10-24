"""Classification interface and conservative dispatch; regression stays in ar_* unchanged."""
from dataclasses import dataclass, field
from pathlib import Path
import re

import pandas as pd


METRICS = {
    'f1_macro': r'macro[\s_-]*f1|f1[\s_-]*macro|макро[\s_-]*f1',
    'f1_weighted': r'weighted[\s_-]*f1|f1[\s_-]*weighted|взвешенн\w*\s+f1',
    'f1_micro': r'micro[\s_-]*f1|f1[\s_-]*micro|микро[\s_-]*f1',
    'balanced_accuracy': r'balanced[\s_-]*accuracy|сбалансированн\w*\s+точност',
    'roc_auc': r'roc[\s_-]*auc|\bauc[\s_-]*roc\b',
    'log_loss': r'log[\s_-]*loss|cross[\s_-]*entropy|логарифмическ\w*\s+потер',
    'accuracy': r'\baccuracy\b',
    'f1': r'\bf1(?:[\s_-]*score)?\b',
}


def task_text(path):
    path = Path(path).resolve()
    path = path/'task.md' if path.is_dir() else path
    return path, path.read_text(encoding='utf-8-sig')


def find_files(path, text):
    root = path.parent
    result = {}
    names = ('train', 'test', 'sample_submission', 'data_dictionary')
    for role in names:
        candidates = sorted(root.rglob(role+'.csv'), key=lambda p: (len(p.parts), str(p)))
        candidates = [p for p in candidates if not any(x in {'artifacts', '.git', '.venv'} for x in p.relative_to(root).parts)]
        if candidates:
            result[role] = str(candidates[0].resolve())
    # Explicit paths may use names such as train_data.csv.
    for value in re.findall(r'[\w./\\-]+\.csv', text):
        candidate = (root/value).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            for role in names:
                if role in candidate.stem.lower() and role not in result:
                    result[role] = str(candidate)
    if not {'train', 'test'}.issubset(result):
        raise ValueError('Task must provide existing train/test CSV files')
    return result


def detect_task_type(path):
    path, text = task_text(path)
    # The actual task section wins over explanatory comparisons of metrics.
    opening = text[:2200].lower()
    classification = re.search(r'классификац|classif(?:ication|ier)|бинарн\w*\s+класс', opening)
    regression = re.search(r'регресси|regression|regressor', opening)
    if classification and not regression:
        return 'classification'
    if regression and not classification:
        return 'regression'
    if any(re.search(pattern, text, re.I) for pattern in METRICS.values()):
        return 'classification'
    if re.search(r'\b(?:rmse|rmsle|mse|mae|mape|smape)\b|R²|R\^2', text, re.I):
        return 'regression'
    files = find_files(path, text)
    train = pd.read_csv(files['train'], nrows=200)
    test = pd.read_csv(files['test'], nrows=1)
    targets = [name for name in train if name not in test]
    if len(targets) == 1 and not pd.api.types.is_numeric_dtype(train[targets[0]]):
        return 'classification'
    # Numeric cardinality alone cannot distinguish integer regression from labels.
    raise ValueError('Task type is ambiguous: state classification/regression and its primary metric in task.md')


def primary_metric(text):
    # Read an explicit metric section before searching prose that may mention alternatives.
    sections = re.split(r'(?m)^#{1,6}\s+', text)
    scopes = [line for line in text.splitlines() if re.search(r'основн\w*\s+метрик|primary\s+metric|scoring\s+metric', line, re.I)]
    scopes += [part for part in sections if re.match(r'(?:метрик|metric|оценк|scoring)', part, re.I)]
    scopes += [text]
    for scope in scopes:
        matches = []
        for name, pattern in METRICS.items():
            match = re.search(pattern, scope, re.I)
            if match:
                matches.append((match.start(), -len(match.group(0)), name, match.group(0)))
        if matches:
            _, _, name, quote = min(matches)
            return name, quote
    raise ValueError('A supported classification primary metric must be explicitly specified; accuracy is not assumed')


@dataclass
class Task:
    task_md: str
    task_dir: str
    files: dict
    target: str
    id_cols: list
    metric: str
    metric_quote: str
    prediction_columns: list
    prediction_type: str = 'labels'
    positive_label: str | None = None
    description: str = ''
    dictionary: list = field(default_factory=list)
    task_type: str = 'classification'


def read_task(path):
    path, text = task_text(path)
    files = find_files(path, text)
    train = pd.read_csv(files['train'], nrows=2)
    test = pd.read_csv(files['test'], nrows=2)
    targets = [name for name in train if name not in test]
    if len(targets) != 1:
        raise ValueError(f'Classification requires one target absent from test, found {targets}')
    target = targets[0]
    dictionary = pd.read_csv(files['data_dictionary']).fillna('').to_dict('records') if files.get('data_dictionary') else []
    if files.get('sample_submission'):
        sample = list(pd.read_csv(files['sample_submission'], nrows=1).columns)
        ids = [name for name in sample if name in test]
        predictions = [name for name in sample if name not in ids]
    else:
        ids = [str(row.get('feature')) for row in dictionary if str(row.get('role', '')).lower() == 'id']
        ids = ids or [name for name in test if name.lower() == 'id']
        predictions = [target]
    if not predictions:
        raise ValueError('Submission must have a prediction column')
    metric, quote = primary_metric(text)
    requested_probability = bool(re.search(r'(?:submission|output|csv|сохран\w*|выда\w*)[^\n]{0,100}(?:вероятност|probabilit)', text, re.I))
    prediction_type = 'probability' if len(predictions) > 1 or metric in {'roc_auc', 'log_loss'} or requested_probability else 'labels'
    # Explicit class-label output overrides mentions of probability-based diagnostics.
    if len(predictions) == 1 and re.search(r'(?:сохран|выда|предска|submit|output)[^\n]{0,100}(?:метк\w*\s+класс|class\s+labels)', text, re.I):
        prediction_type = 'labels'
    positive = re.search(r'(?:positive\s+(?:class|label)|положительн\w*\s+класс)\s*[:=—-]?\s*[`\"\']?([^\s`\"\',;]+)', text, re.I)
    return Task(str(path), str(path.parent), files, target, ids, metric, quote, predictions,
                prediction_type, positive.group(1) if positive else None, text, dictionary)
