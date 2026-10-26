"""Build, verify, export and score the five local practice tasks without Qwen calls."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error, mean_squared_log_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .catalog import CATALOG, FILES, SEED

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'data/practice'
PRIVATE = ROOT / 'evaluation/private'
CACHE = ROOT / 'tmp/practice_sources'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_csv(path):
    return pd.read_csv(path, dtype={'id': str}, keep_default_na=False)


def source_path(spec, offline=False):
    path = CACHE / spec['cache']
    if not path.exists():
        if offline:
            raise ValueError(f'Missing source cache: {path}')
        response = httpx.get(spec['url'], follow_redirects=True, timeout=90)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != spec['sha256']:
            raise ValueError(f'Source checksum changed: {spec["url"]}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    if sha256(path) != spec['sha256']:
        raise ValueError(f'Source checksum mismatch: {path}')
    return path


def load_source(slug, spec, offline=False):
    path = source_path(spec, offline)
    if slug == 'concrete_strength':
        frame = pd.read_csv(path)
        frame.columns = list(spec['fields'])
    elif slug == 'dry_bean':
        frame = pd.read_csv(path).rename(columns={'Class': spec['target']})
    elif slug == 'bank_subscription':
        with zipfile.ZipFile(path) as archive:
            with zipfile.ZipFile(io.BytesIO(archive.read('bank-additional.zip'))) as nested:
                frame = pd.read_csv(io.BytesIO(nested.read('bank-additional/bank-additional-full.csv')), sep=';')
        frame = frame.drop(columns='duration').rename(columns={'y': spec['target']})
        frame[spec['target']] = frame[spec['target']].map({'no': 0, 'yes': 1})
    elif slug == 'bike_demand':
        with zipfile.ZipFile(path) as archive:
            frame = pd.read_csv(io.BytesIO(archive.read('hour.csv')))
        frame.insert(0, 'timestamp', (pd.to_datetime(frame.dteday) + pd.to_timedelta(frame.hr, unit='h'))
                     .dt.strftime('%Y-%m-%d %H:%M:%S'))
        frame = frame.drop(columns=['instant', 'casual', 'registered', 'dteday', 'hr'])
        frame = frame.rename(columns={'cnt': spec['target']}).sort_values('timestamp').reset_index(drop=True)
    else:
        with zipfile.ZipFile(path) as archive:
            text = archive.read('SMSSpamCollection').decode('utf-8')
        # The source format defines physical lines, not CSV quoting rules.
        rows = [line.split('\t', 1) for line in text.splitlines() if line]
        if any(len(row) != 2 for row in rows):
            raise ValueError('Invalid SMS source row')
        frame = pd.DataFrame(rows, columns=[spec['target'], 'message'])[['message', spec['target']]]
    if set(frame) != set(spec['fields']) or frame.isna().any().any():
        raise ValueError(f'Unexpected source schema or missing values: {slug}')
    return frame


def groups_for(slug, frame, target):
    features = frame.drop(columns=[target, 'id'], errors='ignore')
    if slug == 'sms_spam':
        return features.message.str.lower().str.split().str.join(' ')
    if slug == 'concrete_strength':
        features = features.drop(columns='age_days')
    return pd.util.hash_pandas_object(features, index=False).astype(str)


def split_source(slug, frame, spec):
    if slug == 'bike_demand':
        test = frame.timestamp >= '2012-10-01'
        return np.flatnonzero(~test), np.flatnonzero(test)
    groups = groups_for(slug, frame, spec['target'])
    if slug == 'concrete_strength':
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
        return next(splitter.split(frame, groups=groups))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return next(splitter.split(frame, frame[spec['target']], groups))


def task_text(slug, spec, train, test):
    def count(number):
        return f'{number:,}'.replace(',', ' ')

    def quantity(number, singular, few, many):
        if 11 <= number % 100 <= 14:
            word = many
        elif number % 10 == 1:
            word = singular
        elif number % 10 in (2, 3, 4):
            word = few
        else:
            word = many
        return f'{count(number)} {word}'

    features = len(train.columns) - 2
    feature_count = quantity(features, 'входной признак', 'входных признака', 'входных признаков')
    fields = ['| Поле | Тип и роль | Описание и единицы измерения |', '|---|---|---|',
              '| `id` | Строковый идентификатор | Уникальный идентификатор строки |']
    units = {'kg/m3': 'кг/м³', 'days': 'дни', 'years': 'годы', 'MPa': 'МПа',
             'pixels': 'пиксели', 'count': 'количество', 'month': 'месяц', 'count/hour': 'аренд/час'}
    for column in train.columns[1:]:
        unit, description = spec['fields'][column]
        if column == 'message':
            description = 'Текст SMS-сообщения на английском языке'
        if column == spec['target']:
            role = 'Категориальный таргет' if 'labels' in spec else 'Числовой таргет'
        elif unit in ('category', '0/1'):
            role = 'Категориальный признак'
        elif unit == 'text':
            role = 'Текстовый признак'
        elif unit == 'ISO datetime':
            role = 'Дата и время'
        else:
            role = 'Числовой признак'
        if unit in units:
            description += f'; {units[unit]}'
        fields.append(f'| `{column}` | {role} | {description} |')
    field_table = '\n'.join(fields)
    return f'''# {spec['title']}

## Задача

{spec['problem']}

## Данные

Пути указаны относительно каталога этого Markdown-файла:

- `train.csv` — {quantity(len(train), 'строка', 'строки', 'строк')} с признаками и таргетом;
- `test.csv` — {quantity(len(test), 'строка', 'строки', 'строк')} с признаками без таргета;
- `data_dictionary.csv` — описание всех полей;
- `sample_submission.csv` — шаблон ответа; значения в нём не являются правильными ответами.

В `train.csv` {quantity(len(train.columns), 'столбец', 'столбца', 'столбцов')}: {feature_count}, идентификатор `id` и таргет
`{spec['target']}`. В `test.csv` {quantity(len(test.columns), 'столбец', 'столбца', 'столбцов')}: те же признаки и `id`, без таргета.
CSV используют UTF-8, запятую как разделитель и точку как десятичный разделитель.
Текстовые значения, содержащие запятые или кавычки, заключены в кавычки по правилам CSV.

{spec['data_note']}

## Описание полей

{field_table}

Полное описание находится в `data_dictionary.csv`.

## Что предсказать

{spec['prediction']}

{spec['metric_text']}
Ответы теста недоступны.

## Формат результата

Файл `submission.csv` в каталоге этого Markdown-файла: CSV в UTF-8,
разделитель — запятая, без дополнительного индекса, ровно два столбца:

```csv
id,{spec['target']}
```

{spec['result_text']}
'''


def build(slugs, *, offline=False):
    for slug in slugs:
        spec = CATALOG[slug]
        frame = load_source(slug, spec, offline)
        fit, holdout = split_source(slug, frame, spec)
        rng = np.random.default_rng(SEED)
        # IDs must not encode source row order: Dry Bean source is sorted by its target.
        frame.insert(0, 'id', [f'{slug}_{n:06d}' for n in rng.permutation(len(frame))])
        if slug != 'bike_demand':
            fit, holdout = rng.permutation(fit), rng.permutation(holdout)
        train, reference = frame.iloc[fit].reset_index(drop=True), frame.iloc[holdout].reset_index(drop=True)
        target = spec['target']
        test, labels = reference.drop(columns=target), reference[['id', target]]
        public, private = PUBLIC / slug, PRIVATE / slug
        public.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)
        train.to_csv(public / 'train.csv', index=False, lineterminator='\n')
        test.to_csv(public / 'test.csv', index=False, lineterminator='\n')
        labels.to_csv(private / 'test_labels.csv', index=False, lineterminator='\n')
        sample = labels.copy()
        sample[target] = spec.get('labels', [0])[0]
        sample.to_csv(public / 'sample_submission.csv', index=False, lineterminator='\n')
        dictionary = [{'feature': 'id', 'role': 'id', 'dtype': 'string', 'unit': '',
                       'description_ru': 'Случайный идентификатор; не использовать как признак'}]
        for column in frame.columns[1:]:
            unit, description = spec['fields'][column]
            dictionary.append({'feature': column, 'role': 'target' if column == target else 'feature',
                               'dtype': str(frame[column].dtype), 'unit': unit, 'description_ru': description})
        pd.DataFrame(dictionary).to_csv(public / 'data_dictionary.csv', index=False, lineterminator='\n')
        (public / 'task.md').write_text(task_text(slug, spec, train, test), encoding='utf-8')
        manifest = {
            'dataset': slug, 'seed': SEED, 'source_rows': len(frame), 'train_rows': len(train),
            'test_rows': len(test), 'feature_count': len(test.columns) - 1,
            'target': target, 'metric': spec['metric'], 'direction': spec['direction'],
            'source': {key: spec[key] for key in ('source_name', 'page', 'url', 'sha256', 'citation')},
            'license': 'CC BY 4.0', 'adaptation_date': '2026-10-20', 'changes': spec['changes'],
            'split': spec['split'], 'participant_sha256': {name: sha256(public / name) for name in FILES},
            'labels_sha256': sha256(private / 'test_labels.csv'),
        }
        if 'labels' in spec or slug == 'bank_subscription':
            manifest['train_class_counts'] = {str(k): int(v) for k, v in train[target].value_counts().items()}
            manifest['test_class_counts'] = {str(k): int(v) for k, v in labels[target].value_counts().items()}
        write_json(private / 'manifest.json', manifest)
        print(f'Built {slug}: train={len(train)}, test={len(test)}')


def validate_prediction(predictions, reference, spec):
    target = spec['target']
    if list(predictions) != ['id', target]:
        raise ValueError(f'Expected exactly these ordered columns: id,{target}')
    if len(predictions) != len(reference):
        raise ValueError('Prediction row count differs from test')
    if predictions.id.eq('').any() or predictions.id.isna().any() or predictions.id.duplicated().any():
        raise ValueError('Missing or duplicate prediction IDs')
    if predictions.id.tolist() != reference.id.tolist():
        raise ValueError('Prediction IDs or their order differ from test')
    values = predictions[target]
    if 'labels' in spec:
        if not values.isin(spec['labels']).all():
            raise ValueError('Unknown or missing predicted class')
        return values
    try:
        values = pd.to_numeric(values, errors='raise').to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise ValueError('Predictions must be numeric') from None
    if not np.isfinite(values).all():
        raise ValueError('Non-finite predictions')
    if spec['metric'] == 'roc_auc' and ((values < 0) | (values > 1)).any():
        raise ValueError('Probabilities must lie in [0, 1]')
    if spec['metric'] == 'rmsle' and (values < 0).any():
        raise ValueError('RMSLE predictions must be nonnegative')
    return values


def metric_value(truth, predictions, spec):
    metric = spec['metric']
    if metric == 'roc_auc':
        return float(roc_auc_score(truth, predictions))
    if metric == 'rmse':
        return float(np.sqrt(mean_squared_error(truth, predictions)))
    if metric == 'rmsle':
        return float(np.sqrt(mean_squared_log_error(truth, predictions)))
    if metric == 'macro_f1':
        return float(f1_score(truth, predictions, labels=spec['labels'], average='macro', zero_division=0))
    if metric == 'binary_f1':
        return float(f1_score(truth, predictions, pos_label='spam', average='binary', zero_division=0))
    raise ValueError(f'Unsupported metric: {metric}')


def score(slug, submission):
    spec = CATALOG[slug]
    private = PRIVATE / slug
    manifest = json.loads((private / 'manifest.json').read_text(encoding='utf-8'))
    labels_path = private / 'test_labels.csv'
    if sha256(labels_path) != manifest['labels_sha256']:
        raise ValueError('Hidden labels checksum mismatch')
    reference = read_csv(labels_path)
    predictions = read_csv(submission)
    values = validate_prediction(predictions, reference, spec)
    return {'dataset': slug, 'metric': spec['metric'], 'direction': spec['direction'],
            'value': metric_value(reference[spec['target']], values, spec), 'rows': len(reference),
            'submission_sha256': sha256(submission)}


def verify(slugs):
    for slug in slugs:
        spec, public, private = CATALOG[slug], PUBLIC / slug, PRIVATE / slug
        manifest = json.loads((private / 'manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['participant_sha256'].items():
            if sha256(public / name) != expected:
                raise ValueError(f'Participant file changed: {slug}/{name}')
        if sha256(private / 'test_labels.csv') != manifest['labels_sha256']:
            raise ValueError(f'Hidden labels changed: {slug}')
        train, test = read_csv(public / 'train.csv'), read_csv(public / 'test.csv')
        labels, sample = read_csv(private / 'test_labels.csv'), read_csv(public / 'sample_submission.csv')
        target = spec['target']
        if list(train.drop(columns=target)) != list(test) or list(labels) != ['id', target]:
            raise ValueError(f'Invalid train/test schema: {slug}')
        if (len(train), len(test)) != (manifest['train_rows'], manifest['test_rows']):
            raise ValueError(f'Unexpected row counts: {slug}')
        if len(train) + len(test) != manifest['source_rows']:
            raise ValueError(f'Source rows lost: {slug}')
        for table in (train, test):
            if table.id.eq('').any() or table.id.duplicated().any():
                raise ValueError(f'Missing/duplicate IDs: {slug}')
        if set(train.id) & set(test.id) or test.id.tolist() != labels.id.tolist():
            raise ValueError(f'Overlapping or mismatched IDs: {slug}')
        if train[target].eq('').any() or train[target].isna().any():
            raise ValueError(f'Missing training target: {slug}')
        if 'labels' in spec and (set(train[target]) != set(spec['labels']) or set(labels[target]) != set(spec['labels'])):
            raise ValueError(f'Missing or unknown classes: {slug}')
        validate_prediction(sample, labels, spec)
        dictionary = read_csv(public / 'data_dictionary.csv')
        if dictionary.feature.tolist() != list(train) or dictionary.description_ru.eq('').any():
            raise ValueError(f'Incomplete data dictionary: {slug}')
        if slug == 'bike_demand':
            if train.timestamp.max() >= test.timestamp.min() or test.timestamp.min() < '2012-10-01':
                raise ValueError('Temporal leakage')
            if pd.concat([train.timestamp, test.timestamp]).duplicated().any():
                raise ValueError('Duplicate hours')
            if {'casual', 'registered', 'instant'} & set(train):
                raise ValueError('Target proxies in bike inputs')
        else:
            train_groups, test_groups = groups_for(slug, train, target), groups_for(slug, test, target)
            if set(train_groups) & set(test_groups):
                raise ValueError(f'Groups leak across train and test: {slug}')
        if slug == 'bank_subscription' and 'duration' in train:
            raise ValueError('Call duration leakage')
        if set(p.name for p in public.iterdir()) - set(FILES) - {'submission.csv'}:
            raise ValueError(f'Unexpected files in participant folder: {slug}')
        print(f'OK {slug}: schema, IDs, targets, split, dictionary, hashes')


def export(slug, destination):
    """Copy an allowlist to a NEW directory, never evaluation code, labels or raw sources."""
    destination = Path(destination).resolve()
    if destination.is_relative_to(PUBLIC.resolve()) or destination.is_relative_to(PRIVATE.resolve()):
        raise ValueError('Export destination must be outside dataset and evaluator inputs')
    destination.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        shutil.copy2(PUBLIC / slug / name, destination / name)
    return destination


def smoke(slugs, output):
    """Dataset-specific diagnostic models; deliberately separate from the universal agent."""
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from threadpoolctl import threadpool_limits

    report = {'description': 'Diagnostic models, not a Qwen/solution.py run or an official score', 'datasets': []}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for slug in slugs:
        spec = CATALOG[slug]
        train, test = read_csv(PUBLIC / slug / 'train.csv'), read_csv(PUBLIC / slug / 'test.csv')
        target = spec['target']
        x, query = train.drop(columns=['id', target]), test.drop(columns='id')
        if slug == 'sms_spam':
            x, query = x.message, query.message
            model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                  LogisticRegression(C=3, max_iter=500, random_state=SEED))
        elif slug == 'concrete_strength':
            model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=2, random_state=SEED)
        elif slug == 'dry_bean':
            model = RandomForestClassifier(n_estimators=150, class_weight='balanced', n_jobs=2, random_state=SEED)
        elif slug == 'bank_subscription':
            numeric = x.select_dtypes(include='number').columns.tolist()
            categorical = x.select_dtypes(exclude='number').columns.tolist()
            transform = ColumnTransformer([('num', StandardScaler(), numeric),
                                           ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)])
            model = make_pipeline(transform, LogisticRegression(max_iter=700, random_state=SEED))
        else:
            for table in (x, query):
                stamp = pd.to_datetime(table.pop('timestamp'))
                table['hour'] = stamp.dt.hour
                table['day_index'] = (stamp - pd.Timestamp('2011-01-01')).dt.total_seconds() / 86400
            model = TransformedTargetRegressor(
                regressor=HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=23, random_state=SEED),
                func=np.log1p, inverse_func=np.expm1)
        with threadpool_limits(limits=2):
            model.fit(x, train[target])
            values = model.predict_proba(query)[:, list(model.classes_).index(1)] if spec['metric'] == 'roc_auc' else model.predict(query)
        if spec['metric'] == 'rmsle':
            values = np.maximum(values, 0)
        destination = output / f'{slug}.csv'
        pd.DataFrame({'id': test.id, target: values}).to_csv(destination, index=False)
        result = score(slug, destination)
        # A train-only constant is a transparent reference, not the organizer's undisclosed B.
        constant = (train[target].mode().iloc[0] if 'labels' in spec else
                    float(np.expm1(np.log1p(train[target]).mean())) if spec['metric'] == 'rmsle' else
                    float(train[target].mean()))
        reference = read_csv(PRIVATE / slug / 'test_labels.csv')
        result['constant_reference_value'] = metric_value(reference[target], np.repeat(constant, len(test)), spec)
        report['datasets'].append(result)
        print(json.dumps(result, ensure_ascii=False))
    write_json(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('build', 'verify', 'smoke'):
        command = commands.add_parser(name)
        command.add_argument('--dataset', choices=CATALOG)
        if name == 'build':
            command.add_argument('--offline', action='store_true')
        if name == 'smoke':
            command.add_argument('--output', type=Path, required=True, help='New output directory')
    command = commands.add_parser('export')
    command.add_argument('dataset', choices=CATALOG)
    command.add_argument('destination', type=Path)
    command = commands.add_parser('score')
    command.add_argument('dataset', choices=CATALOG)
    command.add_argument('submission', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'score':
            print(json.dumps(score(args.dataset, args.submission), ensure_ascii=False, indent=2))
        elif args.command == 'export':
            print(export(args.dataset, args.destination))
        else:
            slugs = [args.dataset] if args.dataset else list(CATALOG)
            if args.command == 'build':
                build(slugs, offline=args.offline)
            elif args.command == 'verify':
                verify(slugs)
            else:
                smoke(slugs, args.output)
    except (ValueError, OSError, httpx.HTTPError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
