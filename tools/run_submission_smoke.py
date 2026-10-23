"""One isolated platform-style acceptance run, using the real configured Qwen API."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('task', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from agent.config import Settings
    settings = Settings.from_env()
    settings.validate()
    submission, task, output = args.submission.resolve(), args.task.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(submission.iterdir())
    if not files or any(not path.is_file() or (path.suffix != '.py' and path.name != 'requirements.txt') for path in files):
        raise ValueError('Upload folder must contain only root Python files and requirements.txt')
    for path in files:
        shutil.copyfile(path, output/path.name)
    for name in ('train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
        source = task.parent/name
        if source.is_file():
            try:
                os.link(source, output/name)
            except OSError:
                shutil.copyfile(source, output/name)
    shutil.copyfile(task, output/'task.md')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(OPENAI_API_KEY=settings.api_key, OPENAI_BASE_URL=settings.base_url,
               OPENAI_MODEL=settings.model, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if settings.timeout_seconds is not None:
        env['QWEN_TIMEOUT_SECONDS'] = str(settings.timeout_seconds)
    if settings.max_tokens is not None:
        env['QWEN_MAX_TOKENS'] = str(settings.max_tokens)
    started = time.monotonic()
    with (output/'platform_execution.log').open('wb') as stream:
        process = subprocess.run([sys.executable, 'solution.py', 'task.md'], cwd=output, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, timeout=4860)
    reports = sorted((output/'artifacts').glob('platform-*/platform_report.json'))
    report = json.loads(reports[-1].read_text(encoding='utf-8')) if reports else {}
    checks = {'exit_code': process.returncode, 'status': report.get('status'),
              'elapsed_seconds': time.monotonic()-started, 'run_directory': str(output),
              'platform_report': str(reports[-1]) if reports else None,
              'real_qwen_configured': True, 'credentials_bundled': False}
    if process.returncode == 0 and report.get('status') == 'completed':
        import numpy as np
        import pandas as pd
        frame = pd.read_csv(output/'submission.csv', dtype=str)
        contract_path = reports[-1].parent/'task_contract.json'
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        test = pd.read_csv(output/'test.csv', dtype={key: str for key in contract['id_columns']})
        ids = contract['id_columns']
        classification = contract['task_type'] == 'classification'
        if classification and contract.get('prediction_type', 'labels') == 'labels':
            labels = pd.read_csv(output/'train.csv', usecols=[contract['target']], dtype=str)[contract['target']].dropna()
            valid_predictions = bool(frame[contract['prediction_column']].notna().all()
                                     and set(frame[contract['prediction_column']]).issubset(set(labels)))
            checks['class_labels_valid'] = valid_predictions
        else:
            values = pd.to_numeric(frame[contract['prediction_column']]).to_numpy()
            valid_predictions = bool(np.isfinite(values).all())
            if classification:
                valid_predictions = valid_predictions and bool(((values >= 0) & (values <= 1)).all())
            checks['finite_predictions'] = valid_predictions
        training = json.loads((reports[-1].parent/'pipeline/training/report.json').read_text(encoding='utf-8'))
        checks.update(task_type=contract['task_type'], primary_metric=contract['primary_metric'],
                      successful_trials=sum(trial['status'] == 'completed' for trial in training['trials']),
                      failed_trials=sum(trial['status'] == 'failed' for trial in training['trials']),
                      role_failures=sum(role['status'] == 'failed' for role in training['role_history']),
                      submission_rows=len(frame), expected_test_rows=len(test),
                      columns=list(frame.columns), full_training_rows=report.get('full_training_rows'),
                      winner=report.get('winner'), search_stop_reason=report.get('search_stop_reason'),
                      identifiers_match=all(frame[key].tolist() == test[key].astype(str).tolist() for key in ids))
        checks['passed'] = (len(frame) == len(test) and valid_predictions and checks['identifiers_match']
                            and list(frame.columns) == ids+[contract['prediction_column']])
    else:
        checks.update(passed=False, error=report.get('error'))
    (output/'acceptance.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks['passed'] else 2


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
