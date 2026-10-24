"""Process entry point; isolated training errors are returned as structured data."""

import argparse
import json
from pathlib import Path
import traceback

from max_training_backend import run_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--time-budget', required=True, type=float)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding='utf-8-sig'))
    output = Path(job['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = run_job(job, args.time_budget)
    except Exception as exc:
        result = {'status': 'failed', 'phase': job.get('phase'), 'candidate': job.get('candidate', {}).get('name'),
                  'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
