"""Run the original external model-selection stage with our Qwen HTTP adapter."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

from agents import JSONAgents
from config import Settings
from model_selection_bridge import build_stage_input, build_training_selection
from storage import save


class SelectionLLMAdapter:
    """Preserve the external system/user/validator transcript and JSON decision schema."""
    def __init__(self, output, deadline, schema):
        self.schema = schema
        self.agents = JSONAgents(Settings.from_env(), deadline, max_calls=6)
        self.agents.root.log_directory = output

    def __call__(self, messages):
        prompt = (
            'Execute the regression model-family and loss selection conversation below. '
            'Its system message contains the selection methodology; its user messages contain '
            'the factual data and may contain the external validator feedback for a retry. '
            'Follow the methodology and return the full LLMDecision JSON. '
            'Retain exact FACTS values in evidence. Never invent probe measurements.\n'
            + json.dumps({'selection_conversation': messages}, ensure_ascii=False, allow_nan=False)
        )
        result = asyncio.run(self.agents.ask(self.schema, prompt, 'model_selection', thinking=False))
        return result.model_dump_json()


def register_selection(source_root):
    source_root = Path(source_root).resolve()
    package = source_root/'model_selection'
    if not (package/'stage.py').is_file():
        raise ValueError('source_root must contain the external model_selection package')
    cached = sys.modules.get('model_selection')
    if cached and Path(cached.__file__).resolve().parent != package:
        raise ValueError('A different model_selection package is already imported')
    sys.dont_write_bytecode = True
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    builders = importlib.import_module('model_selection.builders')
    builders.N_JOBS = 2
    for settings in builders.PROBE_PARAMS.values():
        for key in ('n_jobs', 'thread_count'):
            if key in settings:
                settings[key] = 2
    return importlib.import_module('model_selection.stage'), importlib.import_module('model_selection.schemas')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--time-budget', type=float, required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding='utf-8-sig'))
    output = Path(job['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    adapter = None
    try:
        stage, schemas = register_selection(job['source_root'])
        stage_input = build_stage_input(
            job['preparation_dir'], job['margo_root'], output,
            probe_seconds=job.get('probe_seconds', 90),
            probe_max_rows=job.get('probe_max_rows', 12000), probe_folds=job.get('probe_folds', 2))
        # Validate the actual external contract before invoking its stage, not a duplicated schema.
        stage_input = schemas.StageInput.model_validate(stage_input).model_dump(mode='json')
        save(output/'stage_input.json', stage_input)
        if job.get('use_llm', True):
            try:
                adapter = SelectionLLMAdapter(output, started+args.time_budget, schemas.LLMDecision)
            except Exception as error:
                # Let the external decide() record the error and select its documented rules fallback.
                unavailable_error = f'{type(error).__name__}: {error}'
                def unavailable(messages):
                    raise RuntimeError(unavailable_error)
                llm_fn = unavailable
            else:
                llm_fn = adapter
        else:
            llm_fn = None
        stage_output = stage.run_stage(stage_input, llm_fn=llm_fn, use_llm=job.get('use_llm', True))
        stage_output = schemas.StageOutput.model_validate(stage_output).model_dump(mode='json')
        save(output/'stage_output.json', stage_output)
        provenance = json.loads((output/'selection_input_provenance.json').read_text(encoding='utf-8'))
        selection = build_training_selection(
            stage_output, primary_metric=stage_input['task']['target_metric'],
            target=stage_input['task']['target'], id_columns=provenance['id_columns'],
            submission=job.get('submission'))
        save(output/'training_selection.json', selection)
        save(output/'result.json', {
            'status': 'completed', 'phase': 'model_selection',
            'duration_seconds': time.monotonic()-started,
            'source': stage_output['source'], 'source_root': str(Path(job['source_root']).resolve()),
            'primary_metric': selection['primary_metric'], 'target': stage_input['task']['target'],
            'selected_models': selection['selected_models'],
            'selection_path': str(output/'training_selection.json'),
            'outer_fit_rows': provenance['outer_fit_rows'],
            'probe_rows': stage_output['probe']['rows_used'],
            'probe_results': len(stage_output['probe']['results']),
            'llm_calls': adapter.agents.requests if adapter else 0,
            'warnings': stage_output['warnings'], 'validation_errors': stage_output['validation_errors'],
        })
        return 0
    except Exception as error:
        save(output/'result.json', {
            'status': 'failed', 'phase': 'model_selection',
            'duration_seconds': time.monotonic()-started,
            'error_type': type(error).__name__, 'error': str(error)[-1500:],
            'traceback': traceback.format_exc()[-5000:]})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
