"""Run the untouched external preparation package using our runtime/API adapter."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import traceback

from .agents import JSONAgents
from .config import Settings
from .margo_recipe import register_margo
from .storage import save


class PreparationLLMAdapter:
    def __init__(self, output, deadline, schema, enabled):
        self.schema, self.enabled = schema, enabled
        self.agents = JSONAgents(Settings.from_env(), deadline, max_calls=3) if enabled else None
        if self.agents:
            self.agents.root.log_directory = output

    @property
    def is_available(self):
        return self.enabled and bool(self.agents.settings.api_key and self.agents.settings.base_url)

    @property
    def calls_made(self):
        return self.agents.requests if self.agents else 0

    @property
    def tokens_used(self):
        return sum((call.prompt_tokens or 0)+(call.completion_tokens or 0) for call in self.agents.calls) if self.agents else 0

    def request_structured(self, system_prompt, user_prompt):
        try:
            return asyncio.run(self.agents.ask(self.schema, system_prompt+'\n'+user_prompt, 'margo_preparation'))
        except Exception:
            # The upstream package already supports a deterministic fallback plan.
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--time-budget', type=float, required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding='utf-8-sig'))
    output = Path(job['output_dir'])
    started = time.monotonic()
    try:
        _, schemas = register_margo(job['source'])
        from data_preparation.config import PipelineConfig
        from data_preparation.pipeline import DataPreparationPipeline
        config = PipelineConfig(**job['config'])
        config.output_dir = str(output)
        config.time_budget_seconds = max(1, int(args.time_budget))
        llm = PreparationLLMAdapter(output, started+args.time_budget, schemas.LLMResponse, job['use_llm'])
        result = DataPreparationPipeline(config, llm_client=llm).run()
        if result.status.value != 'success':
            raise ValueError('margo preparation failed: '+json.dumps([item.model_dump() for item in result.errors], ensure_ascii=False))
        report = {'status': 'completed', 'phase': 'preparation', 'duration_seconds': time.monotonic()-started,
                  'output_dir': str(output), 'train_rows': len(result.train_features),
                  'validation_rows': len(result.val_features), 'test_rows': len(result.test_features),
                  'feature_count': len(result.feature_names), 'llm_calls': llm.calls_made,
                  'llm_features_used': result.transformer.plan.llm_suggestions_used,
                  'warnings': [item.model_dump() for item in result.warnings]}
        save(output/'result.json', report)
        return 0
    except Exception as error:
        save(output/'result.json', {'status': 'failed', 'phase': 'preparation', 'error_type': type(error).__name__,
                                   'error': str(error)[-1500:], 'traceback': traceback.format_exc()[-4000:]})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
