"""Reviewed execution/analysis lessons, selected locally; no LLM or archive lookup."""
import json


RULES = {
    'metric_units': {
        'roles': {'data_reviewer', 'analyzer'}, 'signals': ('improvement', 'comparisons'),
        'errors': ('relative_improvement',),
        'text': 'improvement/absolute_improvement is a metric difference, not percent. Use improvement_percent when supplied. A single holdout does not establish statistical significance or an optimal configuration.'},
    'entity_denominators': {
        'roles': {'data_researcher', 'data_reviewer', 'baker'}, 'signals': ('entities', 'key_candidates', 'history'),
        'errors': ('key_candidates',),
        'text': 'repeated_fit_row_fraction measures repetition across ALL periods, NOT key conflation: cross-period repeats can enable history. within_period_ambiguity_fraction measures duplicate (key,time) source rows; comparison_strict_past_coverage measures earlier-history availability in query rows. These denominators differ. A failed specific key does not rule out simpler keys.'},
    'shift_evidence': {
        'roles': {'data_reviewer', 'analyzer', 'baker'}, 'signals': ('shift', 'ks_distance', 'total_variation'),
        'errors': (),
        'text': 'Distribution shift alone does not prove feature harm or a test-score ceiling. Compare actual matched model ablations; marginal MI is not proof of useful or useless conditional signal.'},
    'ablation_inputs': {
        'roles': {'data_researcher', 'configuration_repair'}, 'signals': ('ablation',),
        'errors': ('every model input', 'at least one input feature'),
        'text': 'baseline_columns is the original input set; ablation columns lists ONLY removed inputs. Preserve shared recipes and retain at least one model input. baseline_features preserves existing encodings when testing a new recipe.'},
    'group_statistics': {
        'roles': {'data_researcher'}, 'signals': ('groups', 'relationships', 'group_mean'),
        'errors': ('groups needs', 'group_mean needs'),
        'text': 'groups requires an explicit non-target group_by. group_mean takes exactly [group_column, numeric_column]; use history or target_encoding for supervised grouping, not group_mean on the target.'},
    'question_identity': {
        'roles': {'data_reviewer'}, 'signals': ('open_questions',),
        'errors': ('resolved questions must',),
        'text': 'resolves contains exact supplied open_questions strings, not paraphrases. Leave it empty if none were answered. Unsupported measurements are not evidence.'},
    'temporal_smoke': {
        'roles': {'configuration_repair', 'analyzer'}, 'signals': ('distinct periods', 'smoke'),
        'errors': ('temporal holdout needs',),
        'text': 'A smoke-only lack of periods can be caused by sampling, not the full dataset or hypothesis. Inspect measured smoke period counts. Keep the declared validation policy instead of silently switching to random.'},
    'catboost_class': {
        'roles': {'node_planner', 'configuration_repair', 'analyzer', 'baker'}, 'signals': ('catboost',),
        'errors': ('catboost.__init__',),
        'text': 'Use catboost.CatBoostRegressor or CatBoostClassifier for sklearn-style constructor kwargs. Generic CatBoost uses a params dictionary; repairing iterations alone does not fix selection of the wrong class.'},
}


def observed_rules(error):
    text = str(error).lower()
    return [key for key, rule in RULES.items() if any(signal in text for signal in rule['errors'])]


def augment_prompt(prompt, role, observations):
    try:
        instructions, raw = prompt.rsplit('\n', 1)
        context = json.loads(raw)
    except (ValueError, TypeError):
        return prompt, []
    if not isinstance(context, dict):
        return prompt, []
    # Catalogs mention every possible operation: they must not trigger unrelated guidance.
    focus = {key: context[key] for key in ('idea', 'hypothesis', 'previous_operation', 'execution_error',
        'experiment', 'measurements', 'comparisons', 'plan', 'open_questions') if key in context}
    focus['direction'] = context.get('search', {}).get('direction')
    questions = context.get('data_research', {}).get('open_questions')
    if questions:
        focus['open_questions'] = questions
    text = json.dumps(focus, ensure_ascii=False).lower()
    selected = [key for key, rule in RULES.items() if role in rule['roles']
                and any(signal in text for signal in rule['signals'])]
    selected.sort(key=lambda key: observations.get(key, 0), reverse=True)
    selected = selected[:3]
    if selected:
        context['execution_guidance'] = [{'id': key, 'rule': RULES[key]['text']} for key in selected]
        prompt = instructions + '\n' + json.dumps(context, ensure_ascii=False, separators=(',', ':'))
    return prompt, selected


def prompt_sizes(messages):
    """Character attribution for diagnostics; these counts are not tokenizer estimates."""
    system = sum(len(message.get('content') or '') for message in messages if message['role'] == 'system')
    user = next((message.get('content', '') for message in messages if message['role'] == 'user'), '')
    continuation = len(json.dumps(messages[2:], ensure_ascii=False, separators=(',', ':'))) if len(messages) > 2 else 0
    try:
        instructions, raw = user.rsplit('\n', 1)
        context = json.loads(raw)
    except (ValueError, TypeError):
        return {'system_schema': system, 'instructions': len(user), 'continuation': continuation}
    return {'system_schema': system, 'instructions': len(instructions),
            'continuation': continuation,
            'context_fields': {key: len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
                               for key, value in context.items()} if isinstance(context, dict) else {}}
