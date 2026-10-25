"""Role-specific JSON views. Full provenance stays on disk, outside model prompts."""

import re


def compact(value):
    """Omit empty metadata, preserving false, zero and complete strings."""
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if item not in (None, '', [], {})}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def table(rows):
    """Lossless column-oriented JSON for uniform records; no row selection."""
    if isinstance(rows, dict):
        return {key: table(value) for key, value in rows.items()}
    if not isinstance(rows, list):
        return rows
    if len(rows) >= 3 and all(isinstance(row, dict) and set(row) == set(rows[0]) for row in rows):
        columns = list(rows[0])
        return {'columns': columns, 'rows': [[table(row[key]) for key in columns] for row in rows]}
    return [table(row) for row in rows]


def prompt_schema(schema, *, root=True):
    # Property names such as "title" must survive; only schema annotations are removed.
    if isinstance(schema, dict):
        return {key: ({name: prompt_schema(item, root=False) for name, item in value.items()}
                      if key in {'properties', '$defs'} else prompt_schema(value, root=False))
                for key, value in schema.items() if key != 'title' or root}
    if isinstance(schema, list):
        return [prompt_schema(item, root=False) for item in schema]
    return schema


def operation_view(value):
    from .plans import AutoMLPlan, EnsemblePlan
    value = value.model_dump() if hasattr(value, 'model_dump') else dict(value or {})
    parameters = value.get('parameters', {})
    schema = {'automl': AutoMLPlan, 'saved_ensemble': EnsemblePlan}.get(value.get('kind'))
    if schema is not None:
        parameters = schema.model_validate(parameters).model_dump(exclude_defaults=True, exclude_none=True)
    return compact({**value, 'parameters': parameters,
                    'inputs': {key: name for key, name in value.get('inputs', {}).items() if key != name},
                    'outputs': {key: name for key, name in value.get('outputs', {}).items() if key != name}})


def task_context(context, *, profile=True):
    preparation = context.get('preparation', {})
    keys = ['data', 'metric', 'output', 'validation', 'issues'] + (['profile'] if profile else [])
    return compact({'requirements': context.get('requirements', {}),
                    'preparation': {key: preparation[key] for key in keys if key in preparation},
                    'installed_libraries': context.get('installed_libraries', {})})


def family(card):
    model = card.get('hypothesis', {}).get('pipeline', {}).get('model', {})
    return str(model.get('engine') or model.get('estimator') or model.get('name') or 'unknown')


def select_experiments(cards, leader_id, direction, limit=8):
    """Leader, recent evidence, a fresh failure, diverse best models and ancestry."""
    by_id = {card['node_id']: card for card in cards}
    selected = []
    def add(card):
        if card is not None and card['node_id'] not in selected and len(selected) < limit:
            selected.append(card['node_id'])
    add(by_id.get(leader_id))
    for card in cards[-2:]:
        add(card)
    add(next((card for card in reversed(cards) if card.get('result', {}).get('status') != 'completed'), None))
    ranked = sorted([card for card in cards if card.get('result', {}).get('value') is not None],
                    key=lambda card: card['result']['value'], reverse=direction == 'maximize')
    families = {family(by_id[node]) for node in selected}
    diverse = 0
    for card in ranked:
        if family(card) not in families and diverse < 2:
            add(card)
            families.add(family(card))
            diverse += 1
    parent = by_id.get(leader_id, {}).get('parent_id')
    for _ in range(2):
        add(by_id.get(parent))
        parent = by_id.get(parent, {}).get('parent_id')
    for card in ranked:
        add(card)
    return [card for card in cards if card['node_id'] in selected]


def experiment_index(cards):
    return table([{'node_id': card['node_id'], 'parent_id': card.get('parent_id'), 'model': family(card),
                   'status': card.get('result', {}).get('status'), 'value': card.get('result', {}).get('value'),
                   'seconds': card.get('result', {}).get('duration_seconds')}
                  for card in cards])


def select_research(memory, query, limit=8):
    from .research_policy import select_insights
    insights = memory.get('insights', [])
    selected = select_insights(insights, query, limit)
    return {**memory, 'insights': selected,
            'omitted_insight_count': memory.get('omitted_insight_count', 0) + len(insights) - len(selected)}


def research_memory_view(memory, *, review=False, baker=False):
    """Keep claims and applicable recipes; detailed evidence remains in the research archive."""
    keys = ('id', 'phase', 'statement', 'kind', 'importance', 'confidence')
    if not review:
        keys += ('implication',)
    result_keys = ('revision', 'omitted_insight_count') if baker else ('revision', 'open_questions', 'coverage', 'omitted_insight_count')
    result = {key: memory[key] for key in result_keys if key in memory}
    result['insights'] = [{key: item[key] for key in keys if key in item} for item in memory.get('insights', [])]
    for source, insight in zip(memory.get('insights', []), result['insights']):
        if baker:
            improvements = [facts['improvement_percent'] for facts in source.get('evidence', {}).values()
                            if isinstance(facts, dict) and facts.get('improvement_percent') is not None]
            if improvements:
                insight['improvement_percent'] = round(improvements[0], 2) if len(improvements) == 1 else [round(x, 2) for x in improvements]
            continue
        measured = []
        for facts in source.get('evidence', {}).values():
            if not isinstance(facts, dict):
                continue
            values = {key: facts[key] for key in ('kind', 'scope', 'rows_used', 'improvement_percent') if key in facts}
            if 'improvement_percent' in values and isinstance(values['improvement_percent'], float):
                values['improvement_percent'] = round(values['improvement_percent'], 2)
            for side in ('baseline', 'candidate'):
                if side in facts:
                    values[side] = {key: (round(facts[side][key], 4) if isinstance(facts[side][key], float) else facts[side][key])
                                    for key in ('metric', 'value') if key in facts[side]}
            if values:
                measured.append(values)
        if measured:
            insight['measured'] = measured
    if not review:
        result['feature_candidates'] = [{key: value for key, value in item.items() if key in {
            'id', 'build_id', 'research_node', 'features', 'columns', 'metric', 'improvement', 'relative_improvement',
            'scope', 'trial_status', 'trials', 'decision'}}
                                        for item in memory.get('feature_candidates', [])]
    return result


def planner_catalog(catalog, idea, limit=24):
    from .storage import fingerprint
    selected = {idea.parent_id: list(catalog[idea.parent_id])} if idea.parent_id in catalog else {}
    count = sum(len(blocks) for blocks in selected.values())
    identity = lambda block: fingerprint({'name': block['name'], 'requires': block['requires'],
                                         'provides': block['provides'], 'operation': operation_view(block.get('operation'))})
    seen = {identity(block) for blocks in selected.values() for block in blocks}
    wanted = set(idea.change_blocks)
    candidates = [(node, block) for node, blocks in reversed(list(catalog.items()))
                  if node != idea.parent_id for block in blocks]
    candidates.sort(key=lambda pair: pair[1]['name'] not in wanted)
    for node, block in candidates:
        if count >= limit:
            break
        key = identity(block)
        if key in seen:
            continue
        seen.add(key)
        selected.setdefault(node, []).append(block)
        count += 1
    return {node: [{**{key: block[key] for key in ('name', 'requires', 'provides')},
                    'operation': operation_view(block.get('operation'))} for block in blocks]
            for node, blocks in selected.items()}


def repair_context(context):
    context = node_context(context, context['hypothesis'].get('research_ids', []))
    target = context['target_block']
    keys = set(target['requires']) | set(target['provides'])
    neighbors = [block for block in context['assembly']['blocks']
                 if block['name'] != target['name'] and keys.intersection(block['requires'] + block['provides'])]
    kind = (context.get('previous_operation') or target.get('operation') or {}).get('kind')
    return compact({**task_context(context, profile=False),
        'node_id': context['node_id'], 'purpose': context['purpose'],
        'hypothesis': {'rationale': context['hypothesis']['rationale'],
                       'pipeline': {key: context['hypothesis']['pipeline'].get(key) for key in ('model', 'training')}},
        'target_block': {key: target[key] for key in ('name', 'requires', 'provides')},
        'previous_operation': operation_view(context.get('previous_operation')),
        'neighbors': [{**{key: block[key] for key in ('name', 'requires', 'provides')},
                       'operation': operation_view(block.get('operation'))} for block in neighbors],
        'execution_error': context['execution_error'], 'failed_stage': context['failed_stage'],
        'experiment_timeout_seconds': context['experiment_timeout_seconds'],
        'operations': {kind: context['operations'][kind]} if kind else context['operations'],
        'data_research': context.get('data_research')})


def automl_facts(value):
    if isinstance(value, dict):
        return {key: automl_facts(item) for key, item in value.items()
                if key not in {'configuration', 'model_directory', 'model_file', 'selected_model_file'}}
    if isinstance(value, list):
        return [automl_facts(item) for item in value]
    return value


def analyzer_context(context):
    row = context['experiment']
    context = node_context(context, row['hypothesis'].get('research_ids', []))
    manifest = context.get('executed_assembly') or row['assembly']
    actual = context.get('tool_configurations', {})
    operations = [{**{key: block[key] for key in ('name', 'requires', 'provides')},
                   'operation': operation_view(actual.get(block['name'], block.get('operation')))}
                  for block in manifest['blocks']]
    comparisons = []
    for comparison in context.get('comparisons', []):
        changes = []
        for change in comparison.get('changes', []):
            if change['path'] == 'assembly.blocks':
                change = {**change, **{side: [{'name': block['name'], 'operation': operation_view(block.get('operation'))}
                                             for block in change[side]] for side in ('before', 'after')}}
            changes.append(change)
        comparisons.append({**comparison, 'changes': changes})
    return compact({**task_context(context, profile=False),
        'experiment': {'node_id': row['node_id'], 'parent_id': row['parent_id'],
                       'hypothesis': {'rationale': row['hypothesis']['rationale'],
                                      'pipeline': row['hypothesis']['pipeline']}, 'result': row['result']},
        'saved_operations': operations, 'comparisons': comparisons,
        'comparison_facts': context.get('comparison_facts', {}),
        'automl_diagnostics': automl_facts(context.get('automl_diagnostics', [])),
        'repair_attempts': context.get('repair_attempts', []), 'execution_stages': context.get('execution_stages', []),
        'input_facts': context.get('input_facts', {}), 'data_research': context.get('data_research')})


def research_history(nodes):
    from .research_policy import FINISHED
    active = [row for row in nodes if row['stage'] not in FINISHED]
    recent = [row for row in nodes if row['stage'] in FINISHED][-4:]
    return [compact({'node_id': row['node_id'], 'phase': row['phase'], 'stage': row['stage'],
        'direction': row.get('direction'), 'mode': row.get('mode'), 'question': row.get('question'),
        'summary': row.get('review', {}).get('summary'), 'error': row.get('error'),
        'probes': [{'kind': value['kind'], 'columns': value.get('columns'),
                    'baseline_columns': value.get('baseline_columns'),
                    'features': [recipe['name'] for recipe in value.get('features', [])],
                    'sample_rows': value.get('sample_rows'), 'filters': value.get('filters')}
                   for value in row.get('analyses', [])]}) for row in [*recent, *active]]


def research_columns(profile, direction, focus, offset=0, limit=10):
    """All columns stay discoverable; detailed statistics follow the current focus."""
    source = profile['tables']['train']
    excluded = set(profile.get('target_columns', []) + profile.get('id_columns', []))
    names = [name for name in source['columns'] if name not in excluded]
    mentioned = [name for name in names if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', str(focus), re.I)]
    if direction == 'quality':
        ranked = sorted(names, key=lambda name: source['columns'][name]['missing'], reverse=True)
    elif direction == 'entities':
        ranked = sorted(names, key=lambda name: 1 < source['columns'][name]['unique'] < source['rows'], reverse=True)
    else:
        shift = offset % max(1, len(names))
        ranked = names[shift:] + names[:shift]
    return set([*mentioned, *ranked[:limit]])


def model_feedback(row):
    res = result_view(row.result)
    if 'value' in res and isinstance(res['value'], float):
        res['value'] = round(res['value'], 4)
    if 'duration_seconds' in res and isinstance(res['duration_seconds'], float):
        res['duration_seconds'] = round(res['duration_seconds'], 2)
    return compact({'node_id': row.node_id,
                    'model': {key: value for key, value in row.hypothesis.pipeline.model.items() if key in {'name', 'engine'}},
                    'result': res,
                    'analysis': row.analysis.summary if row.analysis else None})


def requirements(extraction, *, prepared=False):
    result = {role: facts.model_dump(exclude={'evidence'}) for role, facts in (
        ('problem', extraction.problem), ('data', extraction.data),
        ('metric', extraction.metric), ('output', extraction.output))}
    if prepared:
        result.pop('data')  # Nodes receive DataFrames; input paths and parsing options belong to preparation.
        result['output'] = {k: v for k, v in result['output'].items()
                            if k not in {'submission_file', 'files', 'format'}}
    return result


def preparation_view(preparation):
    profiles = {}
    for role, table in preparation.profile.items():
        types, missing = {}, {}
        for name, column in table['columns'].items():
            types.setdefault(column['dtype'], []).append(name)
            if column['missing']:
                missing[name] = column['missing']
        profiles[role] = {'rows': table['rows'], 'columns_by_dtype': types, 'missing': missing}
    return {'data': preparation.data.model_dump(exclude={'files'}), 'metric': preparation.metric.model_dump(),
            'output': preparation.output.model_dump(), 'profile': profiles,
            'validation': {key: value for key, value in preparation.validation.items()
                           if key not in {'split_file', 'split_sha256', 'split_plan'}}, 'issues': preparation.issues}


def execution_contract(preparation):
    return {
        'completed_setup': ['requirements_extracted', 'input_tables_loaded', 'profile_measured',
                            'fit_validation_split_saved', 'scorer_and_submission_writer_defined'],
        'inputs': {
            'fit': {'type': 'DataFrame', 'rows': preparation.validation.get('model_fit_rows', preparation.validation['fit_rows']),
                    'target_columns': preparation.data.target_columns},
            'validation': {'type': 'DataFrame', 'rows': preparation.validation['validation_rows'],
                           'target_columns': [], 'row_order': 'fixed'},
            'test': {'type': 'DataFrame', 'rows': preparation.profile['test']['rows'],
                     'target_columns': [], 'row_order': 'fixed'},
            'preparation': {'metadata': 'Fixed compact metadata, not a recipe to execute again',
                            'runtime': {'time_budget_seconds': 'Total budget for training/search in this invocation',
                                        'deadline': 'time.monotonic() deadline; compute remaining time before each fit',
                                        'num_cpus': 1, 'num_gpus': 0,
                                        'is_smoke_check': 'True for the small preliminary check',
                                        'work_directory': 'Current node directory; persist model files here'}},
            'auxiliary': 'Other already loaded named input tables, without full train or validation labels',
        },
        'fixed': ['input_files', 'outer_split', 'validation_and_test_order', 'scorer', 'submission_writer'],
        'node_work': 'Transform supplied inputs, fit models on fit, return validation/test prediction DataFrames',
    }


def result_view(finish):
    return finish.model_dump(exclude={'schema_version', 'metric', 'preparation_sha256', 'start_sha256',
                                     'code_sha256', 'validation_file', 'submission_file', 'notes'})


def history_view(row):
    return {'node_id': row.node_id, 'parent_id': row.parent_id,
            'hypothesis': row.hypothesis.model_dump(), 'assembly': row.assembly.model_dump(),
            'result': result_view(row.result), 'analysis_status': row.analysis_status,
            'analysis': row.analysis.model_dump() if row.analysis else None}


def idea_history_view(row):
    result = history_view(row)
    result['hypothesis'].pop('expected_effect')  # Forecasts are not new evidence about past experiments.
    result['assembly'] = {
        'order': [block.name for block in row.assembly.blocks],
        'sources': {block.name: block.source_node or row.node_id for block in row.assembly.blocks},
        'write_blocks': list(row.assembly.write_blocks),
    }
    return result


def catalog_view(catalog):
    # Identical bytes have the same interface. Advertise each immutable version once.
    seen, result = set(), {}
    for node_id, blocks in catalog.items():
        for block in blocks:
            identity = (block['name'], block['sha256'], tuple(block['requires']), tuple(block['provides']))
            if identity not in seen:
                result.setdefault(node_id, []).append({**{key: block[key] for key in ('name', 'description', 'requires', 'provides')},
                                                       'operation': block.get('operation')})
                seen.add(identity)
    return result


def node_context(context, research_ids):
    """Only Baker selects research; execution roles receive the selected evidence."""
    result = {key: value for key, value in context.items() if key != 'data_research'}
    if research_ids:
        research = context.get('data_research', {})
        insights = [item for item in research.get('insights', []) if item['id'] in research_ids]
        result['data_research'] = {'insights': insights,
            'unavailable_ids': sorted(set(research_ids) - {item['id'] for item in insights})}
    return result


def research_profile_view(profile, *, detailed=True, focus_columns=None):
    """Keep measured values and column names while removing repeated per-column JSON keys."""
    result = {key: value for key, value in profile.items() if key != 'tables'}
    if 'entity_key_candidates' in result and isinstance(result['entity_key_candidates'], dict):
        cands = [dict(c) for c in result['entity_key_candidates'].get('candidates', [])]
        for c in cands:
            for k, v in c.items():
                if isinstance(v, float):
                    c[k] = round(v, 4)
        result['entity_key_candidates'] = {**result['entity_key_candidates'], 'candidates': cands}
    result['quantile_levels'] = [0, .25, .5, .75, 1] if detailed else [0, .5, 1]
    if not detailed:
        result['frequent_values_limit'] = 3
    result['tables'] = {}
    for role, table in profile['tables'].items():
        types, unique, missing, nonfinite, quantiles, frequent = {}, {}, {}, {}, {}, {}
        for name, column in table['columns'].items():
            types.setdefault(column['dtype'], []).append(name)
            unique[name] = column['unique']
            if column['missing']:
                missing[name] = column['missing']
            if focus_columns is not None and name not in focus_columns:
                continue
            if column.get('nonfinite'):
                nonfinite[name] = column['nonfinite']
            if column.get('quantiles'):
                quantiles[name] = [round(column['quantiles'][str(float(q))], 4)
                                   if isinstance(column['quantiles'][str(float(q))], float)
                                   else column['quantiles'][str(float(q))]
                                   for q in result['quantile_levels']]
            if 'frequent_values' in column:
                frequent[name] = column['frequent_values'] if detailed else dict(
                    sorted(column['frequent_values'].items(), key=lambda item: item[1], reverse=True)[:3])
        result['tables'][role] = {'rows': table['rows'], 'columns_by_dtype': types, 'unique': unique,
            'missing': missing, 'nonfinite': nonfinite, 'quantiles': quantiles, 'frequent_values': frequent}
    return result
