"""Role-specific JSON views. Full provenance stays on disk, outside model prompts."""


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
                           if key not in {'split_file', 'split_sha256'}}, 'issues': preparation.issues}


def execution_contract(preparation):
    return {
        'completed_setup': ['requirements_extracted', 'input_tables_loaded', 'profile_measured',
                            'fit_validation_split_saved', 'scorer_and_submission_writer_defined'],
        'inputs': {
            'fit': {'type': 'DataFrame', 'rows': preparation.validation['fit_rows'],
                    'target_columns': preparation.data.target_columns},
            'validation': {'type': 'DataFrame', 'rows': preparation.validation['validation_rows'],
                           'target_columns': [], 'row_order': 'fixed'},
            'test': {'type': 'DataFrame', 'rows': preparation.profile['test']['rows'],
                     'target_columns': [], 'row_order': 'fixed'},
            'preparation': {'metadata': 'Fixed compact metadata, not a recipe to execute again',
                            'runtime': {'time_budget_seconds': 'Total budget for training/search in this invocation',
                                        'deadline': 'time.monotonic() deadline; compute remaining time before each fit',
                                        'num_cpus': 2, 'num_gpus': 0,
                                        'is_smoke_check': 'True for the small preliminary check',
                                        'work_directory': 'Current node directory; persist model files here'}},
            'auxiliary': 'Other already loaded named input tables, without full train or validation labels',
        },
        'fixed': ['input_files', 'outer_split', 'validation_and_test_order', 'scorer', 'submission_writer'],
        'node_work': 'Transform supplied inputs, fit models on fit, return validation/test prediction DataFrames',
    }


def result_view(finish):
    return finish.model_dump(exclude={'schema_version', 'metric', 'preparation_sha256', 'start_sha256',
                                     'code_sha256', 'validation_file', 'submission_file', 'diagnostics_file', 'notes'})


def history_view(row):
    return {'node_id': row.node_id, 'parent_id': row.parent_id,
            'hypothesis': row.hypothesis.model_dump(), 'assembly': row.assembly.model_dump(),
            'result': result_view(row.result), 'analysis_status': row.analysis_status,
            'analysis': row.analysis.model_dump() if row.analysis else None}


def idea_history_view(row):
    result = history_view(row)
    result['assembly'] = [block.name for block in row.assembly.blocks]
    return result


def catalog_view(catalog):
    # Identical bytes have the same interface. Advertise each immutable version once.
    seen, result = set(), {}
    for node_id, blocks in catalog.items():
        for block in blocks:
            identity = (block['name'], block['sha256'], tuple(block['requires']), tuple(block['provides']))
            if identity not in seen:
                result.setdefault(node_id, []).append({key: block[key] for key in ('name', 'description', 'requires', 'provides')})
                seen.add(identity)
    return result
