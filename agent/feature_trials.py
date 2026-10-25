"""Track measured feature recipes through planning and actual model experiments."""
import json

from .plans import FeatureSpec
from .storage import fingerprint


def candidate_id(candidate):
    key = candidate.get('measurement_key') or fingerprint({name: candidate.get(name)
        for name in ('features', 'columns', 'filters', 'validation')})
    return candidate.get('id') or 'feature_' + key[:12]


def recipe_keys(features):
    result = set()
    for feature in features:
        value = FeatureSpec.model_validate(feature).model_dump()
        if value['operation'] == 'history':
            # Omitted executor defaults and explicit defaults describe the same recipe.
            defaults = {'window': None, 'max_age': None, 'lag': 1, 'min_history': 1, 'fallback': 'global'}
            value['parameters'] = {**defaults, **value['parameters']}
        result.add(fingerprint(value))
    return result


def assembly_recipes(assembly):
    recipes = []
    for block in assembly['blocks']:
        call = block.get('operation') or {}
        parameters = call.get('parameters', {})
        if call.get('kind') == 'features':
            recipes.extend(parameters.get('recipes', []))
        elif call.get('kind') == 'automl':
            recipes.extend(parameters.get('features', []))
    return recipes


def candidate_view(candidates, directory, rows, records, decisions):
    if not candidates:
        return []
    measured = []
    for row in rows:
        path = directory / 'nodes' / row.node_id / 'assembly.json'
        assembly = json.loads(path.read_text(encoding='utf-8')) if path.exists() else row.assembly.model_dump()
        measured.append((row, recipe_keys(assembly_recipes(assembly))))
    result = []
    for candidate in candidates:
        identity = candidate_id(candidate)
        wanted = recipe_keys(candidate['features'])
        trials = [{'node_id': row.node_id, 'status': row.result.status, 'value': row.result.value}
                  for row, recipes in measured if wanted <= recipes]
        pending = [record.get('node_id') or record['id'] for record in records
                   if record['status'] in {'queued', 'planning', 'planned'}
                   and identity in record['idea'].get('feature_candidate_ids', [])
                   and not any(trial['node_id'] == record.get('node_id') for trial in trials)]
        status = ('tested' if any(trial['status'] == 'completed' for trial in trials)
                  else 'pending' if pending else 'failed' if trials else 'untested')
        result.append({**candidate, 'id': identity, 'trial_status': status, 'trials': trials,
                       'pending': pending, 'decision': decisions.get(identity)})
    return result


def selected_candidates(candidates, identities):
    by_id = {candidate_id(item): item for item in candidates}
    missing = set(identities) - by_id.keys()
    if missing:
        raise ValueError(f'Unknown feature candidate IDs: {sorted(missing)}')
    selected = [by_id[identity] for identity in identities]
    names = {}
    for candidate in selected:
        for recipe in candidate['features']:
            key = next(iter(recipe_keys([recipe])))
            if recipe['name'] in names and names[recipe['name']] != key:
                raise ValueError('Conflicting feature names require separate hypotheses')
            names[recipe['name']] = key
    return selected


def prediction_features(assembly):
    lineage = {}
    for block in assembly['blocks']:
        call = block.get('operation') or {}
        kind, parameters = call.get('kind'), call.get('parameters', {})
        inputs, outputs = call.get('inputs', {}), call.get('outputs', {})
        source = lambda key: lineage.get(inputs.get(key, key), set())
        if kind == 'features':
            used = source('fit') | {recipe['name'] for recipe in parameters.get('recipes', [])}
        elif kind == 'preprocess':
            used = source('fit') - set(parameters.get('drop_columns', []))
            if parameters.get('feature_columns'):
                used &= set(parameters['feature_columns'])
        elif kind == 'fit':
            used = source('fit_features')
        elif kind == 'predict':
            used = source('model')
        elif kind in {'automl', 'forecast'}:
            used = source('fit')
            selected = parameters.get('feature_columns' if kind == 'forecast' else 'columns')
            if selected:
                used &= set(selected)
            used |= {recipe['name'] for recipe in parameters.get('features', [])}
            used -= set(parameters.get('drop_columns', []))
        else:
            used = set().union(*(lineage.get(key, set()) for key in block.get('requires', [])))
        from .operations import CATALOG
        for key in CATALOG.get(kind, {}).get('provides', []):
            lineage[outputs.get(key, key)] = used
    return lineage.get('validation_predictions', set()) & lineage.get('test_predictions', set())


def check_recipes(proposal, selected):
    """A selected recipe must survive planning exactly; its model choice remains free."""
    if not selected:
        return
    assembly = proposal.assembly.model_dump()
    actual = recipe_keys(assembly_recipes(assembly))
    used = prediction_features(assembly)
    for candidate in selected:
        if not recipe_keys(candidate['features']) <= actual:
            raise ValueError(f"Include the exact recipes selected by {candidate_id(candidate)}")
        missing = {feature['name'] for feature in candidate['features']} - used
        if missing:
            raise ValueError(f'Connect selected features to model training and predictions: {sorted(missing)}')
