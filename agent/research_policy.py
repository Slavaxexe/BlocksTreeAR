"""Research coverage, measurement identity and importance-based context selection."""
from collections import Counter
import re

from .storage import fingerprint


DIRECTIONS = {
    'quality': 'Schema, missingness, constants, duplicates and reliability of inputs.',
    'distribution': 'Train/query shift and stability across relevant populations.',
    'relationships': 'Marginal and joint dependencies; distinguish association from predictive gain.',
    'entities': 'Unit of observation, proposed single/composite keys, repetition and ordering; check applicability.',
    'transformations': 'Non-target feature transformations and combinations supported by the catalog.',
    'encoding': 'Target-derived categorical representations with honest internal evaluation.',
    'history': 'Ordered dependencies and historical features, only if the observations support them.',
    'model_checks': 'Proxy prediction, permutation, ablation and error-relevant feature comparisons.',
}
IMPORTANCE = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
CONFIDENCE = {'low': 0, 'medium': 1, 'high': 2}
FINISHED = {'completed', 'failed', 'cancelled', 'duplicate'}


def directions(requests):
    result = set()
    kinds = {'quality': 'quality', 'redundancy': 'quality', 'missingness': 'quality',
             'shift': 'distribution', 'stability': 'distribution', 'entities': 'entities',
             'association': 'relationships', 'correlation': 'relationships', 'groups': 'relationships',
             'interactions': 'relationships', 'predictive': 'model_checks',
             'permutation': 'model_checks', 'ablation': 'model_checks',
             'targets': 'relationships', 'queries': 'entities', 'series': 'history',
             'clusters': 'model_checks', 'task_predictive': 'model_checks'}
    for request in requests:
        value = request.model_dump() if hasattr(request, 'model_dump') else request
        if value['kind'] == 'feature':
            for feature in [*value.get('baseline_features', []), *value.get('features', [])]:
                result.add({'history': 'history', 'target_encoding': 'encoding'}.get(feature['operation'], 'transformations'))
        else:
            result.add(kinds[value['kind']])
    return result


def measurement_key(request):
    value = request.model_dump() if hasattr(request, 'model_dump') else dict(request)
    value = {key: item for key, item in value.items() if key not in {'name', 'top_k'}}
    for key in ('columns', 'baseline_columns'):
        value[key] = sorted(value.get(key, []))
    if value['kind'] == 'entities':
        value['key_candidates'] = sorted(sorted(key) for key in (value.get('key_candidates') or [value['columns']]))
        return fingerprint({key: value.get(key) for key in ('kind', 'dataset', 'compare', 'key_candidates',
            'split_column', 'filters', 'comparison_filters')})
    names = {}
    for field in ('baseline_features', 'features'):
        recipes = []
        for feature in value.get(field, []):
            recipe = dict(feature)
            recipe['columns'] = [names.get(column, column) for column in recipe['columns']]
            recipe['output'] = names.setdefault(recipe.pop('name'), f'feature_{len(names)}')
            recipes.append(recipe)
        value[field] = recipes
    return fingerprint(value)


def coverage(nodes):
    result = {name: {'attempts': 0, 'tested': 0, 'active': 0, 'declined': 0} for name in DIRECTIONS}
    for node in nodes:
        assigned = node.get('direction')
        if assigned in result:
            result[assigned]['attempts'] += 1
            if node.get('stage') not in FINISHED:
                result[assigned]['active'] += 1
            if node.get('alternative_reason') and assigned not in node.get('measured_directions', []):
                result[assigned]['declined'] += 1
        if node.get('stage') == 'completed':
            for name in node.get('measured_directions', []):
                result[name]['tested'] += 1
    return result


def choose_direction(nodes, phase, insights):
    allowed = list(DIRECTIONS) if phase == 'fit' else ['quality', 'distribution', 'relationships', 'entities']
    ledger = coverage(nodes)
    refine = sum(node.get('phase') == phase for node in nodes) % 3 == 2
    if refine:
        for insight in sorted(insights, key=importance_score, reverse=True):
            name = insight.get('direction')
            if (insight.get('active', True) and insight.get('importance') != 'low'
                    and name in allowed and not ledger[name]['active']):
                return name, 'refine'
    name = min(allowed, key=lambda key: (ledger[key]['active'], ledger[key]['attempts']))
    return name, 'explore'


def importance_score(item):
    return (4 * IMPORTANCE.get(item.get('importance', 'medium'), 1)
            + CONFIDENCE.get(item.get('confidence', 'medium'), 1)
            + (item.get('kind') == 'observation'))


def select_insights(insights, query='', limit=8):
    words = lambda value: set(re.findall(r'\w{3,}', str(value).lower()))
    focus = words(query)
    ranked = sorted((item for item in insights if item.get('active', True)),
        key=lambda item: (importance_score(item), min(4, len(words(item['statement']) & focus)),
                          item.get('node_id', '')), reverse=True)
    selected, counts, topics = [], Counter(), set()
    for item in ranked:
        if item.get('importance') == 'low':
            continue
        topic = ' '.join(str(item.get('topic', '')).lower().split())
        if topic and topic in topics:
            continue
        family = item.get('direction') or item.get('topic') or item.get('node_id') or item['id']
        if counts[family] >= 2 and item.get('importance') != 'critical':
            continue
        selected.append(item)
        if topic:
            topics.add(topic)
        counts[family] += 1
        if len(selected) >= limit:
            break
    return selected


def select_questions(questions, metadata, limit=6):
    ranked = sorted(questions, key=lambda text: IMPORTANCE.get(metadata.get(text, {}).get('importance', 'medium'), 1), reverse=True)
    result, counts = [], Counter()
    for text in ranked:
        family = metadata.get(text, {}).get('direction', '')
        if counts[family] >= 2:
            continue
        result.append(text)
        counts[family] += 1
        if len(result) >= limit:
            break
    return result


def select_candidates(candidates, limit=3):
    ranked = sorted(candidates, key=lambda item: (importance_score(item), not item.get('filters'),
        item.get('relative_improvement', 0), item.get('validation_rows', 0)), reverse=True)
    result, families = [], set()
    for item in ranked:
        family = tuple(sorted({recipe['operation'] for recipe in item['features']}))
        if family not in families:
            result.append(item)
            families.add(family)
        if len(result) >= limit:
            return result
    result.extend(item for item in ranked if item not in result)
    return result[:limit]


def evidence_summary(value):
    if not isinstance(value, dict):
        return {'value': value}
    result = {key: value[key] for key in ('kind', 'scope', 'status', 'rows_used', 'rows_available',
        'target_available', 'improvement', 'absolute_improvement', 'improvement_percent') if key in value}
    for key in ('baseline', 'candidate'):
        if key in value:
            result[key] = {name: value[key][name] for name in ('metric', 'value', 'estimator',
                'inner_train_rows', 'inner_validation_rows', 'inner_reference_rows', 'feature_count', 'recipe_names') if name in value[key]}
    if 'comparison' in value:
        result['comparison'] = value['comparison']
    if 'candidates' in value:
        result['candidates'] = [{key: candidate[key] for key in ('key', 'unique_keys', 'keys_in_multiple_periods',
            'comparison_rows', 'comparison_key_coverage', 'comparison_strict_past_coverage',
            'within_period_ambiguity_fraction', 'within_period_ambiguous_rows', 'source_rows',
            'duplicate_key_period_row_fraction') if key in candidate} for candidate in value['candidates']]
    return result
