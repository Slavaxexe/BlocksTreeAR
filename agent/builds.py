"""Immutable ready assemblies and explicit changes; no LLM configuration copying."""
import copy
import json
import re

from .contracts import Assembly, AssemblyChanges, Block, Hypothesis, Pipeline, Proposal
from .plans import ToolCall, FeatureSpec
from .storage import fingerprint, save


def block(name, call):
    call = call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
    requires, provides = call.interface()
    return Block(name=name, description=name, requires=requires, provides=provides, operation=call)


def store(root, identity, assembly, model, evidence=None):
    if not re.fullmatch(r'(node_[0-9]{3,}|eda_[a-f0-9]{12,64})', identity):
        raise ValueError('Invalid build ID')
    value = {'id': identity, 'assembly': assembly.model_dump(), 'model': model, 'evidence': evidence or {}}
    value['sha256'] = fingerprint(value['assembly'])
    path = root / 'builds' / f'{identity}.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))['sha256'] != value['sha256']:
            raise ValueError('A registered build cannot change')
    else:
        save(path, value)
    return value


def load_build(root, identity):
    if not re.fullmatch(r'(node_[0-9]{3,}|eda_[a-f0-9]{12,64})', identity):
        raise ValueError('Invalid build ID')
    value = json.loads((root / 'builds' / f'{identity}.json').read_text(encoding='utf-8'))
    if fingerprint(value['assembly']) != value['sha256']:
        raise ValueError('Saved assembly changed')
    return value


def register_node(root, row):
    path = root / 'builds' / f'{row.node_id}.json'
    if path.exists():
        return load_build(root, row.node_id)
    from .assembly import implementation_hash
    source = root / 'nodes' / row.node_id
    if implementation_hash(source) != row.result.code_sha256:
        raise ValueError('Executed source changed')
    manifest = json.loads((source / 'assembly.json').read_text(encoding='utf-8'))
    blocks = []
    for saved in manifest['blocks']:
        call = copy.deepcopy(saved.get('operation'))
        if call is None:
            return None  # Legacy generated Python has no declarative inheritance.
        changed = False
        if call['kind'] == 'fit' and call['parameters'].get('search_space'):
            from .storage import fingerprint as digest
            key = digest(call.get('outputs', {}).get('model', 'model'))[:12]
            facts = json.loads((source / f'fit_{key}_full_diagnostics.json').read_text(encoding='utf-8'))
            call['parameters']['fixed_parameters'] = {
                **call['parameters'].get('fixed_parameters', {}), **facts.get('best_parameters', {})}
            call['parameters']['search_space'] = {}
            changed = True
        if call['kind'] == 'automl':
            from .model_evidence import frozen_available
            facts = json.loads((source / f"{call['parameters']['engine']}_full_diagnostics.json").read_text(encoding='utf-8'))
            if not frozen_available(facts):
                return None
            # Preserve the entire adapter/preprocessing recipe, freezing only model selection.
            call['parameters']['reuse_node'] = row.node_id
            call['parameters']['search_space'] = {}
            changed = True
        item = block(saved['name'], call)
        if not changed:
            item = item.model_copy(update={'source_node': row.node_id})
        blocks.append(item)
    assembly = Assembly(blocks=blocks, write_blocks=[item.name for item in blocks if item.source_node is None])
    return store(root, row.node_id, assembly, row.hypothesis.pipeline.model)


def register_eda(root, identity, configuration, evidence):
    blocks = []
    recipes = configuration['features']
    if recipes:
        blocks.append(block('features', {'kind': 'features', 'parameters': {'recipes': recipes}}))
    inputs = {scope: scope + '_table' for scope in ('fit', 'validation', 'test')} if recipes else {}
    blocks += [block('preprocess', {'kind': 'preprocess', 'parameters': configuration['preprocess'], 'inputs': inputs}),
               block('fit', {'kind': 'fit', 'parameters': configuration['fit']}),
               block('predict', {'kind': 'predict'})]
    return store(root, identity, Assembly(blocks=blocks, write_blocks=[item.name for item in blocks]),
                 {'name': configuration['fit']['estimator']}, evidence)


def merge(before, changes):
    result = copy.deepcopy(before)
    for key, value in changes.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else copy.deepcopy(value)
    return result


def compile_build(value, changes, idea):
    if fingerprint(value['assembly']) != value['sha256']:
        raise ValueError('Saved assembly changed')
    changes = changes or AssemblyChanges()
    original = Assembly.model_validate(value['assembly'])
    blocks = {item.name: item for item in original.blocks}
    order = [item.name for item in original.blocks]
    added, removed = set(), set()
    for edit in changes.edits:
        if edit.block not in blocks:
            raise ValueError(f'Unknown block: {edit.block}; insert a new stage explicitly')
        item = blocks[edit.block]
        call = item.operation.model_dump()
        delta = copy.deepcopy(edit.parameters)
        replacing_model = (call['kind'] == 'fit' and 'estimator' in delta
                           and delta['estimator'] != call['parameters'].get('estimator'))
        if replacing_model:
            # A new estimator cannot inherit the old family's constructor or tuned settings.
            for key in ('parameters', 'fixed_parameters', 'search_space', 'source_node'):
                call['parameters'].pop(key, None)
        elif call['kind'] == 'fit' and 'parameters' in delta:
            # Apply constructor/deep sklearn parameter changes after creating the unchanged estimator.
            delta['fixed_parameters'] = {**delta.get('fixed_parameters', {}), **delta.pop('parameters')}
        call['parameters'] = merge(call['parameters'], delta)
        if edit.features or edit.remove_features or edit.feature_parameters:
            field = 'recipes' if call['kind'] == 'features' else 'features' if call['kind'] == 'automl' else None
            if field is None:
                raise ValueError('Recipe changes need a features/automl block; insert a features stage if absent')
            recipes = {recipe['name']: recipe for recipe in call['parameters'].get(field, [])}
            if set(edit.remove_features) - recipes.keys():
                raise ValueError('Cannot remove an unknown recipe')
            for name in edit.remove_features:
                recipes.pop(name)
            recipes.update({recipe.name: recipe.model_dump() for recipe in edit.features})
            for name, parameters in edit.feature_parameters.items():
                if name not in recipes:
                    raise ValueError(f'Unknown recipe: {name}')
                recipes[name]['parameters'] = merge(recipes[name].get('parameters', {}), parameters)
            call['parameters'][field] = [FeatureSpec.model_validate(recipe).model_dump() for recipe in recipes.values()]
            added.update(recipe.name for recipe in edit.features)
            removed.update(edit.remove_features)
        blocks[edit.block] = block(item.name, call)
    for name in changes.remove:
        if name not in blocks:
            raise ValueError(f'Unknown removed block: {name}')
        blocks.pop(name)
        order.remove(name)
    for item in changes.insert:
        if item.name in blocks or item.operation is None:
            raise ValueError('Inserted stages must be new trusted operations; remove an old stage explicitly before replacing it')
        blocks[item.name] = block(item.name, item.operation)
    if changes.insert and changes.order is None:
        raise ValueError('Specify the complete order when inserting stages')
    if changes.order is not None:
        order = changes.order
    if len(order) != len(blocks) or set(order) != blocks.keys():
        raise ValueError('Assembly order must contain every block exactly once')
    for name, item in list(blocks.items()):
        call = item.operation.model_dump()
        if call['kind'] == 'preprocess' and call['parameters'].get('feature_columns') and (added or removed):
            columns = [column for column in call['parameters']['feature_columns'] if column not in removed]
            columns += sorted(added - set(columns))
            call['parameters']['feature_columns'] = columns
            blocks[name] = block(name, call)
    assembly = Assembly(blocks=[blocks[name] for name in order],
                        write_blocks=[name for name in order if blocks[name].source_node is None])
    signature = lambda built: fingerprint([item.operation.model_dump() for item in built.blocks])
    if value['id'].startswith('node_') and signature(assembly) == signature(original):
        raise ValueError('A node build needs an executable change; the unchanged parent is already evaluated')
    model = copy.deepcopy(value['model'])
    for item in assembly.blocks:
        if item.operation.kind == 'fit' and item.operation.parameters.get('estimator'):
            model = {'name': item.operation.parameters['estimator']}
        elif item.operation.kind == 'saved_ensemble':
            model = {'name': item.operation.parameters['method'] + ' ensemble',
                     'sources': item.operation.parameters['source_nodes']}
    return Proposal(purpose=idea.purpose, parent_id=idea.parent_id,
        hypothesis=Hypothesis(rationale=idea.change, expected_effect=idea.expected_effect,
            research_ids=idea.research_ids,
            pipeline=Pipeline(data={'build_id': value['id'], 'build_sha256': value['sha256']},
                preprocessing={'source': 'immutable build with explicit changes'}, model=model,
                training={'changes': changes.model_dump(exclude_defaults=True)})), assembly=assembly)


def card(value):
    recipes = [recipe['name'] for item in value['assembly']['blocks']
        for recipe in item['operation']['parameters'].get('recipes', item['operation']['parameters'].get('features', []))]
    return {'id': value['id'], 'model': value['model'], 'blocks': [item['name'] for item in value['assembly']['blocks']],
            'key_features': recipes, 'evidence': value['evidence']}
