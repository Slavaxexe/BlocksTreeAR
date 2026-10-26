"""Build a root-level offline API reference from installed packages, without fitting models.

Usage: python build_ml_reference.py [--output ml_reference_data.py] [--verify]
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import pprint
import re
import textwrap
import warnings

from ml_reference_notes import FORWARDED_TO, MODULES, SYMBOLS, VALIDATED_KWARGS


def resolve(symbol):
    """Import the longest existing module prefix, then resolve public attributes."""
    parts = symbol.split('.')
    for count in range(len(parts), 0, -1):
        try:
            value = importlib.import_module('.'.join(parts[:count]))
        except ModuleNotFoundError as error:
            # A missing dependency inside an existing package must remain visible.
            requested = '.'.join(parts[:count])
            if not (requested == error.name or requested.startswith(error.name + '.')):
                raise
            continue
        for part in parts[count:]:
            value = getattr(value, part)
        return value
    raise ImportError(symbol)


def documentation_parameters(doc):
    """Read NumPy/Google/rst parameter sections, retaining their actual declarations."""
    sections = {'Parameters', 'Other Parameters', 'Keyword Arguments', 'Keyword Args', 'Args', 'Arguments'}
    end_sections = {'Returns', 'Yields', 'Raises', 'Notes', 'Examples', 'Attributes', 'See Also',
                    'References', 'Warnings', 'Methods'}
    active, headers = False, []
    lines = doc.splitlines()
    for index, line in enumerate(lines):
        title = line.strip().rstrip(':')
        if title in sections:
            active = True
            continue
        if title in end_sections:
            active = False
        if active:
            match = re.match(r'^(\s*)(\*{0,2}[a-zA-Z_]\w*(?:\s*,\s*\*{0,2}[a-zA-Z_]\w*)*)\s*(?:\([^)]*\))?\s*:\s*(.*)$', line)
            if match:
                headers.append((index, len(match[1]), match[2], match[3]))
    if not headers:
        return {}
    indent = min(item[1] for item in headers)
    parents, selected = [], []
    scopes = {}
    for item in headers:
        while parents and parents[-1][1] >= item[1]:
            parents.pop()
        # AutoGluon documents accepted **kwargs one indentation level deeper.
        if item[1] == indent or any(parent[2].lstrip('*').endswith('kwargs') for parent in parents):
            selected.append(item)
            scopes[item[0]] = [p[2].lstrip('*') for p in parents
                              if not p[2].lstrip('*').endswith('kwargs')]
        parents.append(item)
    result = {}
    for index, level, names, declaration in selected:
        end = next((i for i, depth, _, _ in headers if i > index and depth <= level), len(lines))
        body = []
        for line in lines[index + 1:end]:
            if line.strip().rstrip(':') in end_sections:
                break
            body.append(line)
        for name in names.split(','):
            path = '.'.join(scopes[index] + [name.strip().lstrip('*')])
            result[path] = {
                'declaration': declaration[:800],
                'description': textwrap.dedent('\n'.join(body)).strip()[:2200],
                'source': 'installed_docstring',
            }
    return result


def source_info(value):
    try:
        lines, line = inspect.getsourcelines(value)
        filename = Path(inspect.getsourcefile(value))
        # Keep portable package paths, not personal workspace/user directories.
        parts = filename.parts
        index = next((i + 1 for i, part in enumerate(parts) if part in {'site-packages', 'dist-packages'}), None)
        return {'file': '/'.join(parts[index:]) if index is not None else filename.name,
                'line': line, 'sha256': hashlib.sha256(''.join(lines).encode()).hexdigest()}
    except (TypeError, OSError):
        return None


def signature_info(value):
    try:
        signature = inspect.signature(value, eval_str=False)
    except (TypeError, ValueError):
        return None, None, {}, False
    parameters = {}
    for name, parameter in signature.parameters.items():
        parameters[name] = {'kind': parameter.kind.name}
        if parameter.default is not inspect.Parameter.empty:
            default = re.sub(r' at 0x[0-9A-Fa-f]+', '', repr(parameter.default))
            parameters[name]['default'] = default[:400]
    compact = signature.replace(parameters=[p.replace(annotation=inspect.Parameter.empty)
                                             for p in signature.parameters.values()],
                                return_annotation=inspect.Signature.empty)
    return str(signature), str(compact), parameters, any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())


def doc_signature(doc):
    lines = doc.strip().splitlines()
    if not lines or not re.match(r'^[\w.]+\(', lines[0]):
        return None
    parts, balance = [], 0
    for line in lines[:20]:
        if not line.strip():
            break
        parts.append(line.strip())
        balance += line.count('(') - line.count(')')
        if balance == 0:
            return ' '.join(parts)
    return None


def describe(symbol, value, package):
    doc = inspect.getdoc(value) or ''
    signature, compact, parameters, variadic = signature_info(value)
    documented = documentation_parameters(doc)
    explicit = set(parameters) - {'self', 'cls', 'args', 'kwargs'}
    extra = {name: item for name, item in documented.items()
             if '.' not in name and name not in explicit and name not in {'args', 'kwargs'}}
    overview = re.split(r'\n\s*(?:Parameters|Args|Arguments|Returns|Attributes)\s*:?(?:\n|$)', doc, maxsplit=1)[0]
    return {
        'package': package, 'kind': 'class' if inspect.isclass(value) else 'property' if isinstance(value, property) else 'callable',
        'signature': signature, 'call_signature': compact,
        'doc_signature': doc_signature(doc) if signature is None else None,
        'parameters': parameters, 'accepts_kwargs': variadic,
        'summary': overview.strip()[:1800], 'documented_parameters': documented,
        'documented_kwargs': extra if variadic else {},
        'kwargs_complete': False,  # Docs/MRO never prove an exhaustive accepted set.
        'source': source_info(value),
    }


def named_dict_keys(function, variable):
    """Read a specific validator dictionary without executing the private method."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    for node in ast.walk(tree):
        candidate = None
        if variable == 'return' and isinstance(node, ast.Return):
            candidate = node.value
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            candidate = node.value
        if isinstance(candidate, ast.Dict):
            return [key.value for key in candidate.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
        if isinstance(candidate, ast.Call) and isinstance(candidate.func, ast.Name) and candidate.func.id == 'dict':
            return [key.arg for key in candidate.keywords if key.arg]
    raise ValueError(f'Validator dictionary not found: {function.__name__}/{variable}')


def package_info(name):
    distribution = metadata.distribution(name)
    info = distribution.metadata
    licenses = {}
    for relative in distribution.files or []:
        if '.dist-info/' not in str(relative).replace('\\', '/'):
            continue
        if Path(relative).name.lower() in {'license', 'license.txt', 'license.md', 'copying', 'copyright'}:
            path = Path(distribution.locate_file(relative))
            if path.is_file() and path.stat().st_size < 100_000:
                licenses[Path(relative).name] = path.read_text(encoding='utf-8', errors='replace')
    return {'version': distribution.version, 'home_page': info.get('Home-page'),
            'project_urls': info.get_all('Project-URL') or [], 'license': info.get('License'),
            'license_texts': licenses}


def collect():
    records, packages, issues = {}, {}, []
    for package, modules in MODULES.items():
        try:
            packages[package] = package_info(package)
        except metadata.PackageNotFoundError:
            issues.append({'package': package, 'error': 'Distribution is not installed'})
            continue
        before = len(records)

        def add(symbol, value, *, methods=True):
            if symbol in records:
                return
            records[symbol] = describe(symbol, value, package)
            callable_object = callable(value) and not inspect.isroutine(value)
            if methods and (inspect.isclass(value) or callable_object):
                for name in dir(value):
                    if name.startswith('_'):
                        continue
                    try:
                        member = inspect.getattr_static(value, name)
                        if isinstance(member, (classmethod, staticmethod)):
                            member = getattr(value, name)
                        if callable(member) or isinstance(member, property):
                            add(symbol + '.' + name, member, methods=False)
                    except (AttributeError, TypeError, ValueError) as error:
                        issues.append({'symbol': symbol + '.' + name, 'error': str(error)[:200]})

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                exports = getattr(module, '__all__', None)
                names = exports or [n for n in vars(module) if not n.startswith('_')]
                for name in names:
                    if name.startswith('_'):
                        continue
                    value = getattr(module, name)
                    origin = getattr(value, '__module__', '') or ''
                    # Ignore imported helpers from unrelated packages and typing.
                    prefix = module_name.split('.')[0]
                    if (inspect.isclass(value) or callable(value)) and (exports or origin.split('.')[0] == prefix):
                        add(module_name + '.' + name, value)
            except (ImportError, AttributeError, TypeError, ValueError) as error:
                issues.append({'module': module_name, 'error': str(error)[:300]})
        for symbol in SYMBOLS.get(package, []):
            try:
                add(symbol, resolve(symbol))
            except (ImportError, AttributeError, TypeError, ValueError) as error:
                issues.append({'symbol': symbol, 'error': str(error)[:300]})
        print(f'{package} {packages[package]["version"]}: {len(records) - before} records', flush=True)

    for symbol, targets in FORWARDED_TO.items():
        if symbol not in records:
            continue
        forwarded = {}
        for target in targets:
            value = resolve(target)
            signature, _, parameters, _ = signature_info(value)
            forwarded[target] = {'signature': signature, 'parameters': parameters, 'source': source_info(value)}
        records[symbol]['kwargs_forwarded_to'] = forwarded
    for symbol, validators in VALIDATED_KWARGS.items():
        owner = resolve(symbol.rsplit('.', 1)[0])
        keys = {}
        for method, variable in validators:
            for name in named_dict_keys(getattr(owner, method), variable):
                if not name.startswith('_'):
                    keys[name] = {'source': 'installed_validator', 'method': method}
        records[symbol]['validated_kwargs'] = keys
    # The installed preset YAML describes nested *_params that signatures cannot.
    import yaml
    symbol = 'lightautoml.automl.presets.tabular_presets.TabularAutoML'
    owner = resolve(symbol)
    path = Path(inspect.getsourcefile(owner)).with_name(owner._default_config_path)
    records[symbol]['configuration_defaults'] = yaml.safe_load(path.read_text(encoding='utf-8'))
    records[symbol]['configuration_source'] = {'file': 'lightautoml/automl/presets/' + path.name,
                                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    # This binding asks the installed native engine for its recognized option
    # names/aliases. Applicability (binary vs multiclass, etc.) is still conditional.
    from lightgbm.basic import _ConfigAliases
    aliases = _ConfigAliases._get_all_param_aliases()
    for symbol in ('lightgbm.LGBMClassifier', 'lightgbm.LGBMRegressor', 'lightgbm.LGBMRanker'):
        records[symbol]['native_parameter_aliases'] = aliases
        records[symbol]['native_parameter_source'] = 'Installed LightGBM engine: LGBM_DumpParamAliases'
    return packages, records, issues


def write_reference(path, packages, records, issues):
    header = ('"""Generated offline API facts; rebuild with build_ml_reference.py.\n'
              'Signatures/docstring excerpts retain their packages\' licenses and attribution below.\n'
              'No imports of the documented ML libraries occur when this data module is loaded.\n"""\n\n')
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(header)
        for name, value in [('BUILT_UTC', datetime.now(timezone.utc).isoformat()), ('PACKAGES', packages),
                            ('BUILD_ISSUES', issues)]:
            stream.write(name + ' = ' + pprint.pformat(value, width=110, sort_dicts=True) + '\n\n')
        # Store inherited/re-exported API facts once, retaining every public name.
        canonical, aliases = {}, {}
        stream.write('ENTRIES = {\n')
        for symbol, record in sorted(records.items()):
            identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            if identity in canonical:
                aliases[symbol] = canonical[identity]
                continue
            canonical[identity] = symbol
            text = pprint.pformat(record, width=110, sort_dicts=True)
            stream.write('    ' + repr(symbol) + ': ' + text.replace('\n', '\n    ') + ',\n')
        stream.write('}\n\nENTRY_ALIASES = ' + pprint.pformat(aliases, width=110, sort_dicts=True) + '\n')
        stream.write('ENTRIES.update({name: ENTRIES[target] for name, target in ENTRY_ALIASES.items()})\n')


def verify(path):
    namespace = {}
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
    records, packages = namespace['ENTRIES'], namespace['PACKAGES']
    checks = {
        'flaml.AutoML.fit': {'metric', 'time_budget', 'log_file_name', 'estimator_list'},
        'autogluon.tabular.TabularPredictor.fit': {'train_data', 'time_limit', 'hyperparameters'},
        'catboost.CatBoostClassifier': {'auto_class_weights', 'thread_count', 'iterations'},
        'lightautoml.automl.presets.tabular_presets.TabularAutoML': {'task', 'timeout', 'cpu_limit'},
        'xgboost.XGBClassifier.fit': {'X', 'y', 'sample_weight', 'eval_set'},
        'optuna.study.Study.optimize': {'n_trials', 'timeout', 'n_jobs'},
    }
    for symbol, required in checks.items():
        actual = set(records[symbol]['parameters'])
        if not required <= actual:
            raise ValueError(f'{symbol}: missing signature parameters {sorted(required - actual)}')
    for symbol in ('catboost.CatBoostClassifier', 'catboost.CatBoostClassifier.fit', 'catboost.train'):
        if 'time_left' in records[symbol]['parameters']:
            raise ValueError('Review time_left guidance against the installed CatBoost API')
    ag_kwargs = records['autogluon.tabular.TabularPredictor.fit']['validated_kwargs']
    if not {'num_bag_folds', 'num_stack_levels', 'calibrate'} <= ag_kwargs.keys():
        raise ValueError('Incomplete AutoGluon validator extraction')
    if not {'num_bag_folds', 'hyperparameter_tune_kwargs', 'ag_args_fit'} <= records['autogluon.tabular.TabularPredictor.fit']['documented_kwargs'].keys():
        raise ValueError('Incomplete nested AutoGluon kwargs documentation')
    native = records['lightgbm.LGBMClassifier']['native_parameter_aliases']
    if not {'bagging_freq', 'is_unbalance', 'max_bin'} <= native.keys():
        raise ValueError('Incomplete native LightGBM parameter extraction')
    for package, info in packages.items():
        if metadata.version(package) != info['version']:
            raise ValueError(f'{package}: installed version changed during generation')
    from ml_reference_notes import GUIDANCE
    missing = sorted(symbol for symbol in GUIDANCE if symbol not in records
                     and not any(name.startswith(symbol + '.') for name in records))
    if missing:
        raise ValueError(f'Guidance without a matching reference: {missing}')
    print(json.dumps({'verified_entries': len(records), 'packages': len(packages),
                      'build_issues': len(namespace['BUILD_ISSUES']),
                      'documented_kwargs': sum(bool(r['documented_kwargs']) for r in records.values()),
                      'forwarded_kwargs': sum(bool(r.get('kwargs_forwarded_to')) for r in records.values()),
                      'bytes': path.stat().st_size}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('ml_reference_data.py'))
    parser.add_argument('--verify', action='store_true', help='Verify an existing reference without rebuilding it')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      MPLBACKEND='Agg', OMP_NUM_THREADS='2')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        packages, records, issues = collect()
    write_reference(args.output, packages, records, issues)
    verify(args.output)


if __name__ == '__main__':
    main()
