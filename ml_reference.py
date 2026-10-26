"""Local API lookup: portable root-level Python files, no model/network calls.

python ml_reference.py search "AutoGluon fit time_limit"
python ml_reference.py lookup catboost.CatBoostClassifier.fit
python ml_reference.py lookup flaml.AutoML.fit --parameter metric
python ml_reference.py catalog
"""

import argparse
import ast
from functools import lru_cache
from importlib import metadata
import json
import re

from ml_reference_notes import ALIASES, GUIDANCE, PRIMARY, REVIEWED_VERSIONS


@lru_cache(maxsize=1)
def data():
    from ml_reference_data import ENTRIES, PACKAGES
    return ENTRIES, PACKAGES


@lru_cache(maxsize=None)
def installed_version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def catalog():
    entries, packages = data()
    return {name: {'reference_version': info['version'], 'installed_version': installed_version(name),
                   'entries': sum(r['package'] == name for r in entries.values()),
                   'primary_symbols': PRIMARY.get(name, [])}
            for name, info in packages.items()}


def guidance(symbol):
    # Module-level advice is inherited; class/method-specific advice is additive.
    entries, _ = data()
    parts = symbol.split('.')
    owners = []
    for i in range(1, len(parts) + 1):
        prefix = '.'.join(parts[:i])
        if prefix in entries:
            owners.append(entries[prefix])
    return [{'symbol': name, **value} for name, value in GUIDANCE.items()
            if symbol == name or symbol.startswith(name + '.')
            or (name in entries and any(entries[name] is owner for owner in owners))]


def lookup(symbol, *, parameter=None):
    entries, packages = data()
    record = entries.get(symbol)
    if record is None:
        return {'symbol': symbol, 'status': 'not_indexed', 'matches': search(symbol, limit=5),
                'note': 'Missing from this index does not mean the library API is unsupported.'}
    package = record['package']
    actual = installed_version(package)
    result = {'symbol': symbol, 'status': 'available' if actual else 'package_not_installed',
              'reference_version': packages[package]['version'], 'installed_version': actual,
              'version_matches': actual == packages[package]['version'], **record,
              'reviewed_guidance': guidance(symbol)}
    if parameter is not None:
        result['requested_parameter'] = parameter
        result['parameter'] = record['parameters'].get(parameter)
        result['documentation'] = record['documented_parameters'].get(parameter)
        result['validated_kwarg'] = record.get('validated_kwargs', {}).get(parameter)
        result['delegated_parameter'] = {name: target['parameters'][parameter]
            for name, target in record.get('kwargs_forwarded_to', {}).items() if parameter in target['parameters']}
        result['native_parameter'] = {name: aliases for name, aliases in record.get('native_parameter_aliases', {}).items()
                                      if parameter == name or parameter in aliases}
        configuration = record.get('configuration_defaults', {})
        for part in parameter.split('.'):
            if not isinstance(configuration, dict) or part not in configuration:
                break
            configuration = configuration[part]
        else:
            result['configuration_parameter'] = configuration
        if result['parameter']:
            result['parameter_status'] = 'explicit'
        elif result['validated_kwarg']:
            result['parameter_status'] = 'validated_kwarg'
        elif result['delegated_parameter']:
            result['parameter_status'] = 'delegated'
        elif result['native_parameter']:
            result['parameter_status'] = 'native_conditional'
        elif result['documentation']:
            result['parameter_status'] = 'documented'
        elif 'configuration_parameter' in result:
            result['parameter_status'] = 'configuration_default'
        elif record['signature'] is None:
            result['parameter_status'] = 'unknown_signature'
        elif record['accepts_kwargs']:
            result['parameter_status'] = 'unknown_kwargs'
        else:
            result['parameter_status'] = 'not_in_signature'
        result.pop('documented_parameters')
        result.pop('documented_kwargs')
    if package in REVIEWED_VERSIONS:
        result['guidance_reviewed_version'] = REVIEWED_VERSIONS[package]
    return result


def search(query, *, package=None, limit=8):
    entries, _ = data()
    terms = set(re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*', query.casefold()))
    reviewed = {id(entries[symbol]): ' '.join(value['notes']).casefold()
                for symbol, value in GUIDANCE.items() if symbol in entries}
    scored = []
    for symbol, record in entries.items():
        if package is not None and record['package'] != package:
            continue
        name = symbol.casefold()
        short = symbol.rsplit('.', 1)[-1].casefold()
        score = sum(8 if term == short else 3 if term in name else 0 for term in terms)
        score += sum(term in record['summary'].casefold() for term in terms)
        score += 4 * sum(term in reviewed.get(id(record), '') for term in terms)
        if query.casefold() == name:
            score += 100
        if score:
            scored.append((score, symbol))
    result, seen = [], set()
    for _, symbol in sorted(scored, key=lambda item: (-item[0], item[1])):
        if id(entries[symbol]) in seen:
            continue
        seen.add(id(entries[symbol]))
        result.append({'symbol': symbol, 'signature': entries[symbol]['signature'],
                       'summary': entries[symbol]['summary'][:240]})
        if len(result) == limit:
            break
    return result


def imported_symbols(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    bindings, symbols = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != '*':
                    symbol = node.module + '.' + alias.name
                    bindings[alias.asname or alias.name] = symbol
                    symbols.append(symbol)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
    for node in ast.walk(tree):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if parts and isinstance(node, ast.Name) and node.id in bindings:
            symbols.append('.'.join([bindings[node.id], *reversed(parts)]))
    return list(dict.fromkeys(symbols))


@lru_cache(maxsize=1)
def symbol_index():
    entries, _ = data()
    qualified, classes = {}, {}
    for symbol, record in entries.items():
        qualified[symbol.casefold()] = symbol
        name = symbol.rsplit('.', 1)[-1]
        if record['kind'] == 'class' and name[:1].isupper():
            classes.setdefault(name.casefold(), []).append(symbol)
    return qualified, classes


def mentioned_symbols(text):
    entries, _ = data()
    qualified, classes = symbol_index()
    packages = [package for package, aliases in ALIASES.items()
                if any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text, re.IGNORECASE)
                       for alias in [package, *aliases])]
    symbols = [qualified[token.casefold()] for token in re.findall(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b', text)
               if token.casefold() in qualified]
    for token in dict.fromkeys(re.findall(r'\b[A-Z]\w*\b', text)):
        candidates = classes.get(token.casefold(), [])
        explicit = [symbol for symbol in candidates if entries[symbol]['package'] in packages]
        candidates = explicit or candidates
        # A bare Categorical/DataFrame name is not evidence for every library exporting it.
        versions = list(dict.fromkeys(id(entries[symbol]) for symbol in candidates))
        named = re.search(r'[\"\']' + re.escape(token) + r'[\"\']|\b' + re.escape(token) + r'\s*\(', text)
        camel_case = any(char.isupper() for char in token[1:])
        if candidates and (explicit or len(versions) == 1 and (named or camel_case)):
            for identity in versions:
                aliases = [symbol for symbol in candidates if id(entries[symbol]) == identity]
                symbols.append(min(aliases, key=lambda symbol: (symbol.count('.'), len(symbol), symbol)))
    for package in packages:
        if not any(entries[symbol]['package'] == package for symbol in symbols):
            symbols.extend(PRIMARY.get(package, []))
    return list(dict.fromkeys(symbol for symbol in symbols if symbol in entries))


def parameter_reference(symbol, parameter):
    """Return one parameter's facts without repeating the entire API card."""
    record = lookup(symbol, parameter=parameter)
    keys = ('symbol', 'package', 'status', 'reference_version', 'installed_version', 'version_matches',
            'accepts_kwargs', 'kwargs_complete', 'requested_parameter', 'parameter_status', 'parameter',
            'documentation', 'validated_kwarg', 'delegated_parameter', 'native_parameter',
            'configuration_parameter', 'guidance_reviewed_version')
    result = {key: record[key] for key in keys if record.get(key) is not None}
    if record['status'] == 'not_indexed':
        return record
    # A constructor may forward kwargs while the named option belongs to a public method.
    if record['parameter_status'] in {'not_in_signature', 'unknown_kwargs'}:
        entries, _ = data()
        related = []
        for method in ('fit', 'predict', 'fit_predict', 'transform'):
            child = symbol + '.' + method
            if child in entries:
                found = lookup(child, parameter=parameter)
                if found['parameter_status'] in {'explicit', 'documented', 'validated_kwarg'}:
                    related.append({'symbol': child, 'parameter': found['parameter'],
                                    'documentation': found['documentation']})
        if related:
            result['declared_on_methods'] = related
    notes = []
    for item in record['reviewed_guidance']:
        matched = [note for note in item['notes']
                   if re.search(r'(?<!\w)' + re.escape(parameter) + r'(?!\w)', note, re.IGNORECASE)]
        if matched:
            notes.append({'symbol': item['symbol'], 'notes': matched})
    if notes:
        result['reviewed_guidance'] = notes
    return result


def compact_record(symbol, query, *, brief=False):
    record = lookup(symbol)
    if record['status'] == 'not_indexed':
        return record
    if brief:
        keys = ('symbol', 'package', 'status', 'reference_version', 'installed_version', 'version_matches',
                'accepts_kwargs', 'kwargs_complete', 'guidance_reviewed_version')
        result = {key: record[key] for key in keys if key in record}
        result['signature'] = record.get('call_signature') or record.get('doc_signature') or record['signature']
        result['parameter_help'] = {}
        names = dict.fromkeys([*record['parameters'], *record['documented_parameters'],
                               *record.get('validated_kwargs', {})])
        for name in names:
            if len(name) > 1 and re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', query, re.IGNORECASE):
                fact = parameter_reference(symbol, name)
                help = {'status': fact['parameter_status']}
                if fact.get('parameter'):
                    help['parameter'] = fact['parameter']
                if fact.get('documentation'):
                    doc = fact['documentation']
                    help['declaration'] = doc['declaration']
                    help['description'] = doc['description'].split('\n', 1)[0]
                for key in ('validated_kwarg', 'delegated_parameter', 'native_parameter', 'configuration_parameter'):
                    if fact.get(key):
                        help[key] = fact[key]
                result['parameter_help'][name] = help
        result['reviewed_guidance'] = []
        for item in record['reviewed_guidance']:
            notes = [note for note in item['notes'] if any(
                re.search(r'(?<!\w)' + re.escape(token) + r'(?!\w)', note, re.IGNORECASE)
                for token in re.findall(r'\b[a-zA-Z_]\w*\b', query) if '_' in token)]
            if notes:
                result['reviewed_guidance'].append({'symbol': item['symbol'], 'notes': notes})
        return result
    result = {key: record[key] for key in ('symbol', 'package', 'reference_version', 'installed_version',
                                          'version_matches', 'signature', 'accepts_kwargs', 'kwargs_complete')}
    result['summary'] = record['summary'][:500]
    if 'guidance_reviewed_version' in record:
        result['guidance_reviewed_version'] = record['guidance_reviewed_version']
        result['guidance_version_matches'] = record['guidance_reviewed_version'] == record['installed_version']
    result['signature'] = record.get('call_signature') or record.get('doc_signature') or record['signature']
    result['signature_source'] = 'inspect' if record['signature'] else 'docstring' if record.get('doc_signature') else 'unavailable'
    names = record['documented_parameters']
    result['documented_parameter_names'] = list(names)
    result['parameter_help'] = {name: {'declaration': item['declaration'][:180],
                                     'description': item['description'][:500]}
        for name, item in names.items()
        if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', query, re.IGNORECASE)}
    if record.get('validated_kwargs'):
        result['validated_kwargs'] = list(record['validated_kwargs'])
    if record.get('kwargs_forwarded_to'):
        result['kwargs_forwarded_to'] = record['kwargs_forwarded_to']
    if record.get('native_parameter_aliases'):
        result['native_parameter_aliases'] = {name: aliases for name, aliases in record['native_parameter_aliases'].items()
            if any(re.search(r'(?<!\w)' + re.escape(key) + r'(?!\w)', query) for key in [name, *aliases])}
    if record.get('configuration_defaults'):
        defaults = record['configuration_defaults']
        result['configuration_sections'] = list(defaults)
        result['configuration_defaults'] = {name: value for name, value in defaults.items()
                                            if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', query)}
    result['reviewed_guidance'] = record['reviewed_guidance']
    result['source'] = record['source']
    return result


def context_for(query, *, code='', focus=None, fallback=None, max_chars=18000, known=None):
    """Prefer the requested model over inherited code; resolve names before packing cards."""
    entries, packages = data()
    query_text = query if isinstance(query, str) else json.dumps(query, ensure_ascii=False)
    focus_text = json.dumps(focus, ensure_ascii=False) if focus is not None else ''
    focused = mentioned_symbols(focus_text)
    requested = mentioned_symbols(query_text)
    inherited = mentioned_symbols(json.dumps(fallback, ensure_ascii=False)) if fallback is not None else []
    imported = [symbol for symbol in imported_symbols(code) if symbol in entries]
    primary = list(dict.fromkeys(focused + requested)) or imported or inherited
    secondary = requested + imported
    text = focus_text + '\n' + query_text + '\n' + code
    expanded, seen, priority = [], set(), set()
    for group_number, group in enumerate((primary, secondary)):
        selected = []
        for symbol in group:
            if id(entries[symbol]) not in seen:
                seen.add(id(entries[symbol]))
                selected.append(symbol)
        # In an ensemble both selected estimators precede their auxiliary methods.
        expanded.extend(selected)
        for symbol in selected:
            for method in ('fit', 'fit_predict', 'predict', 'optimize'):
                child = symbol + '.' + method
                if child in entries and id(entries[child]) not in seen:
                    seen.add(id(entries[child]))
                    expanded.append(child)
        if group_number == 0:
            priority.update(expanded)
    result = {'source': 'installed_signatures_and_docstrings_plus_reviewed_guidance',
              'rules': ['A missing entry or parameter is not proof of rejection when kwargs are accepted.',
                        'Docstring kwargs are documented names, not an exhaustive validator.',
                        'Reference/version mismatches require checking the actual installed API.',
                        'Library documentation does not prove training completed or explain a score change.'],
              'packages': {}, 'entries': [], 'omitted_symbols': []}
    cards = {}
    for symbol in dict.fromkeys(expanded):
        card = compact_record(symbol, text, brief=True)
        for (saved_symbol, parameter), fact in (known or {}).items():
            if (entries.get(saved_symbol) is entries[symbol]
                    and fact.get('installed_version') == card.get('installed_version')
                    and re.search(r'(?<!\w)' + re.escape(parameter) + r'(?!\w)', text, re.IGNORECASE)):
                card['parameter_help'][parameter] = {key: value for key, value in fact.items() if key in {
                    'parameter_status', 'parameter', 'documentation', 'validated_kwarg', 'delegated_parameter',
                    'native_parameter', 'configuration_parameter', 'declared_on_methods', 'reviewed_guidance'}}
                card['parameter_help'][parameter]['source'] = 'previous_lookup'
        cards[symbol] = card

    def size():
        return len(json.dumps(result, ensure_ascii=False))

    # Reserve space for learned facts. Otherwise constructor docs can hide fit() and its kwargs.
    learned = [help for card in cards.values() for help in card['parameter_help'].values()
               if help.get('source') == 'previous_lookup']
    reserve = min(max_chars // 3, sum(len(json.dumps(help, ensure_ascii=False)) + 60 for help in learned))
    selected = {}
    for symbol, card in cards.items():
        if symbol not in priority:
            continue
        base = {key: value for key, value in card.items() if key not in {'parameter_help', 'reviewed_guidance'}}
        base['parameter_help'] = {}
        result['entries'].append(base)
        package = entries[symbol]['package']
        result['packages'][package] = {'reference_version': packages[package]['version'],
                                       'installed_version': installed_version(package)}
        if size() > max_chars - reserve:
            result['entries'].pop()
            result['omitted_symbols'].append(symbol)
        else:
            selected[symbol] = base

    parameters = [(symbol, name, help) for symbol, card in cards.items() if symbol in selected
                  for name, help in card['parameter_help'].items()]
    parameters.sort(key=lambda row: (row[2].get('source') != 'previous_lookup',
                                    '.fit' not in row[0]))
    for symbol, name, help in parameters:
        selected[symbol]['parameter_help'][name] = help
        if size() > max_chars:
            selected[symbol]['parameter_help'].pop(name)

    seen_notes = set()
    for symbol in selected:
        for guidance in cards[symbol]['reviewed_guidance']:
            notes = [note for note in guidance['notes'] if note not in seen_notes]
            if not notes:
                continue
            item = {'symbol': guidance['symbol'], 'notes': notes}
            selected[symbol].setdefault('reviewed_guidance', []).append(item)
            if size() > max_chars:
                selected[symbol]['reviewed_guidance'].pop()
            else:
                seen_notes.update(notes)
    # General imports are optional: keep the changed APIs and their parameter facts first.
    for symbol, card in cards.items():
        if symbol in priority:
            continue
        base = {key: value for key, value in card.items() if key not in {'parameter_help', 'reviewed_guidance'}}
        result['entries'].append(base)
        if size() > max_chars:
            result['entries'].pop()
            result['omitted_symbols'].append(symbol)
    while result['entries'] and size() > max_chars:
        result['omitted_symbols'].append(result['entries'].pop()['symbol'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['catalog', 'search', 'lookup'])
    parser.add_argument('query', nargs='?')
    parser.add_argument('--package')
    parser.add_argument('--parameter')
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    if args.command != 'catalog' and not args.query:
        parser.error('search/lookup require a query or symbol')
    if args.command == 'catalog':
        result = catalog()
    elif args.command == 'search':
        result = search(args.query, package=args.package, limit=args.limit)
    else:
        result = lookup(args.query, parameter=args.parameter)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
