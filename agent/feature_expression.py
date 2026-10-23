"""Composable vector expressions with statistics learned on fit only."""
import ast
import operator

import numpy as np
import pandas as pd


BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
          ast.Pow: operator.pow, ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
FUNCTIONS = {'abs': np.abs, 'log1p': np.log1p, 'log': np.log, 'exp': np.exp,
             'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos, 'tanh': np.tanh,
             'minimum': np.minimum, 'maximum': np.maximum, 'where': np.where,
             'clip': np.clip, 'isna': pd.isna, 'notna': pd.notna,
             'fillna': lambda value, default: value.fillna(default),
             'length': lambda value: value.astype('string').str.len(),
             'lower': lambda value: value.astype('string').str.lower(),
             'contains': lambda value, text: value.astype('string').str.contains(text, regex=False, na=False),
             'concat': lambda a, b: (a.astype('string').fillna('') if isinstance(a, pd.Series) else str(a)) +
                                  (b.astype('string').fillna('') if isinstance(b, pd.Series) else str(b))}


class FeatureExpression:
    def __init__(self, expression, columns):
        self.tree = ast.parse(expression, mode='eval').body
        self.columns = columns
        self.statistics = {}

    def fit(self, frame):
        self._evaluate(self.tree, frame, True)
        return self

    def transform(self, frame):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            values = self._evaluate(self.tree, frame, False)
        return pd.Series(values, index=frame.index)

    def _evaluate(self, node, frame, fit):
        evaluate = lambda child: self._evaluate(child, frame, fit)
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
            return node.value
        if isinstance(node, ast.Name) and node.id.startswith('x') and node.id[1:].isdigit():
            return frame[self.columns[int(node.id[1:])]]
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            return BINARY[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return np.logical_not(value)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            function = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            values = [evaluate(value) for value in node.values]
            result = values[0]
            for value in values[1:]:
                result = function(result, value)
            return result
        if isinstance(node, ast.Compare):
            result, previous = True, evaluate(node.left)
            for operation, right in zip(node.ops, node.comparators):
                current = evaluate(right)
                result = np.logical_and(result, COMPARE[type(operation)](previous, current))
                previous = current
            return result
        if isinstance(node, ast.IfExp):
            return np.where(evaluate(node.test), evaluate(node.body), evaluate(node.orelse))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not node.keywords
                and node.func.attr in {'isna', 'notna', 'fillna'}):
            return FUNCTIONS[node.func.attr](evaluate(node.func.value), *(evaluate(arg) for arg in node.args))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name == 'col':
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or node.args[0].value not in self.columns:
                    raise ValueError('col() must name one of the declared feature columns')
                return frame[node.args[0].value]
            args = [evaluate(arg) for arg in node.args]
            if name in {'frequency', 'group_mean', 'group_std', 'group_count'}:
                key = ast.dump(node)
                groups = pd.Series(args[0], index=frame.index).astype('string').fillna('__missing__')
                if fit:
                    if name == 'frequency':
                        self.statistics[key] = (groups.value_counts(normalize=True), 0.)
                    elif name == 'group_count':
                        self.statistics[key] = (groups.value_counts(), 0.)
                    else:
                        values = pd.to_numeric(pd.Series(args[1], index=frame.index), errors='coerce')
                        method = name.removeprefix('group_')
                        self.statistics[key] = (getattr(values.groupby(groups), method)(), getattr(values, method)())
                mapping, default = self.statistics[key]
                return groups.map(mapping).fillna(default)
            if name in FUNCTIONS:
                return FUNCTIONS[name](*args)
        raise ValueError('Unsupported feature expression; use columns, arithmetic, conditions and declared vector functions')
