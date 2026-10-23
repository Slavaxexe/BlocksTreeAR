"""Translate typed model-search domains to each framework's own API."""
def model_settings(plan, smoke=False):
    if plan.engine == 'flaml':
        from flaml import tune
        defaults = ['lgbm', 'catboost', 'xgboost', 'extra_tree']
        def domain(value):
            if value.kind == 'categorical':
                return tune.choice(value.choices)
            if value.kind == 'int':
                function = tune.lograndint if value.log else tune.randint
                return function(int(value.low), int(value.high) + 1)
            return (tune.loguniform if value.log else tune.uniform)(value.low, value.high)
    else:
        from autogluon.common import space
        defaults = ['GBM', 'CAT', 'XGB', 'RF', 'XT']
        def domain(value):
            if value.kind == 'categorical':
                return space.Categorical(*value.choices)
            if value.kind == 'int':
                if value.log:
                    raise ValueError('Use categorical choices or a non-log integer domain with AutoGluon')
                return space.Int(int(value.low), int(value.high))
            return space.Real(value.low, value.high, log=value.log)
    models = plan.models or list(dict.fromkeys([*defaults, *plan.model_parameters, *plan.search_space]))
    parameters = {name: dict(plan.model_parameters.get(name, {})) for name in models}
    for name, fields in plan.search_space.items():
        if name not in parameters:
            raise ValueError(f'Search domain names an unselected model: {name}')
        parameters[name].update({key: domain(value) for key, value in fields.items()})
    if smoke:
        return (['lgbm'], {}) if plan.engine == 'flaml' else (['GBM'], {'GBM': {'num_boost_round': 20}})
    if plan.engine == 'flaml':
        parameters = {name: {key: {'domain': value} for key, value in fields.items()}
                      for name, fields in parameters.items() if fields}
        # FLAML's single-trial path reads init_value, not the search domain.
        for name, fields in parameters.items():
            for key, setting in fields.items():
                if key not in plan.search_space.get(name, {}):
                    setting['init_value'] = setting['domain']
    return models, parameters
