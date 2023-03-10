import ast

def make_params_dict(params_string: str, assign=" = ", divide="; ") -> dict:
    """ Make dict of params passed to optimizer/scheduler

    Args:
        params_string (str): string of params in config
        assign (str): the denote for assignment. Defaults to " = ".
        divide (str): the denote for diving param groups. Defaults to "; ".

    Returns:
        dict: params_dict

    Example:
        params_string: "lr = 5e-4; weight_decay = 1e-4; eps = 1e-8; betas = (0.9, 0.999)"
        params_dict: {'lr': 0.05, 'weight_decay': 0.0001, 'eps': 1e-08, 'betas': (0.9, 0.999)}
        
    """
    if params_string == '':
        return None
    params_dict = dict(param_string.split(assign) for param_string in params_string.split(divide))
    for key in params_dict:
        # int
        if params_dict[key].isdigit():
            params_dict[key] = int(params_dict[key])
        # tuple or list
        elif params_dict[key].startswith('(') or params_dict[key].startswith('['):
            params_dict[key] = ast.literal_eval(params_dict[key])
        # float
        else:
            params_dict[key] = float(params_dict[key])
    return params_dict