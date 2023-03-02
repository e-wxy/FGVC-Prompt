
def make_params_dict(params_string: str, assign=" = ", divide=", "):
    params_dict = dict(param_string.split(assign) for param_string in params_string.split(divide))

    return params_dict