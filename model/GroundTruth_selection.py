def groundtruth_selection(groundtruth_kind, groundtruth_options, field, BF_mesh, s_grid):
    """
    This function helps to find the correct ground truth to your input strings
    :param ground truth_kind: str: e.g. "MVN" for Multivariate Normal
    :param ground truth_options: dict with options like initial mean and sigma
    :return:
    """
    all_groundtruths = ["Circular"]
    if groundtruth_kind == "Circular":
        from utils.torch_funcs.function_circular_inclusion import function_circular_inclusion
        return function_circular_inclusion(field, BF_mesh, groundtruth_options)
    elif groundtruth_kind == "Rectangular":
        from utils.torch_funcs.function_rectangular_inclusion import function_rectangular_inclusion
        return function_rectangular_inclusion(s_grid, groundtruth_options)
    
    # raise an Error, when we do not find a fit ground truth
    raise RuntimeError(("Please select an implemented ground truth for x. You selected \"{}\". \n"
                        "Currently available are: {}. \n").format(groundtruth_kind,
                                                                  ', '.join(map(str, all_groundtruths))))