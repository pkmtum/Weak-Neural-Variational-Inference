def posterior_selection_X(posterior_x_kind, posterior_x_options):
    """
    This function helps to find the correct posterior to your input strings
    :param posterior_kind: str: e.g. "MVN" for Multivariate Normal
    :param posterior_options: dict with options like initial mean and sigma
    :return:
    """
    all_posteriors = ["MVN", "DiagMVN", "Delta", "CNNDelta", "CNNDiagMVN", "FFNDelta", "ReducedMVN", "FFNMVN", "FFNReducedMVN"]
    if posterior_x_kind == "MVN":
        from model.Posteriors.class_MVNPosterior_X import MVNPosterior_X
        return MVNPosterior_X(posterior_x_options)
    elif posterior_x_kind == "DiagMVN":
        from model.Posteriors.class_DiagMVNPosterior_X import DiagMVNPosterior_X
        return DiagMVNPosterior_X(posterior_x_options)
    elif posterior_x_kind == "Delta":
        from model.Posteriors.class_DeltaPosterior_X import DeltaPosteriorX
        return DeltaPosteriorX(posterior_x_options)
    elif posterior_x_kind == "CNNDelta":
        from model.Posteriors.class_CNNDeltaPosterior_X import CNNDeltaPosteriorX
        return CNNDeltaPosteriorX(posterior_x_options)
    elif posterior_x_kind == "CNNDiagMVN":
        from model.Posteriors.class_CNNDiagMVNPosterior_X import CNNDiagMVNPosteriorX
        return CNNDiagMVNPosteriorX(posterior_x_options)
    elif posterior_x_kind == "FFNDelta":
        from model.Posteriors.class_FFNDeltaPosterior_X import FFNDeltaPosteriorX
        return FFNDeltaPosteriorX(posterior_x_options)
    elif posterior_x_kind == "FFNMVN":
        from model.Posteriors.class_FFNMVNPosterior_X import FFNMVNPosteriorX
        return FFNMVNPosteriorX(posterior_x_options)
    elif posterior_x_kind == "ReducedMVN":
        from model.Posteriors.class_ReducedMVNPosterior_X import ReducedMVNPosterior_x
        return ReducedMVNPosterior_x(posterior_x_options)
    elif posterior_x_kind == "FFNReducedMVN":
        from model.Posteriors.class_FFNReducedMVNPosterior_X import FFNReducedMVNPosteriorX
        return FFNReducedMVNPosteriorX(posterior_x_options)

    # raise an Error, when we do not find a fit posterior
    raise RuntimeError(("Please select an implemented prior for x. You selected \"{}\". \n"
                        "Currently available are: {}. \n").format(posterior_x_kind,
                                                                  ', '.join(map(str, all_posteriors))))

def posterior_selection_Y(posterior_y_kind, posterior_y_options):
    """
    This function helps to find the correct posterior to your input strings
    :param posterior_kind: str: e.g. "MVN" for Multivariate Normal
    :param posterior_options: dict with options like initial mean and sigma
    :return:
    """
    all_posteriors = ["Delta", "Delta_given_X", "MVN", "DiagMVN", "ReducedMVN"]
    if posterior_y_kind == "Delta_given_X":
        from model.Posteriors.class_DeltaPosterior_Y_given_X import DeltaPosteriorY_given_X
        return DeltaPosteriorY_given_X(posterior_y_options)
    elif posterior_y_kind == "MVN":
        from model.Posteriors.class_MVNPosterior_Y import MVNPosterior_y
        return MVNPosterior_y(posterior_y_options)
    elif posterior_y_kind == "DiagMVN":
        from model.Posteriors.class_DiagMVNPosterior_Y import DiagMVNPosterior_Y
        return DiagMVNPosterior_Y(posterior_y_options)
    elif posterior_y_kind == "Delta":
        from model.Posteriors.class_DeltaPosterior_Y import DeltaPosteriorY
        return DeltaPosteriorY(posterior_y_options)
    elif posterior_y_kind == "ReducedMVN":
        from model.Posteriors.class_ReducedMVNPosterior_Y import ReducedMVNPosterior_y
        return ReducedMVNPosterior_y(posterior_y_options)

    # raise an Error, when we do not find a fit posterior
    raise RuntimeError(("Please select an implemented prior for x. You selected \"{}\". \n"
                        "Currently available are: {}. \n").format(posterior_y_kind,
                                                                  ', '.join(map(str, all_posteriors))))


def posterior_selection_lambda(lambda_posterior_kind, lambda_posterior_options):
    posteriors = ["Gamma", "Constant"]
    if lambda_posterior_kind == "Gamma":
        from model.Posteriors.class_LambdaGamma_posterior import LambdaGamma_posterior
        return LambdaGamma_posterior(lambda_posterior_options)
    elif lambda_posterior_kind == "Constant":
        from model.Posteriors.class_LambdaConstant_posterior import LambdaConstant_posterior
        return LambdaConstant_posterior(lambda_posterior_options)

    raise RuntimeError(("Please select an implemented posterior for lambda. You selected \"{}\" . \n"
                        "Currently available for are: {}.").format(lambda_posterior_kind,
                                                                   ', '.join(map(str, posteriors))))
