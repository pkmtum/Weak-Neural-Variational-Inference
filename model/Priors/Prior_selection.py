from utils.function_load_from_list import load_from_list

def prior_x_selection(prior_x_kind, prior_x_options):
    prior_classes = {
        "Normal": "model.Priors.class_PriorX_Normal.PriorXNormal",
        "MVN": "model.Priors.class_PriorX_MvN.PriorXMvN",
        "FieldNormal": "model.Priors.class_PriorX_FieldNormal.PriorXFieldNormal"
    }
    return load_from_list(prior_classes, prior_x_kind, prior_x_options)


def prior_y_selection(prior_y_kind, prior_y_options):
    prior_classes = {
        "MVN": "model.Priors.class_PriorY_MvN.PriorYMvN"
    }
    return load_from_list(prior_classes, prior_y_kind, prior_y_options)


def prior_lambda_selection(prior_lambda_kind, prior_lambda_options):
    prior_classes = {
        "Gamma": "model.Priors.class_PriorLambda_Gamma.PriorLambdaGamma",
        "MVN": "model.Priors.class_PriorLambda_MVN.PriorLambdaMVN"
    }
    return load_from_list(prior_classes, prior_lambda_kind, prior_lambda_options)
