from input import *

"""
This file mainly exists to make sure you don't fuck up in the input.py file and deals with things that are
added automatically to e.g. the dicts for the options of the prior and posterior.
"""


def input_preparation(prior_x_options, posterior_options, dim_x):
    prior_x_options["dim"] = dim_x
    posterior_options["dim_x"] = dim_x
    return prior_x_options, posterior_options


def gamma_stuff(lambda_posterior_kind, lambda_posterior_options, prior_lambda_options):
    if lambda_posterior_kind == "Gamma":
        lambda_posterior_options["a_0"] = prior_lambda_options["a_0"]
        lambda_posterior_options["b_0"] = prior_lambda_options["b_0"]
    elif lambda_posterior_kind == "Constant":
        lambda_posterior_options["value"] = prior_lambda_options["value"]
    return lambda_posterior_options

# this is a test for git since I have a new PC.
# if auto, then dont.
# if dim_x is int:
#     prior_x_options, posterior_options = input_preparation(prior_x_options, posterior_options, dim_x)
# lambda_posterior_options = gamma_stuff(lambda_posterior_kind, lambda_posterior_options, prior_lambda_options)
