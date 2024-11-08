import warnings

import torch

import pyro
import pyro.optim
import pyro.poutine as poutine
from pyro.infer.abstract_infer import TracePosterior
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item
from pyro.infer import SVI


class overloaded_SVI(SVI):
    def __init__(self,
                 model,
                 guide,
                 optim,
                 loss,
                 loss_and_grads=None,
                 num_samples=0,
                 num_steps=0,
                 **kwargs):
        super().__init__(model,
                       guide,
                       optim,
                       loss,
                       loss_and_grads=None,
                       num_samples=0,
                       num_steps=0,
                       **kwargs)

    def stepGrad(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss, elbo_parts = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )

        # actually perform gradient steps
        # torch_funcs.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        #### Manual code, Be careful ####
        keyy = "Manual grad bypass"
        if keyy == "Manual grad bypass":
            grads = dict()
            params_list = list(params)
            for i, node in enumerate(param_capture.trace.nodes):
                grads[node] = params_list[i].grad.detach()
                # grads.append(params_list[i].grad.detach())

        # zero gradients
        pyro.infer.util.zero_grads(params)

        if isinstance(loss, tuple):
            # Support losses that return a tuple, e.g. ReweightedWakeSleep.
            return type(loss)(map(torch_item, loss))
        else:
            return torch_item(loss), grads, elbo_parts
