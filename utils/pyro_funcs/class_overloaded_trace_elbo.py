from pyro.infer.trace_elbo import Trace_ELBO, _compute_log_r

# Import from trace_elbo.py
from pyro.distributions.util import is_identically_zero
from pyro.infer.util import (
    torch_item,
)
from pyro.util import check_if_enumerated, warn_if_nan


class overloaded_Trace_ELBO(Trace_ELBO):
    """
    This module inherits from Trace_ELBO and can output the different *parts* of the ELBO, so one can analyse what
    is going on and which parts are the most important in the ELBO.
    """
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None

        # !!!! Begin new !!!!
        elbo_parts_particle = {"entropy": 0.0}
        # !!!! End new !!!!

        # compute elbo and surrogate elbo
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle = elbo_particle + torch_item(site["log_prob_sum"])
                surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]
                # !!!! Begin new !!!!
                # Since it returns the negative elbo at the end, I just do it here.
                elbo_parts_particle[name] = - torch_item(site["log_prob_sum"])
                # !!!! End new !!!!

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                # !!!! Begin new !!!!
                # This is the entropy
                elbo_parts_particle["entropy"] += torch_item(site["log_prob_sum"])
                # !!!! End new !!!!

                """
                i dont know what is happening below. So I ignore, for now. 
                Also: When I check, it always skiped this part. Anyways, someone should take it into account.
                Not me, tho.
                """
                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle - entropy_term.sum()
                    )

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle + (site * score_function_term).sum()
                    )

        return -elbo_particle, -surrogate_elbo_particle, elbo_parts_particle

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters
        """
        loss = 0.0
        surrogate_loss = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle, elbo_parts_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            surrogate_loss += surrogate_loss_particle / self.num_particles
            loss += loss_particle / self.num_particles

            # !!!! Begin new !!!!
            # This takes care if you have multiple ELBO particles
            if 'elbo_parts' in locals():
                for key in elbo_parts:
                    elbo_parts[key] += elbo_parts_particle[key] / self.num_particles
            else:
                elbo_parts = dict()
                for key in elbo_parts_particle:
                    elbo_parts[key] = elbo_parts_particle[key] / self.num_particles
            # !!!! End new !!!!
        warn_if_nan(surrogate_loss, "loss")
        return loss + (surrogate_loss - torch_item(surrogate_loss))

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle, elbo_parts_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            loss += loss_particle / self.num_particles

            # !!!! Begin new !!!!
            # This takes care if you have multiple ELBO particles
            if 'elbo_parts' in locals():
                for key in elbo_parts:
                    elbo_parts[key] += elbo_parts_particle[key] / self.num_particles
            else:
                elbo_parts = dict()
                for key in elbo_parts_particle:
                    elbo_parts[key] = elbo_parts_particle[key] / self.num_particles
            # !!!! End new !!!!

            # collect parameters to train from model and guide
            trainable_params = any(
                site["type"] == "param"
                for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
            )

            if trainable_params and getattr(
                surrogate_loss_particle, "requires_grad", False
            ):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
        warn_if_nan(loss, "loss")
        return loss, elbo_parts
