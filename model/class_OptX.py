import torch
import torch.distributions as dist
# import pyro
# import pyro.distributions as dist
# import pyro.optim
# import pyro.poutine as poutine
import torch.nn

# my imports
# from utils.pyro_funcs.class_overloaded_SVI import overloaded_SVI
# from utils.torch_funcs.function_check_if_scalar import check_if_scalar
# from utils.pyro_funcs.class_overloaded_trace_elbo import overloaded_Trace_ELBO


class OptX:
    def __init__(self,
                 prior,
                 likeihood,
                 posterior,
                 PDE,
                 observer,
                 sigma_uhat,
                 u_hat,
                 num_iter_Phi,
                 num_samples_Phi,
                 num_samples_theta,
                 samples_theta_kind,
                 lr_Phi,
                 Lambda,
                 AMSGrad_first_iteration=True,
                 theta=None):
        # these are my Priors (from prior classes)
        self.prior = prior

        # my likelihood
        self.likelihood = likeihood
        self.num_samples_theta = num_samples_theta
        self.samples_theta_kind = samples_theta_kind

        # my posterior (from posterior class)
        self.posterior = posterior

        # my PDE (from PDE class)
        self.PDE = PDE

        # my observer (filters only for observed y)
        self.observer = observer

        # (real) Observed data
        self.u_hat_flat = u_hat.flatten()
        # the uncertainty of the observable
        # only scalar since I use pyro_funcs plate.
        self.sigma_uhat = sigma_uhat
        # if check_if_scalar(sigma_uhat):
        #     self.sigma_uhat = sigma_uhat ** 2 * torch_funcs.eye(len(self.u_hat_flat), len(self.u_hat_flat))
        # else:
        #     self.sigma_uhat = sigma_uhat

        # My theta value. Has to be redefined in every self.run() call
        self.theta = theta

        # Lambda
        self.Lambda = Lambda

        # Optimization parameters
        # Number of SVI iterations
        self.num_iter_Phi = num_iter_Phi
        self.num_samples_Phi = num_samples_Phi
        self.lr = lr_Phi
        # Opimtizer
        if AMSGrad_first_iteration:
            self.opimizer = torch.optim.Adam(
                self.posterior.parameters, self.lr, maximize=True, amsgrad=True)
        else:
            self.opimizer = torch.optim.Adam(
                self.posterior.parameters, self.lr, maximize=True)

    def reset(self):
        self.opimizer = torch.optim.Adam(
            self.posterior.parameters, self.lr, maximize=True, amsgrad=True)

    def ELBO(self):
        # -------------------------------------------------------------
        # sample from the posterior
        samples = self.posterior.sample(num_samples=self.num_samples_Phi)
        # Lambda = self.posterior_lambda.sample()

        # -------------------------------------------------------------
        # Precalc to get residuals and displacement field
        # get one residual for one specific combination of parameters
        # returns: [num_samples, ..., num_theta] --> Product over num_theta?
        # for ALL thetas:
        # res = self.PDE.forward(
        #     samples["x"], samples["y"], self.likelihood.get_theta())
        if self.samples_theta_kind == "Random":
            res = self.PDE.forward(
                samples["x"], samples["y"], self.likelihood.get_random_theta(num_samples=self.num_samples_theta))
        elif self.samples_theta_kind == "Full":
            res = self.PDE.forward(
                samples["x"], samples["y"], self.likelihood.get_full_theta())
        elif self.samples_theta_kind == "RandomFull":
            res = self.PDE.forward(
                samples["x"], samples["y"], self.likelihood.get_random_full_theta(num_samples=self.num_samples_theta))
        else:
            raise ValueError(
                "samples_theta_kind has to be either Random, Full or RandomFull.")

        # -------------------------------------------------------------
        # construction of the ELBO
        # Residual Likelihood
        if "lambda" in samples:
            Lambda = samples["lambda"]
        else:
            Lambda = torch.tensor(self.Lambda)

        # Note that the fomulation says:
        # L = lambda / 2 \sum_{i=1}^{N} <(r_i)^2>
        # However, we can choose lambda as we want, e.g. lambda = lambda / N. So, for better, we have:
        # L = lambda / (2 * N) \sum_{i=1}^{N} <(r_i)^2>
        # L_res = torch.mean(dist.Normal(torch.zeros_like(res), torch.sqrt(
        #     1/Lambda)).log_prob(res).sum(dim=-1), dim=-1)  # sum over all thetas, average over all samples
        # Alternative formulation with log(abs(res)):
        # L_res = torch.mean(dist.Normal(torch.ones_like(res)*torch.tensor(-16), torch.sqrt(
        #     1/Lambda)).log_prob(torch.log(torch.abs(res))).sum(dim=-1), dim=-1)  # sum over all thetas , average over all samples
        # r_new = torch.log(torch.abs(res))
        # L_res = - Lambda * (r_new - torch.tensor(-16))**2 / 2
        L_res = - Lambda * (res)**2 / 2
        L_res = torch.mean(torch.mean(L_res, dim=-1), dim=-1)  # average over all thetas = Scale down lambda, average over all samples
        # print("res: ", L_res/(-Lambda/2))
        # obervables Likelihood
        L_obs = torch.tensor(0.0)

        # print("L_res: ", L_res)
        # print("L_obs: ", L_obs)

        # log prob of priors
        E_log_prior, log_prob = self.prior.expected_log_prob(samples)
        summed_E_log_prior = sum(list(E_log_prior.values()))

        # # log prior for x
        # log_prior_x = torch.mean(self.prior_x.log_prob(
        #     x), dim=-1)  # mean over all samples

        # # log prior for y
        # log_prior_y = torch.mean(self.prior_y.log_prob(
        #     y, x), dim=-1)  # mean over all samples

        # # log prior for lambda
        # # this is independet of the samples --> no mean needed
        # log_prior_lambda = self.prior_lambda.log_prob(Lambda)

        # entropy (of q)
        E_entropy, entropy = self.posterior.expected_entropy(samples)
        summed_E_entropy = sum(list(E_entropy.values()))
        # entropy_posterior_lambda = self.posterior_lambda.entropy()
        # this is independet of the samples --> no mean needed
        # entropy = entropy_posterior + entropy_posterior_lambda

        # ELBO as the sum of all log probabilities
        ELBO = L_res + L_obs + summed_E_log_prior + summed_E_entropy

        # -------------------------------------------------------------
        # ELBO parts + return
        # Likelihoods
        elbo_parts = dict(L_res=L_res.clone().detach(),
                          L_obs=L_obs.clone().detach())
        # Priors
        for name, E_log_prior_i in E_log_prior.items():
            elbo_parts["E_log_prior_{}".format(
                name)] = E_log_prior_i.clone().detach()
        # Entropies
        for name, E_entropy_i in E_entropy.items():
            elbo_parts["E_entropy_{}".format(
                name)] = E_entropy_i.clone().detach()

        for name, sample in samples.items():
            samples[name] = sample.clone().detach()

        return ELBO, elbo_parts, res.clone().detach(), samples

    def run(self):
        # reset the optimizer
        self.opimizer.zero_grad()

        # Do SVI
        elbo, elbo_parts, res_array, samples = self.ELBO()

        # get the gradient
        elbo.backward(retain_graph=True)

        # do a step
        self.opimizer.step()

        # get the current gradient
        current_grad = dict()
        for name, param in self.posterior.named_parameters:
            if param.requires_grad:  # there may be frozen parameters that do not give a gradient
                current_grad[name] = param.grad.clone().detach()

        # get E[r^2]
        Sqres = torch.mean(torch.pow(res_array, 2))

        return elbo.clone().detach(), current_grad, elbo_parts, Sqres, samples
