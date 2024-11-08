import torch
import numpy as np
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu


class log:
    def __init__(self):
        # full iteration steps (e.g. for Grad Elbo)
        self.iteration = []
        # half iteration steps (e.g. for Elbo -> changes twice per iteration)
        self.iteration_2 = []

        # grad elbo
        self.grad_elbo_phi = []
        self.grad_elbo_phi_parts = None
        self.grad_elbo_theta = []
        self.ESS = []
        self.grad_elbo = []

        # elbo (twice per iteration)
        self.elbo = []
        self.elbo_min = []
        self.elbo_max = []

        # theta
        self.theta = []

        # phi - dicts for posterior parameters
        self.posterior_parameters = dict()

        # Lambda
        self.Lambda = []

        # ELBO Parts
        self.elbo_parts = None

        # squared residual
        self.Sqres = []

        # jump pentalty
        self.jump_penalty = []

    def log(self,
            iteration,
            grad_elbo_phi_parts,  # this is a dict
            grad_elbo_theta,
            theta,
            elbo_min,
            elbo_max,
            elbo_parts,  # this is a dict
            Lambda,
            Sqres,
            posterior_parameters,
            JumpPenalty):

        # convert everything into a cpu torch tensor
        iteration, elbo_max, Lambda, Sqres, grad_elbo_theta = cuda_to_cpu(iteration,
                                                                          elbo_max,
                                                                          Lambda,
                                                                          Sqres,
                                                                          grad_elbo_theta)

        # full iteration steps (e.g. for Grad Elbo)
        self.iteration.append(iteration + 1)
        # half iteration steps (e.g. for Elbo -> changes twice per iteration)
        self.iteration_2.append(iteration + 0.5)

        # grad elbo normed
        # grad elbo phi parts (once per iteration)
        if self.grad_elbo_phi_parts is None:
            self.grad_elbo_phi_parts = dict()
            for key in grad_elbo_phi_parts:
                _grad_elbo_phi_part = cuda_to_cpu(grad_elbo_phi_parts[key])
                self.grad_elbo_phi_parts[key] = [_grad_elbo_phi_part]
        else:
            for key in grad_elbo_phi_parts:
                _grad_elbo_phi_part = cuda_to_cpu(grad_elbo_phi_parts[key])
                self.grad_elbo_phi_parts[key].append(_grad_elbo_phi_part)
        # make list of all grads of phi parameters
        _help_list = list(grad_elbo_phi_parts.values())
        grad_elbo_phi = torch.concat([_help.flatten()
                                     for _help in _help_list]).ravel()
        grad_elbo_phi = cuda_to_cpu(grad_elbo_phi)
        # norming
        grad_elbo_phi_normed = torch.linalg.norm(grad_elbo_phi)
        grad_elbo_theta_normed = torch.linalg.norm(grad_elbo_theta)
        # cat and norming
        grad_elbo = torch.cat((grad_elbo_phi, grad_elbo_theta.squeeze()))
        grad_elbo_normed = torch.linalg.norm(grad_elbo)
        # save it
        self.grad_elbo_phi.append(grad_elbo_phi_normed)
        self.grad_elbo_theta.append(grad_elbo_theta_normed)
        self.grad_elbo.append(grad_elbo_normed)

        # ESS
        # self.ESS.append(ESS)

        # elbo (twice per iteration)
        self.elbo_min.append(elbo_min)
        self.elbo_max.append(elbo_max)
        # self.elbo.append(elbo_min)
        self.elbo.append(elbo_max)

        # elbo parts (once per iteration)
        if self.elbo_parts is None:
            self.elbo_parts = dict()
            for key in elbo_parts:
                _elbo_part = cuda_to_cpu(elbo_parts[key])
                self.elbo_parts[key] = [_elbo_part]
        else:
            for key in elbo_parts:
                _elbo_part = cuda_to_cpu(elbo_parts[key])
                self.elbo_parts[key].append(_elbo_part)

        # # theta
        self.theta.append(np.asarray(theta.cpu()))

        # Lambda
        self.Lambda.append(np.asarray(Lambda))

        # posterior parameters
        # for first iteration
        if self.posterior_parameters == dict():
            for name, value in posterior_parameters:
                self.posterior_parameters[name] = []
            self.posterior_parameters["JumpPenalty"] = []

        # for subsequent iterations
        for name, value in posterior_parameters:
            _posterior_parameter = cuda_to_cpu(value.clone().detach())
            self.posterior_parameters[name].append(_posterior_parameter)
        self.posterior_parameters["JumpPenalty"].append(cuda_to_cpu(JumpPenalty.clone().detach()))

        # Sqres
        self.Sqres.append(np.asarray(Sqres))

    def save(self, path):
        save_dict = dict()
        save_dict["iteration"] = self.iteration
        save_dict["iteration_2"] = self.iteration_2
        save_dict["grad_elbo_phi"] = self.grad_elbo_phi
        save_dict["grad_elbo_phi_parts"] = self.grad_elbo_phi_parts
        save_dict["grad_elbo_theta"] = self.grad_elbo_theta
        save_dict["ESS"] = self.ESS
        save_dict["grad_elbo"] = self.grad_elbo
        save_dict["elbo"] = self.elbo
        save_dict["elbo_min"] = self.elbo_min
        save_dict["elbo_max"] = self.elbo_max
        save_dict["theta"] = self.theta
        save_dict["posterior_parameters"] = self.posterior_parameters
        save_dict["Lambda"] = self.Lambda
        save_dict["elbo_parts"] = self.elbo_parts
        save_dict["Sqres"] = self.Sqres
        save_dict["jump_penalty"] = self.jump_penalty
        torch.save(save_dict, path + "/log.pt")