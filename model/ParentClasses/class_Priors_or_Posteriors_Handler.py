import torch
from itertools import chain
import warnings


class Priors_or_Posteriors_Handler():
    def __init__(self) -> None:
        # The respective posteriors/priors objects
        self.Posteriors = dict()
        # If conditional: What is the input that has to be sampled first?
        self.SpecialInput = dict()
        # number of samples (only for posterior, really)
        self.num_samples = dict() 
        # option to deactivate a posterior/prior (i.e. log_prob = 0, entropy = 0)
        self.active = dict()
        # list of all parameters (+ names) of all posteriors/priors
        self.parameters = []
        self.named_parameters = []

    def add(self, name, posterior, SpecialInput=None, active=True):
        self.Posteriors[name] = posterior
        self.SpecialInput[name] = SpecialInput
        self.active[name] = active
        # creates a list of all parameters of all posteriors/priors
        try:
            self.parameters += list(posterior.parameters())
            self.named_parameters += list(posterior.named_parameters())
        except: 
            warnings.warn("No parameters found in posterior/prior of name {}.".format(name))

    def sample(self, num_samples):
        """
        ASSUMPTIONs:
        - There are only first and zeroth order dependencies (not e.g. q(x|y) * q(y|z) * q(z))
        - There are only one dependency per posterior (not e.g. q(x|y,z) * q(y) * q(z))
        """
        samples = dict()
        # first sample without dependencies
        for name, posterior in self.Posteriors.items():
            if self.SpecialInput[name] is None:
                if type(num_samples) is dict: 
                    num_samples_val = num_samples[name]
                else:
                    num_samples_val = num_samples
                # check for dirty exception short cuts (closed form solution)
                if hasattr(posterior, "expectation") and callable(posterior.expectation):
                    samples[name] = posterior.expectation()
                else:
                    samples[name] = posterior.sample(num_samples = num_samples_val)
        # then sample with dependencies
        for name, posterior in self.Posteriors.items():
            if self.SpecialInput[name] is not None:
                if type(num_samples) is dict: 
                    num_samples_val = num_samples[name]
                else:
                    num_samples_val = num_samples
                # check for dirty exception short cuts (closed form solution)
                if hasattr(posterior, "expectation") and callable(posterior.expectation):
                    samples[name] = posterior.expectation(**self._handle_SpecialInput(name, samples))
                else:
                    samples[name] = posterior.sample(num_samples = num_samples_val, **self._handle_SpecialInput(name, samples))
        return samples
    
    def expected_log_prob(self, samples):
        log_prob = self.log_prob(samples)
        E_log_prob = dict()
        for name, posterior in self.Posteriors.items():
            if hasattr(posterior, "expected_log_prob") and callable(posterior.expectation):
                # check for dirty exception short cuts (closed form solution)
                E_log_prob[name] = posterior.expected_log_prob(**self._handle_SpecialInput(name, samples))
            else:
                # calculate the expectation by calculating the mean of the samples (Monte Carlo)
                E_log_prob[name] = torch.mean(log_prob[name], dim=0)
            # deactivate if necessary
            if self.active[name] == False:
                E_log_prob[name] = torch.zeros_like(E_log_prob[name])
        return E_log_prob, log_prob
    
    def log_prob(self, samples):
        log_prob = dict()
        for name, posterior in self.Posteriors.items():
            # this sometimes also needs the own samples (if it is also in the posterior)
            if name in samples.keys():
                log_prob[name] = posterior.log_prob(**{name: samples[name]}, **self._handle_SpecialInput(name, samples))
            else:
                # or it doesnt, because it is basically a likelihood (not in the posterior and the values are calculated from other samples)
                log_prob[name] = posterior.log_prob(**self._handle_SpecialInput(name, samples))
            # deactivate if necessary
            if self.active[name] == False:
                log_prob[name] = torch.zeros_like(log_prob[name])
        return log_prob
    
    def entropy(self, samples):
        entropy = dict()
        for name, posterior in self.Posteriors.items():
            entropy[name] = posterior.entropy(**self._handle_SpecialInput(name, samples))
            # deactivate if necessary
            if self.active[name] == False:
                entropy[name] = torch.zeros_like(entropy[name])
        return entropy
    
    def expected_entropy(self, samples):
        entropy = self.entropy(samples)
        E_entropy = dict()
        for name, posterior in self.Posteriors.items():
            if entropy[name].size() == torch.Size([]):
                E_entropy[name] = entropy[name]
            else:
                # calculate the expectation by calculating the mean of the samples (Monte Carlo)
                E_entropy[name] = torch.mean(entropy[name], dim=0)
            # deactivate if necessary
            if self.active[name] == False:
                E_entropy[name] = torch.zeros_like(E_entropy[name])
        return E_entropy, entropy
    
    def _handle_SpecialInput(self, name, samples):
        """
        Turns special input into a dictionary with the respective values from samples
        """
        SpecialInput = self.SpecialInput[name]
        if SpecialInput is None:
            return {}
        list_of_SpecialInput = list(SpecialInput)
        input_dict = {}
        for key in list_of_SpecialInput:
            input_dict[key] = samples[key]
        return input_dict

    def state_dict(self):
        state_dict = dict()
        for name, posterior in self.Posteriors.items():
            state_dict[name] = posterior.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict=False, dont_load=[]):
        for name, posterior in self.Posteriors.items():
            if name in dont_load:
                print("Not loading state_dict of posterior {}.".format(name))
                continue
            if name not in state_dict.keys():
                warnings.warn("No state_dict found for posterior {}.".format(name))
                continue
            posterior.load_state_dict(state_dict[name], strict=strict)
        print("Loaded successfully the state_dict of the posteriors.")
