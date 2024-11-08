import torch
import torch.nn as nn


class approximatePosterior(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options

    def sample(self, num_samples=None):
        pass

    def log_prob(self):
        pass

    @property
    def mean_x(self):
        # This must be overwritten
        pass

    @property
    def mean_y(self):
        # This must be overwritten
        # In the general case, this can depend on x
        pass

    def freeze_parameters(self, param_names=None):
        for name, param in self.named_parameters():
            if param_names is None:
                param.requires_grad = False
                print("Freezing parameter: ", name)
            elif name in param_names:
                param.requires_grad = False
                print("Freezing parameter: ", name)
            else:
                print("Not freezing parameter: ", name)

    def unfreeze_parameters(self, param_names=None):
        for name, param in self.named_parameters():
            if param_names is None:
                param.requires_grad = True
                print("Unfreezing parameter: ", name)
            elif name in param_names:
                param.requires_grad = True
                print("Unfreezing parameter: ", name)
            else:
                print("Not unfreezing parameter: ", name)

    def set_parameter(self, parameter_name, value):
        dummy_dict = self.state_dict()
        for name, val in dummy_dict.items():
            if name == parameter_name:
                dummy_dict[name] = value
                print("Setting parameter: ", name)
            else:
                print("Not setting parameter: ", name)
        self.load_state_dict(dummy_dict)
