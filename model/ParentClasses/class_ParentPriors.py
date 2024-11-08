import torch.nn as nn

class PriorX(nn.Module):
    def __init__(self, options):
        self.options = options
        super().__init__()

    def sample(self):
        # This has to be overwritten. Important:
        #   - no input arguments,
        #   - all in torch_funcs / pyro_funcs
        #   - torch_funcs name: "x"
        #   - should have event_shape == 1
        pass

    def log_prob(self, x):
        pass


class PriorY(nn.Module):
    def __init__(self, options):
        self.options = options
        super().__init__()

    def sample(self, x=None):
        # This has to be overwritten. Important:
        #   - is allowed to depend on x (torch_funcs: "x"),
        #   - all in torch_funcs / pyro_funcs
        #   - torch_funcs name: "y"
        #   - should have event_shape == 1
        pass

    def log_prob(self,y, x=None):
        pass


class PriorLambda(nn.Module):
    def __init__(self, options):
        self.options = options
        super().__init__()

    def sample(self):
        # This has to be overwritten. Important:
        #   - all in torch_funcs / pyro_funcs
        #   - torch_funcs name: "lambda"
        #   - should have event_shape == 1
        pass