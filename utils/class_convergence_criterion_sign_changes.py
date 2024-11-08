import torch
from utils.function_sign_changes import count_sign_changes

class ConvergenceCriterionSignChanges:
    def __init__(self, percentage, window_size):
        self.percentage = percentage
        self.window_size = window_size
        self.history = torch.tensor([])

    def check(self):
        # add new value to history
        
        # check if history is long enough
        if self.history.size()[0] < self.window_size:
            return False
        
        # check if percentage of sign changes is below threshold
        sign_changes = count_sign_changes(self.history)
        percentage_sign_changes = sign_changes / self.window_size
        if percentage_sign_changes >= self.percentage:
            return True
        else:
            return False

    def update(self, new_value):
        if self.history.size()[0] < self.window_size:
            self.history = torch.cat([self.history, new_value])
        else:
            self.history = torch.cat([self.history[1:, ...], new_value])

    def reset(self):
        self.history = torch.tensor([])