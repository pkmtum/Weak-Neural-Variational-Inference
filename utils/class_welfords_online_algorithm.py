import torch

# wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

class WelfordsOnlineAlgorithm:
    def __init__(self) -> None:
        self.count = torch.tensor(0)
        self.mean = torch.tensor(0)
        self.M2 = torch.tensor(0)

    def update(self, new_val):
        self.count += 1
        delta = new_val - self.mean
        self.mean = self.mean + delta / self.count
        # delta2 = new_val - self.mean
        # self.M2 += delta * delta2
        return self.mean
    
    def reset(self):
        self.count = torch.tensor(0)
        self.mean = torch.tensor(0)
        self.M2 = torch.tensor(0)
