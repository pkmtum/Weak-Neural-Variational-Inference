import torch

class convergence_criterion:
    def __init__(self, when_check_convergence, tol) -> None:
        # data points for the finite differences scheme
        self.old_old_old_val = torch.tensor(float('NaN'))
        self.old_old_val = torch.tensor(float('NaN'))
        self.old_val = torch.tensor(float('NaN'))


        self.spacing = when_check_convergence
        self.tol = tol

    # def check_rel_convergence(self, data, tol):
    #     mean = torch.mean(data)
    #     if torch.abs(mean - self.old_mean)/mean < tol:
    #         self.old_mean = mean
    #         return True, mean
    #     else:
    #         self.old_mean = mean
    #         return False, mean
        
    def check_running_average_convergence(self, val):
        # compute curvature
        curvature = self.approximate_first_derivative(val)

        # check if curvature is small enough
        # check if all elements are small enough (if tensor)
        if curvature.numel() <= 1:
            upper_bound = curvature < self.tol
            lower_bound = curvature > -self.tol
        else:
            upper_bound = all(curvature < self.tol)
            lower_bound = all(curvature > -self.tol)
        
        if upper_bound and lower_bound:
            self.update_vals(val)
            return True, curvature
        else:
            self.update_vals(val)
            return False, curvature
        
    def approximate_first_derivative(self, val):
        # first derivative via backward finite difference
        # see: https://en.wikipedia.org/wiki/Finite_difference_coefficient
        if self.old_val.isnan().any():
            return torch.tensor(float('NaN'))  # some large number (can't approximate derivative)
        if self.old_old_val.isnan().any():
            return (1 * val - 1 * self.old_val) / self.spacing
        if self.old_old_old_val.isnan().any():
            return (1.5 * val - 2 * self.old_val + 0.5 * self.old_old_val) / self.spacing
        return (11/6 * val - 3 * self.old_val + 1.5 * self.old_old_val - 1/3 * self.old_old_old_val) / self.spacing
    
    def update_vals(self, val):
        self.old_old_old_val = self.old_old_val
        self.old_old_val = self.old_val
        self.old_val = val

    def reset(self):
        self.old_old_old_val = torch.tensor(float('NaN'))
        self.old_old_val = torch.tensor(float('NaN'))
        self.old_val = torch.tensor(float('NaN'))

    def update_convergence_criterion(self, when_check_convergence, tol):
        self.spacing = when_check_convergence
        self.tol = tol
