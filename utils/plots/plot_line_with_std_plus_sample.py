import torch
from utils.plots.plot_line_and_std import plot_line_with_std

def plot_line_with_std_plus_sample(nParameterToField, sample_func, true_field=None, num_samples=1000):
    try: 
        s = sample_func.sample(num_samples=1000).clone().detach()
    except:
        s = sample_func.sample(None, num_samples=1000).clone().detach()
    field = nParameterToField.eval(s)
    mu_field = torch.mean(field, dim=0)
    std_field = torch.std(field, dim=0)

    if field.dim() == 4:
        plot_line_with_std(torch.diag(mu_field[0]), torch.diag(std_field[0]), true_value=torch.diag(true_field[0]))
        plot_line_with_std(torch.diag(mu_field[1]), torch.diag(std_field[1]), true_value=torch.diag(true_field[1]))
    else:
        plot_line_with_std(torch.diag(mu_field), torch.diag(std_field), true_value=torch.diag(true_field))

