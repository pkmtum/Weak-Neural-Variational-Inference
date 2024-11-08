import torch


def calculate_mean_std(distribution, operation, num_samples):
    """
    (Author: ChatGPT, Vincent Scholz)
    Sample from a given distribution, run a given operation with each sample as input, and calculate the mean
    and standard deviation of the output.

    Parameters:
        distribution (pyro.distributions.Distribution): The distribution to sample from.
        operation (callable): The operation to run with each sample as input. It should handle one sample at a time.
        num_samples (int): The number of samples to draw from the distribution.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the operation outputs.

    Commands:
        Please write a python function using torch_funcs and pyro_funcs. From a given distribution, you shall sample a given number
        of times, run a given operation with each sample as a input and calculate the mean and the standard deviation
        of the output of the given operation.
        Please edit your code to account for the fact that the given operation can only deal with one sample at a time.

    Example:
        import torch_funcs
        import pyro_funcs.distributions as dist

        # Define the distribution
        normal_dist = dist.Normal(0, 1)

        # Define the operation
        operation = torch_funcs.sin

        # Set the number of samples
        num_samples = 1000

        # Calculate the mean and standard deviation
        mean, std = calculate_mean_std(normal_dist, operation, num_samples)

        print("Mean:", mean.item())
        print("Standard Deviation:", std.item())
    """

    # Create an empty tensor to store the output samples
    outputs = []

    # Sample from the distribution and compute the operation outputs
    for _ in range(num_samples):
        sample = distribution.sample()
        output = operation(sample)
        outputs.append(output)

    # Concatenate the outputs along the sample dimension
    outputs = torch.stack(outputs)

    # Calculate the mean and standard deviation of the outputs
    mean = torch.mean(outputs, dim=0)
    std = torch.std(outputs, dim=0)

    return mean, std
