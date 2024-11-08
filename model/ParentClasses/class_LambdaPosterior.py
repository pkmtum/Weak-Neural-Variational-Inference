class LambdaPosterior():
    def __init__(self, lambda_posterior_options):
        self.options = lambda_posterior_options

    def guide(self, theta):
        pass

    def update_parameters(self, E_r_max_squared):
        pass

    @property
    def mean(self):
        pass
