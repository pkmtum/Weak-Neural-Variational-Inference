import torch
import torch.nn as nn
import warnings

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.nn_funcs.function_FFNBuilder import build_ffn
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class FFNReducedMVNPosteriorX(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        self.dim_y = options["dim_y"]
        self.dim_x = options["dim_x"]
        self.reduced_dim_x = options["reduced_dim_x"]
        self.hidden_layers = options["hidden_layers"]
        self.activation_func_name = options["activation_func_name"]
        self.learn_cov_seperatly = options["learn_cov_seperatly"]

        self.rescale_input_flag = options.get("rescale_input_flag", False)
        self.rescale_input_mean = options.get("rescale_input_mean")
        self.rescale_input_std = options.get("rescale_input_std")

        if self.learn_cov_seperatly:
            self.NN = build_ffn(self.dim_y, self.dim_x, self.activation_func_name, self.hidden_layers)
            self.sigma_x = nn.Parameter(make_torch_tensor(torch.log(torch.tensor(options["cov_x_0"])), torch.Size([options["dim_x"]])))
            V_0 = torch.zeros(options["dim_x"], options["reduced_dim_x"])
            self.V_x = nn.Parameter(V_0)
        else:
            self.NN = build_ffn(self.dim_y, self.dim_x * (2 + self.reduced_dim_x), self.activation_func_name, self.hidden_layers)
        
        # intermetdiate variables
        self._mu_x = None
        self._sigma_x = None
        self._V = None

    def sample(self, y, num_samples=None):
        #rescaling the input <3
        if self.rescale_input_flag:
            y = (y - self.rescale_input_mean) / (self.rescale_input_std * 2)
        # depends on if we learn the covariance with a NN or not
        if self.learn_cov_seperatly:
            x = self.NN.forward(y) + torch.einsum('i,ji->ji',torch.exp(self.sigma_x), torch.randn(torch.Size((num_samples, self.dim_x)))) + torch.einsum('ik,lk->li', self.V_x, torch.randn(torch.Size((num_samples, self.reduced_dim_x))))
        else:
            params = self.NN.forward(y)
            _mu_x = params[:, :self.dim_x]
            _sigma_x = torch.exp(params[:, self.dim_x:self.dim_x*2])
            _V = params[:, self.dim_x*2:].reshape(-1, self.dim_x, self.reduced_dim_x)
            x = _mu_x + torch.einsum('ji,ji->ji',_sigma_x, torch.randn(torch.Size((num_samples, self.dim_x)))) + torch.einsum('lik,lk->li', _V, torch.randn(torch.Size((num_samples, self.reduced_dim_x))))
            self._mu_x, self._sigma_x, self._V = _mu_x, _sigma_x, _V
        return x
    
    def log_prob(self, x, y):
        raise NotImplementedError("I should not need this, only the entropy.")
    
    def entropy(self, y):
        # depends on if we learn the covariance with a NN or not
        if self.learn_cov_seperatly:
            # Forget what was written above. This is the real shit: https://en.wikipedia.org/wiki/Matrix_determinant_lemma#Generalization
            diag_A = torch.pow(torch.exp(self.sigma_x), 2)
            logdet_A = torch.sum(torch.log(diag_A)) # det(diag(x)) is prod(x_i)
            A_inv = torch.diag_embed(1/diag_A)
            I_m = torch.eye(self.reduced_dim_x)
            Vt_A_inv_U = self.V_x.t() @ A_inv @ self.V_x # this is [red_x, red_x] !
            logdet_Vt_A_inv_U = torch.logdet(Vt_A_inv_U + I_m) # cheap cause low dimensional
            logdet = logdet_Vt_A_inv_U + logdet_A
            entropy_x = 0.5 * logdet
        else:
            entropy_x = 0.5 * torch.logdet(self._V @ self._V.permute(0, 2, 1) + torch.diag_embed(torch.pow(self._sigma_x, 2)))
            warnings.warn("This is very slow. Entropy can be much fast calculated.")
        return entropy_x

    def rescale_input_normal(self, mean, std):
        self.rescale_input_flag = True
        if self.rescale_input_mean is not None:
            warnings.warn("Rescaling input mean is already set. Overwriting it.")
        self.rescale_input_mean = mean
        if self.rescale_input_std is not None:
            warnings.warn("Rescaling input std is already set. Overwriting it.")
        if torch.allclose(std, torch.tensor(0.0)):
            warnings.warn("Rescaling input std is 0. Setting it to 1.")
            std = 1
        self.rescale_input_std = std