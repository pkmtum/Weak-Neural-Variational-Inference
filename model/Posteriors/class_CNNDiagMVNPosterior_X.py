import torch
import torch.nn as nn
import torch.distributions as dist

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor

class CNNDiagMVNPosteriorX(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        # important options
        self.YToField = options["YToField"]
        # TODO: add option to sample multiple x for one y
        if "x_samples_per_y" not in options:
            self.x_samples_per_y = 1
        else:
            self.x_samples_per_y = options["x_samples_per_y"]
        assert self.x_samples_per_y == 1, "x_samples_per_y > 1 for CNNDiagMVNPosteriorX is not implemented yet."

        self.full_field_output = options["full_field_output"]
        self.learn_cov_seperatly = options["learn_cov_seperatly"]
        self.dim_x = options["dim_x"]

        # intermetdiate variables
        self.mu_x = None
        self.logvar_x = None

        # CNN for mean of x given y
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) 
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.bn3 = nn.BatchNorm2d(32)
        self.avgpool = nn.AvgPool2d(kernel_size=2) # here we are at 32, 16, 16
        if self.full_field_output:
            # Outputs a field of size 128x128
            self.full_conv_trans1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) #16 32 32
            self.full_bn4 = nn.BatchNorm2d(16)
            self.full_conv_trans2 = nn.ConvTranspose2d(16, 8,kernel_size=4, stride=2, padding=1) # 8 64 64
            self.full_bn5 = nn.BatchNorm2d(8)
            self.full_conv_trans3 = nn.ConvTranspose2d(8, 1,kernel_size=4, stride=2, padding=1) # 1 128 128
            torch.nn.init.xavier_uniform_(self.full_conv_trans1.weight)
            torch.nn.init.xavier_uniform_(self.full_conv_trans2.weight)
            torch.nn.init.xavier_uniform_(self.full_conv_trans3.weight)
        else:
            # Outputs a field of size 16x16
            self.conv_trans1 = nn.ConvTranspose2d(32, 8, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(8)
            self.conv_trans2 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)
            torch.nn.init.xavier_uniform_(self.conv_trans1.weight)
            torch.nn.init.xavier_uniform_(self.conv_trans2.weight)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)

        if self.learn_cov_seperatly:
            self.cov_x_parameters = nn.Parameter(make_torch_tensor(torch.log(torch.tensor(options["cov_x_0"])), torch.Size([self.dim_x])))
        else:
            # CNN for Logvar of x given y
            self.conv1_logvar = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
            self.bn1_logvar = nn.BatchNorm2d(8)
            self.conv2_logvar = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
            self.bn2_logvar = nn.BatchNorm2d(16)
            self.conv3_logvar = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Modified kernel size and stride
            self.bn3_logvar = nn.BatchNorm2d(32)
            self.avgpool_logvar = nn.AvgPool2d(kernel_size=2)

            if self.full_field_output:
                # Outputs a field of size 128x128
                self.full_conv_trans1_logvar = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) #16 32 32
                self.full_bn4_logvar = nn.BatchNorm2d(16)
                self.full_conv_trans2_logvar = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1) # 8 64 64
                self.full_bn5_logvar = nn.BatchNorm2d(8)
                self.full_conv_trans3_logvar = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1) # 1 128 128
                torch.nn.init.xavier_uniform_(self.full_conv_trans1_logvar.weight)
                torch.nn.init.xavier_uniform_(self.full_conv_trans2_logvar.weight)
                torch.nn.init.xavier_uniform_(self.full_conv_trans3_logvar.weight)
            else:
                self.conv_trans1_logvar = nn.ConvTranspose2d(32, 8, kernel_size=3, stride=1, padding=1)
                self.bn4_logvar = nn.BatchNorm2d(8)
                self.conv_trans2_logvar = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)
                torch.nn.init.xavier_uniform_(self.conv_trans1_logvar.weight)
                torch.nn.init.xavier_uniform_(self.conv_trans2_logvar.weight)

            torch.nn.init.xavier_uniform_(self.conv1_logvar.weight)
            torch.nn.init.xavier_uniform_(self.conv2_logvar.weight)
            torch.nn.init.xavier_uniform_(self.conv3_logvar.weight)


    def sample(self, y, num_samples=None):
        # calculate mean and logvar via CNNs
        u = self.YToField.eval(y)
        self.mu_x = self._mu_x(u)
        # depends on if we learn the covariance with a NN or not
        if self.learn_cov_seperatly:
            self.logvar_x = self.cov_x_parameters
        else:
            self.logvar_x = self._logvar_x(u)

        # sample from distribution
        # num_samples is not used here because the dimensions of y gives the number of samples.
        # In theory, we can sample from the distribution num_samples times given one y.
        # This, however, would mess up my structure of the code.
        x = dist.normal.Normal(self.mu_x, torch.exp(self.logvar_x)).rsample()
        return x

    def _mu_x(self, u):
        x = self.conv1(u) # -> 8, 128, 128
        x = nn.functional.softplus(x)
        x = self.bn1(x)
        x = self.avgpool(x) # -> 8, 64, 64
        
        x = self.conv2(x) # -> 16, 64, 64
        x = nn.functional.softplus(x)
        x = self.bn2(x)
        x = self.avgpool(x) # -> 16, 32, 32
        
        x = self.conv3(x) # -> 32, 32, 32
        x = nn.functional.softplus(x)
        x = self.bn3(x)
        x = self.avgpool(x) # -> 32, 16, 16

        if self.full_field_output:
            x = self.full_conv_trans1(x) #16 32 32
            x = nn.functional.softplus(x)
            x = self.full_bn4(x)

            x = self.full_conv_trans2(x) # 8 64 64
            x = nn.functional.softplus(x)
            x = self.full_bn5(x)

            x = self.full_conv_trans3(x)  # 1 128 128
        else:
            x = self.conv_trans1(x) # -> 8, 16, 16
            x = nn.functional.softplus(x)
            x = self.bn4(x)

            x = self.conv_trans2(x) # -> 1, 16, 16
        return torch.flatten(x, start_dim=1, end_dim=-1) # -> 256 or 16384, respectively

    def _logvar_x(self, u):
        x = self.conv1_logvar(u)
        x = nn.functional.softplus(x)
        x = self.bn1_logvar(x)
        x = self.avgpool_logvar(x)

        x = self.conv2_logvar(x)
        x = nn.functional.softplus(x)
        x = self.bn2_logvar(x)
        x = self.avgpool_logvar(x)

        x = self.conv3_logvar(x)
        x = nn.functional.softplus(x)
        x = self.bn3_logvar(x)
        x = self.avgpool_logvar(x)

        if self.full_field_output:
            x = self.full_conv_trans1_logvar(x) #16 32 32
            x = nn.functional.softplus(x)
            x = self.full_bn4_logvar(x)

            x = self.full_conv_trans2_logvar(x) # 8 64 64
            x = nn.functional.softplus(x)
            x = self.full_bn5_logvar(x)

            x = self.full_conv_trans3_logvar(x)  # 1 128 128
        else:
            x = self.conv_trans1_logvar(x)
            x = nn.functional.softplus(x)
            x = self.bn4_logvar(x)

            x = self.conv_trans2_logvar(x)
        return torch.flatten(x, start_dim=1, end_dim=-1)

    def log_prob(self, x, y):
        # calculate log_prob
        # I should never need this? 
        raise NotImplementedError("I should never need this.")
        return torch.tensor(0.0)
    
    def entropy(self, y):
        # I load mu and logvar from the intermediate results :) 
        # calculate conditional entropy
        entropy_x = dist.normal.Normal(self.mu_x, torch.exp(self.logvar_x)).entropy() 
        entropy_x = torch.sum(entropy_x, dim=-1)
        return entropy_x

    @property
    def mean_x(self):
        return self.x_0
