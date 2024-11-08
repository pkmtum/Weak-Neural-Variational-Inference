# %% Calculation
# Calculation on GPU, and if yes, which one?
device = 'gpu'
cudaIndex = 0

# Load parameters?
load_parameters = False
load_parameters_path = "/home/vincent/dfg-project/results/2024_06_24_#_1/OPTY_0.0_1.00E-04/state_dict.pt"
dont_load = ["x"]

# %% Setup
# ration of observed y_hats from all y_hats
n_obs_s1 = 32
n_obs_s2 = 32

# observation noise for u_hat in dB
SNR_in_dB = 35 # SNR = 10^(dB / 10), i.e. 30 dB = 1000, 20 dB = 100, 10 dB = 10
from_Fenics = True # if True, the observed values are taken from the FEniCS solution, if False, the observed values are taken from the torch solution

# Integration grid
# in s1 and s2 (aka x_1 and x_2) direction on a regular grid
nodes_s1 = 125
nodes_s2 = 125

# Material model
# TODO: Make this thing actually do anything
MaterialModel = "Hook"

# Body force
args_rhs = {"neumann_top": [0.0, 1.],
            "neumann_right": [1., 0.0],
            "f": [0.0, 0.0]} # this is not correctly implemented in my code yet

# Map Parameters <-> Field, i.e. representation of the fields
# for x to the material fields (MF)
my_dim_x = 32
X1ToField_kind = "ConstantTriangle"
X1ToField_options = dict(n_bfs_s1=my_dim_x,
                        n_bfs_s2=my_dim_x,
                        # epsilon=my_dim_x)
                        epsilon=int(my_dim_x/2)**2)
X2ToField_kind = "Constant"
X2ToField_options = dict(value=0.45)

# for y to the solution field (u)
my_dim_y = 32
my_Y_kind = "Linear"
Y1ToField_kind = my_Y_kind
Y1ToField_options = dict(n_bfs_s1=my_dim_y,
                        n_bfs_s2=my_dim_y,
                        # epsilon=my_dim_y)
                        epsilon=int(my_dim_y/2)**2)
Y2ToField_kind = my_Y_kind
Y2ToField_options = dict(n_bfs_s1=my_dim_y,
                        n_bfs_s2=my_dim_y,
                        # epsilon=my_dim_y)
                        epsilon=int(my_dim_y/2)**2)

# for theta to the weighting functions (w)
my_dim_theta = 32
my_theta_kind = "Linear"
Theta1ToField_kind = my_theta_kind
Theta1ToField_options = dict(n_bfs_s1=my_dim_theta,
                            n_bfs_s2=my_dim_theta,
                            # epsilon=my_dim_theta)
                            epsilon=int(my_dim_theta/2)**2,
                            flag_norm = False)
Theta2ToField_kind = my_theta_kind
Theta2ToField_options = dict(n_bfs_s1=my_dim_theta,
                            n_bfs_s2=my_dim_theta,
                            # epsilon=my_dim_theta)
                            epsilon=int(my_dim_theta/2)**2,
                            flag_norm = False)

# Field for ground truth creation
# Ground truth

# for RBF (small)
# groundtruth_kind = "Circular"
# groundtruth_options = {"Inclusion_1": {"center_x": 0.7, "center_y": 0.7, "radius": 0.2, "value": 0.7},
#                        "Inclusion_2": {"center_x": 0.35, "center_y": 0.35, "radius": 0.15, "value": 0.55}}

# # RBF (large)
# groundtruth_kind = "Circular"
# groundtruth_options = {"Inclusion_1": {"center_x": 0.7, "center_y": 0.8, "radius": 0.1, "value": 0.08},
#                        "Inclusion_2": {"center_x": 0.35, "center_y": 0.35, "radius": 0.1, "value": 0.06}}

# for linear
groundtruth_kind = "Circular"
groundtruth_options = {"Inclusion_1": {"center_x": 0.7, "center_y": 0.7, "radius": 0.2, "value": 1.6},
                       "Inclusion_2": {"center_x": 0.35, "center_y": 0.35, "radius": 0.15, "value": 1.1}}

# for x to the ground-truth material fields (MF)
my_dim_x = 64
XGT1ToField_kind = "Linear"
XGT1ToField_options = dict(n_bfs_s1=my_dim_x,
                        n_bfs_s2=my_dim_x,
                        # epsilon=my_dim_x)
                        epsilon=int(my_dim_x/2)**2)
BC_mask_xGT1_kind = "None"
BC_mask_xGT1_options = dict()

# Dirichlet BCs
# BCs for x
BC_mask_x1_kind = "None"
BC_mask_x1_options = dict()
BC_mask_x2_kind = "None"
BC_mask_x2_options = dict()

# BCs for y
BC_given_explicitly = True
BC_given_y_flag = True
BC_mask_y1_kind = "None" #left_zero"
BC_mask_y1_options = dict()
BC_mask_y2_kind = "None" #bottom_zero"
BC_mask_y2_options = dict()

# BCs for theta
BC_given_theta_flag = True
BC_mask_theta1_kind = "None"# "left_zero"
BC_mask_theta1_options = dict()
BC_mask_theta2_kind = "None"# "bottom_zero"
BC_mask_theta2_options = dict()

# %% Priors & posteriors
# values for Prior x
prior_x_kind = "Normal" # "FieldNormal"
prior_x_options = dict(mean=1.,
                        sigma=2.**2,
                        SpecialInput=None)

# values for Prior y
prior_y_kind = "MVN"
prior_y_options = dict(mean=0,
                        sigma=1e8 ** 2,  # some big number for uninformative prior
                        SpecialInput=None)

flag_jump_prior = True
flag_penalty_learned = True
prior_jumpPenalty_options = dict(a_0=1e-15, b_0=1e-15, SpecialInput="x")
prior_jumps_options = dict(SpecialInput=["x", "jumpPenalty"])
posterior_jumpPenalty_options = dict(a=1, b=1, SpecialInput=None)

# prior_lambda_kind = "MVN"  # "Gamma"
# prior_lambda_options = dict(a_0=1e-20,
#                             b_0=1e-20)
# prior_lambda_options = dict(value=1e0,
#                             cov=1e15)

posterior_x_kind = "FFNReducedMVN" #"ReducedMVN" #"MVN"#"Delta" #"CNNDelta"
# posterior_x_options = dict(mean_x_0=0.3,
#                             cov_x_0=1.0,
#                             SpecialInput=None,#["y"],
#                             reduced_dim_x=10)
posterior_x_options = dict(mean_x_0=0.0,
                            cov_x_0=0.1,
                            SpecialInput=["y"],
                            reduced_dim_x=10,
                            hidden_layers=[2000, 2000, 2000],
                            activation_func_name="SiLU",
                            rescale_input_flag=True,
                            learn_cov_seperatly=True)
                            # learn_cov_seperatly=True,

# posterior_x_options = dict(mean_x_0=0.0,
#                             cov_x_0=0.5,
#                             SpecialInput=None,#["y"],
#                             reduced_dim_x=10)                            

# posterior_y_kind = 'ReducedMVN' #"MVN"  # "DoubleDelta" #"DiagMVN"
# posterior_y_options = dict(mean_y_0=0.0,
#                             cov_y_0=0.05,
#                             reduced_dim_y=10,
#                             #SpecialInput=["x"])
#                             SpecialInput=None)
# posterior_x_kind = "CNNDelta"
# posterior_x_options =       dict(activation_func_name= "ReLU",                                
#                             pic_size= 128,
#                             input_channels= 2,
#                             feature_dim_layers= [16, 20, 24, 15, 6],
#                             average_pooling_layers= [True, True, True, False, False],
#                             output_size= 34**2,
#                             hidden_layers= [],
#                             SpecialInput= "y")#,
#                             output_trafo_source= "Linear")

posterior_y_kind = "ReducedMVN" #'ReducedMVN' #"MVN"  # "DoubleDelta" #"DiagMVN"
posterior_y_options = dict(mean_y_0=0,
                            cov_y_0=0.15,
                            reduced_dim_y=10,
                            #SpecialInput=["x"])
                            SpecialInput=None)


# %% Generel model parameters
# (maximum) Number of iterations (in each iteration the optimization Phi and Theta is called)
max_MinMax_iterations = int(1e6)

# Optimization Parameters
# Random Residual algorithm
samples_theta_kind = "Circular" #"Circular"  #"Random", "Full", "RandomFull"
num_samples_theta = 200
WF_options = {"min_rad": 0.001, "max_rad": 0.15}

lrs_y = [1e-3, 1e-4]#, 1e-5]
lrs_X = [1e-3]

# OptPhi (minimizing ELBO with SVI via ADAM)
num_iter_Phi = 10  # not relevant atm # number of total iterations per min_max iteration 
num_sample_Phi = 10
min_MinMax_iterations_first_iteration = int(5e5)
min_MinMax_iterations = int(5e5)
min_MinMax_iterations_final_iteration = int(5e5)
AMSGrad_first_iteration = False 
# either learn with seperate lr for x & y (flag_phi_seperate_update=True) or... 
flag_phi_seperate_update = True
lr_Phi_X = 1.0e-4
lr_Phi_Y = 1e-5
# ... with one for both (flag_phi_seperate_update=False)
lr_Phi = 1e-4

# at which iterations should the learning rate be decreased
lr_temp_steps = [] # [1, 5]
lr_decrease_factor = 1 / 10 # by how much (factor!) should the learning rate be decreased

# tempering scheme
likelihood_type = "Laplace" # "Gaussian", "Laplace", "Alternative"
alt_likelihood_r_0 = 1e-3 # only for "Alternative"
Lambda_0 = 1e7  # initial temperature
temepering_factor = 10 #1.58481 # (temepering_factor-1.0)*100% increase in temperature
max_temp = 10 ** 10  # maximum temperature (i.e. lambda)
num_temp_steps = 1  # number of temperature steps

# Optimzation of y
flag_skip_opt_y = False
num_iter_Y = 10 # not relevant atm
num_sample_Y = 10
tol_y_mean_convergence = 5.0e-5
min_iter_y_list = [2e4, 2e4]#[2.5e4, 5e4]

# Optimzation of x
flag_skip_opt_x = True
num_iter_X = 10 # not relevant atm
num_sample_X = 10
lr_X = 1e-4
tol_x_mean_convergence = 1.0e-4
min_iter_x = 3e5

# convergence criterion
# every how many iterations should we check for convergence?
when_check_convergence = 1000
# relative tolerance for convergence (e.g. drop one order of magnitude in 1e4 iterations -> -1/3e4)
tol_grad_ELBO_convergence = 1.0e-3 
# convergence criterion for counting the number of sign changes in the gradients for mean of x-posterior
# convergence_sign_percentage = 0.25
change_tol_w_step_size = False # if True, convergence tolerance is multiplied by lr_decrease_factor at lr_temp_steps

when_save_parameters = 25_000
