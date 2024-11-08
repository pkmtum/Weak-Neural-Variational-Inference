# Weak neural variational inference for solving Bayesian inverse problems without forward models: applications in elastography


## Abstract
The full open-access paper can be found at: https://www.sciencedirect.com/science/article/pii/S0045782524007473

In this paper, we introduce a novel, data-driven approach for solving highdimensional Bayesian inverse problems based on partial differential equations  (PDEs), called Weak Neural Variational Inference (WNVI). The method complements real measurements with virtual observations derived from the physical model.  In particular, weighted residuals are employed as probes to the governing PDE in  order to formulate and solve a Bayesian inverse problem without ever formulating  nor solving a forward model. The formulation treats the state variables of the physical model as latent variables, inferred using Stochastic Variational Inference (SVI),  along with the usual unknowns. The approximate posterior employed uses neural  networks to approximate the inverse mapping from state variables to the unknowns.  We illustrate the proposed method in a biomedical setting where we infer spatially  varying material properties from noisy tissue deformation data. We demonstrate  that WNVI is not only as accurate and more efficient than traditional methods  that rely on repeatedly solving the (non)linear forward problem as a black-box,  but it can also handle ill-posed forward problems (e.g., with insufficient boundary  conditions).

## Dependencies
- Python
- Fenics (install first and all following using pip)
- Fenics Adjoint
- torch with cuda
- scipy
- matplotlib
- tqdm

## Installation
Install Python and all dependencies mentioned above.
To clone this repo:
```
git clone https://github.com/pkmtum/Weak-Neural-Variational-Inference.git
```

## How to run
We provide the code and input files (in the folder examples, with the naming scheme "input_[placeholder].py) to recreate the following results in the paper:
- Result with different noise levels from Subsection 4.3 ([placeholder]="25dB", "30dB" or "35dB" for the respective noise levels)
- Result with simple X prior used for the comparison with traditional methods from Subsection 4.2 ([placeholder]="SimpleXPrior")
- Result without known Dirichlet boundary conditions from Subsection 4.4 ([placeholder]="NoDirichletBCs")
- Result with non-linear (Neo-hookian) material model from Subsection 4.5 ([placeholder]="NeoHookConstitutiveLaw")

To run the code, replace the existing input file in the main directory ("input.py") with the respective input file you want to use and rename it to "input.py." Execute the main.py, and the code will run and produce the results and plots. Note that the results may vary due to the RNG. Further, the exact plots in the paper were custom-created and were not produced by the code. Code to produce the results with traditional methods is not provided.

In the current form, 24 GB of GPU memory (we used a Nvidia RTX 4090) was used.

## Citation
If this code is relevant to your research, we would be grateful if you cite our work:
```
@article{scholz2024weak,
  title={Weak neural variational inference for solving Bayesian inverse problems without forward models: applications in elastography},
  author={Scholz, Vincent C and Zang, Yaohua and Koutsourelakis, Phaedon-Stelios},
  journal={arXiv preprint arXiv:2407.20697},
  year={2024}
}
```

## Contact
If you have questions or problems regarding the code or paper, please feel invited to reach out to us using the following E-Mails
Dipl.-Ing. Vincent C. Scholz:           vincent.scholz@tum.de
Dr. Yaohua Zang:                        yaohua.zang@tum.de
Prof. Phaedon-Stelios Koutsourelakis:   p.s.koutsourelakis@tum.de
