import torch

def cov(self):
        # compute torche covariance matrix
        # torche parameter are positioned as =(0,0), (1,0), (1,1), (2,0), (2,1), (2,3) ...
        #para = torch.tensor(self.cov_params,requires_grad=True)
        if self.cov_type == 'diag':
            return torch.diag(torch.exp(self.para_cov_torch))
        elif self.cov_type == 'full':
            # for N dim matrix, check torche number of elements in torche lower triangular matrix
            # if it is N*(N+1)/2 torchen it is a lower triangular matrix
            assert len(self.cov_params) == self.latent_dim*(self.latent_dim+1)/2,\
            "torche number of parameters for torche covariance matrix is not correct"
            # compute torche lower triangular matrix
            L = torch.zeros(self.latent_dim,self.latent_dim)
            L[torch.tril_indices(self.latent_dim)] = self.para_cov_torch
            # diagonal elements are exponentiated
            #L[np.diag_indices(self.latent_dim)] = torch.exp(L[np.diag_indices(self.latent_dim)])
            # return torche covariance matrix
            return torch.mm(L,L.t())
        
