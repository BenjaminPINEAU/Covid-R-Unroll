import torch
from oplin import discrete_derivation, discrete_derivation_adj

### Functions needed for Primal-dual minimization ###


def proxL1_norm(x, gamma):
    
    """
    Compute the proximal operator of gamma*norm(x)_1
    INPUT: 
    - x: vector of interest
    - gamma: multiplicative factor

    OUTPUT:
    - Proximal operator of gamma*norm(x)_1 evaluated in x
    """

    tmp = torch.abs(x) - gamma*torch.ones(x.shape, device = x.device)
    signs = torch.sign(x)
    return torch.maximum(tmp, torch.zeros(tmp.shape, device = x.device)) * signs


def proxL1_norm_adj(x, gamma):
        
    """
    Compute the proximal operator of (gamma*norm(x)_1)^*
    INPUT: 
    - x: vector of interest
    - gamma: multiplicative factor

    OUTPUT:
    - Proximal operator of (gamma*norm(x)_1)^* evaluated in x
    """
    return torch.nn.functional.hardtanh(x, min_val = -gamma, max_val = gamma)


def prox_DKL(Z, Phi, R, gamma):

    """
    Compute the proximal operator of the Kullback-Leibler divergence (of a Poisson distribution) between 0 and T
    We use the fact that the divergence is 
    INPUT: 
    - Z: daily count
    - Phi: global infectiousness associated to Z
    - R: reproduction number
    - gamma: multiplicative factor

    OUTPUT:
    - proximal operator of DKL evaluated in R
    """

    inside_sqrt = (R - gamma*Phi)**2 + 4*gamma*Z
    prox = ( R - gamma*Phi + torch.sqrt(inside_sqrt) )/2
    #prox[(Phi == 0) * (Z == 0)] = 0
    return prox



def CP_optim(Z, Phi, tau, sigma, lmb, max_iter, eps):

    """
    Implementation of Primal-dual minimization for penalized Kullback-Leibler divergence
    doing optimal computation : does not multipliate matrix but does a vectorial calculation
    INPUT:
    - Z: daily counts
    - Phi: global infectiousness associated to Z
    - tau, sigma: algorithmic parameters
    - lmb: regularisation factor
    - max_iter: maximum of iterations
    - eps: precision for stopping criteria

    OUT: 
    - R_k: reproduction number, the result of the minimization 
    - n_iter: the number of iterations needed to do the minimization
    """

    # Parameters initialization
    R_k = Z
    Q = discrete_derivation(R_k)
    R_bar = R_k.clone()
    n_iter = 0
    obj_function = []

    # Parameters update
    while n_iter < max_iter:
        Q = Q + sigma * discrete_derivation(R_bar) - sigma*proxL1_norm(Q/sigma + discrete_derivation(R_bar), lmb/sigma)
        R_k1 = prox_DKL(Z, Phi, R_k - tau*discrete_derivation_adj(Q), tau)
        R_bar = 2*R_k1 - R_k
        # Test if the precision is reached
        if torch.norm(R_k - R_k1, p = 2) < eps:
            break
        R_k = R_k1
        n_iter += 1

    return R_k, n_iter


def CP_optim_fix(Z, Phi, tau, sigma, lmb, nb_iter):

    """
    Implementation of Primal-dual minimization for penalized Kullback-Leibler
    doing optimal computation : does not multipliate matrix but does a vectorial calculation
    INPUT:
    - Z: daily counts
    - Phi: global infectiousness associated to Z
    - tau, sigma: algorithmic parameters
    - lmb: regularisation factor
    - nb_iter: number of iterations

    OUT: 
    - R_k: reproduction number, the result of the minimization 
    - n_iter: the number of iterations needed to do the minimization
    """

    # Parameters initialization
    R_k = Z
    Q = discrete_derivation(R_k)
    R_bar = R_k.clone()
    n_iter = 0
    obj_function = []

    # Parameters update
    while n_iter < nb_iter:
        Q = Q + sigma * discrete_derivation(R_bar) - sigma*proxL1_norm(Q/sigma + discrete_derivation(R_bar), lmb/sigma)
        R_k1 = prox_DKL(Z, Phi, R_k - tau*discrete_derivation_adj(Q), tau)
        R_bar = 2*R_k1 - R_k
        R_k = R_k1
        n_iter += 1

    return R_k