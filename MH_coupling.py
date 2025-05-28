import numpy as np 
#--------------------------------------------------Metropolis sampler--------------------------------------------
def metropolis_step(x, v, U):
    """
    Metropolis sampler single step.

    Parameters:
    - x: current state (float)
    - v: proposal std deviation (float)
    - U: function to compute -log target density (energy) at x

    Returns:
    - next state (float)
    """
    proposal = x + np.random.normal(0, v)
    log_u = np.log(np.random.uniform(0, 1))

    # Calcul du critère d'acceptation
    accept_prob = U(x) - U(proposal)
    
    if log_u < accept_prob:
        return proposal
    else:
        return x

def metropolis_sampler(x_int, v, U, n_iter):
    samples = [x_int]
    x = x_int
    for _ in range(n_iter):
        x = metropolis_step(x, v, U)
        samples.append(x)
    return samples

#------------------------------------MH Coupling------------------------------------------------------------------
def gaussian_potential(x, m, v):
    return (x-m)**2/(2*v)

def coupling_gaussian(m1, m2, v):
    x = np.random.normal(m1, v)
    log_w = np.log(np.random.uniform(0,1))
    if log_w <= gaussian_potential(x, m1, v) - gaussian_potential(x, m2, v):
        return x, x
    else:
        y = np.random.normal(m2, v)
        log_w = np.log(np.random.uniform(0,1))
        while log_w <= gaussian_potential(y, m2, v) - gaussian_potential(y, m1, v):
            y = np.random.normal(m2, v)
            log_w = np.log(np.random.uniform(0,1))
        
        return x, y 


def MH_coupling_step(x, y, v, U) :
    proposal_x, proposal_y = coupling_gaussian(x, y, v) 

    log_w = np.log(np.random.uniform(0, 1))
    if log_w < U(x) - U(proposal_x):
        x = proposal_x
    if log_w < U(y) - U(proposal_y):
        y = proposal_y
    
    return x, y

def MH_couplig(x_init, y_init, v, U, n_iter):
    samples = [(x_init, y_init)]
    x, y = x_init, y_init
    for _ in range(n_iter):
        x, y = MH_coupling_step(x, y, v, U)
        samples.append(x, y)
    return samples





    
