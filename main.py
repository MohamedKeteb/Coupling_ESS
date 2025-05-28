import numpy as np

# Log-likelihood for a multivariate Gaussian
def log_likelihood(x, m, v):
    """Unnormalized log Gaussian: -0.5 * (x - m)^T V^{-1} (x - m)"""
    diff = x - m
    inv_v = np.linalg.inv(v)
    return -0.5 * diff.T @ inv_v @ diff

def transition(x, m, v):
    """
    One step of Elliptical Slice Sampling (ESS)
    
    Parameters:
        x (np.ndarray): Current state
        m (np.ndarray): Mean of Gaussian target
        v (np.ndarray): Covariance of Gaussian target
        
    Returns:
        np.ndarray: New sample from the ESS transition
    """
    d = len(x)
    w = np.random.randn(d)  # w ~ N(0, I)
    
    log_p_x = log_likelihood(x, m, v)
    log_y = log_p_x + np.log(np.random.uniform(0, 1))  # Slice threshold

    # Initial bracket
    theta = np.random.uniform(0, 2 * np.pi)
    theta_min = theta - 2 * np.pi
    theta_max = theta

    # First proposal
    proposal = x * np.cos(theta) + w * np.sin(theta)
    log_p_proposal = log_likelihood(proposal, m, v)

    # Shrinkage loop
    while log_p_proposal <= log_y:
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = np.random.uniform(theta_min, theta_max)
        proposal = x * np.cos(theta) + w * np.sin(theta)
        log_p_proposal = log_likelihood(proposal, m, v)

    return proposal

def elliptical_slice_sampling(x_0, m, v, n_iter):
    """
    Run ESS for n_iter steps starting from x_0

    Returns:
        np.ndarray: array of shape (n_iter + 1, d)
    """
    d = len(x_0)
    samples = np.zeros((n_iter + 1, d))
    samples[0] = x_0
    x = x_0

    for i in range(1, n_iter + 1):
        x = transition(x, m, v)
        samples[i] = x

    return samples

# Example usage

