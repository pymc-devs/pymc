def stick_breaking(betas):
    '''
    betas is a K-dimensional vector consisting of iid draws from a Beta distribution
    '''
    sticks = at.concatenate([[1], (1 - betas[:-1])])
    
    return at.mul(betas, at.cumprod(sticks))
