import torch
import torch.optim as optim


def ObjectiveG(x, pg, p, w=None, perdim=True):
    
    px, indices = torch.sort(x, dim=-1)
    if w is not None:
        pg = Gaussian_ppf(x.shape[-1], weight=w[indices], device=w.device)

    if perdim:
        WD = torch.mean(torch.abs(px-pg) ** p)
    else:
        WD = torch.mean(torch.abs(px-pg) ** p, dim=-1)
    return WD


def Objective(x, x2, p, perdim=True):

    px = torch.sort(x, dim=-1)[0]
    px2 = torch.sort(x2, dim=-1)[0]

    if perdim:
        WD = torch.mean(torch.abs(px-px2) ** p)
    else:
        WD = torch.mean(torch.abs(px-px2) ** p, dim=-1)
    return WD


def Gaussian_ppf(Nsample, weight=None, device=torch.device("cuda:0")):
    if weight is None:
        start = 50 / Nsample
        end = 100-start
        q = torch.linspace(start, end, Nsample, device=device)
    else:
        q = torch.cumsum(weight, dim=1)
        q = q - 0.5*weight
    pg = 2**0.5 * torch.erfinv(2*q/100-1)
    return pg


def noise_WD(ndata, threshold=0.5, N=100, p=2, device=torch.device("cuda:0")):

    #N: The number of times to draw random samples

    assert threshold >= 0 and threshold <= 1

    pg = Gaussian_ppf(ndata, device=device)

    WD = torch.zeros(N, device=device)
    for i in range(N):
        x = torch.randn(ndata, device=device)
        WD[i] = ObjectiveG(x, pg, p) ** (1/p)

    position = threshold * N
    floored = math.floor(position)
    if floored > N-1:
        floored = N-1
    ceiled = floored + 1
    if ceiled > N-1:
        ceiled = N-1
    WD, arg = torch.sort(WD)
    return (WD[floored]*(position-floored) + WD[ceiled]*(1+floored-position)).item()


def maxSWDdirection(x, x2='gaussian', weight=None, n_component=None, maxiter=200,  p=2, eps=1e-6, wi=None):

    #if x2 is None, find the direction of max sliced Wasserstein distance between x and gaussian
    #if x2 is not None, it needs to have the same shape as x

    if x2 != 'gaussian':
        assert x.shape[1] == x2.shape[1]
        assert weight is None
        if x2.shape[0] > x.shape[0]:
            x2 = x2[torch.randperm(x2.shape[0])][:x.shape[0]]
        elif x2.shape[0] < x.shape[0]:
            x = x[torch.randperm(x.shape[0])][:x2.shape[0]]
    elif weight is not None:
        assert len(weight) == len(x)
        pg = None
        weight = weight / torch.sum(weight)
    else:
        pg = Gaussian_ppf(len(x), device=x.device)

    ndim = x.shape[1]
    if n_component is None:
        n_component = ndim

    
    #initialize w. algorithm from https://arxiv.org/pdf/math-ph/0609050.pdf
    if wi is None:
        wi = torch.randn(ndim, n_component, device=x.device)
    else:
        assert wi.shape[0] == ndim and wi.shape[1] == n_component
    Q, R = torch.qr(wi)
    L = torch.sign(torch.diag(R))
    w = (Q * L).T

    lr = 0.1
    down_fac = 0.5
    up_fac = 1.5
    c = 0.5
    
    #algorithm from http://noodle.med.yale.edu/~hdtag/notes/steifel_notes.pdf
    #note that here w = X.T
    #use backtracking line search
    w1 = w.clone()
    for i in range(maxiter):
        w.requires_grad_(True)
        if x2 == 'gaussian':
            loss = -ObjectiveG(w @ x.T, pg, p, w=weight)
        else:
            loss = -Objective(w @ x.T, w @ x2.T, p)
        loss1 = loss
        GT = torch.autograd.grad(loss, w)[0]
        w.requires_grad_(False)
        with torch.no_grad():
            WT = w.T @ GT - GT.T @ w
            e = - w @ WT #dw/dlr
            m = torch.sum(GT * e) #dloss/dlr

            lr /= down_fac
            while loss1 > loss + c*m*lr:
                lr *= down_fac
                if 2*n_component < ndim:
                    UT = torch.cat((GT, w), dim=0).double()
                    V = torch.cat((w.T, -GT.T), dim=1).double()
                    w1 = (w.double() - lr * w.double() @ V @ torch.pinverse(torch.eye(2*n_component, dtype=torch.double, device=x.device)+lr/2*UT@V) @ UT).to(torch.get_default_dtype())
                else:
                    w1 = (w.double() @ (torch.eye(ndim, dtype=torch.double, device=x.device)-lr/2*WT.double()) @ torch.pinverse(torch.eye(ndim, dtype=torch.double, device=x.device)+lr/2*WT.double())).to(torch.get_default_dtype())
            
                if x2 == 'gaussian':
                    loss1 = -ObjectiveG(w1 @ x.T, pg, p, w=weight)
                else:
                    loss1 = -Objective(w1 @ x.T, w1 @ x2.T, p)
        
            if torch.max(torch.abs(w1-w)) < eps:
                w = w1
                break
        
            lr *= up_fac
            w = w1

    if x2 == 'gaussian':
        WD = ObjectiveG(w @ x.T, pg, p, w=weight, perdim=False)
    else:
        WD = Objective(w @ x.T, w @ x2.T, p, perdim=False)
    return w.T, WD**(1/p)


def SlicedWasserstein(data, second='gaussian', Nslice=1000, weight=None, p=2, batchsize=None):

    #Calculate the Sliced Wasserstein distance between the samples of two distribution. 

    if second != 'gaussian':
        assert data.shape[1] == second.shape[1]
        if data.shape[0] < second.shape[0]:
            second = second[torch.randperm(second.shape[0])][:data.shape[0]]
        elif data.shape[0] > second.shape[0]:
            data = data[torch.randperm(data.shape[0])][:second.shape[0]]
    elif weight is not None:
        assert len(weight) == len(data)
        pg = None
    else:
        pg = Gaussian_ppf(len(data), device=data.device)
    Ndim = data.shape[1]

    if batchsize is None:
        direction = torch.randn(Ndim, Nslice).to(data.device)
        direction /= torch.sum(direction**2, dim=0)**0.5
        data0 = data @ direction
        if second == 'gaussian':
            SWD = ObjectiveG(data0.T, pg, p, w=weight, perdim=True)
        else:
            second0 = second @ direction
            SWD = Objective(data0.T, second0.T, p, perdim=True)
    else:
        i = 0
        SWD = torch.zeros(Nslice, device=data.device)
        while i * batchsize < Nslice:
            if (i+1) * batchsize < Nslice:
                direction = torch.randn(Ndim, batchsize).to(data.device)
            else:
                direction = torch.randn(Ndim, Nslice-i*batchsize).to(data.device)
            direction /= torch.sum(direction**2, dim=0)**0.5
            data0 = data @ direction
            if second == 'gaussian':
                SWD[i * batchsize: (i+1) * batchsize] = ObjectiveG(data0.T, pg, p, w=weight, perdim=False)
            else:
                second0 = second @ direction
                SWD[i * batchsize: (i+1) * batchsize] = Objective(data0.T, second0.T, p, perdim=False)
            i += 1
        SWD = torch.mean(SWD)

    return SWD ** (1/p)


def SlicedWasserstein_direction(data, directions=None, second='gaussian', weight=None, p=2, batchsize=None):

    #calculate the Wasserstein distance of 1D slices on given directions

    if directions is None:
        data0 = data
    else:
        data0 = data @ directions
    if second != 'gaussian':
        assert data.shape[1] == second.shape[1]

        second0 = second
        if directions is not None:
            second0 = second0 @ directions

        if data0.shape[0] < second0.shape[0]:
            second0 = second0[torch.randperm(second0.shape[0])[:data0.shape[0]]]
        elif data0.shape[0] > second0.shape[0]:
            data0 = data0[torch.randperm(data0.shape[0])[:second0.shape[0]]]

        if batchsize is None:
            SWD = Objective(data0.T, second0.T, p, perdim=False)
        else:
            SWD = torch.zeros(data0.shape[1], device=data0.device) 
            i = 0
            while i * batchsize < data0.shape[1]:        
                SWD[i * batchsize: (i+1) * batchsize] = Objective(data0[:, i*batchsize: (i+1)*batchsize].T, second0[:, i*batchsize: (i+1)*batchsize].T, p, perdim=False)
                i += 1
    else:
        if weight is not None:
            assert len(weight) == len(data)
            pg = None
        else:
            pg = Gaussian_ppf(len(data), device=data.device)
        if batchsize is None:
            SWD = ObjectiveG(data0.T, pg, p, w=weight, perdim=False)
        else:
            SWD = torch.zeros(data0.shape[1], device=data0.device) 
            i = 0
            while i * batchsize < data0.shape[1]:        
                SWD[i * batchsize: (i+1) * batchsize] = ObjectiveG(data0[:, i*batchsize: (i+1)*batchsize].T, pg, p, w=weight, perdim=False)
                i += 1

    return SWD ** (1/p)


class Stiefel_SGD(optim.Optimizer):
    # Adapted from https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
    # Algorithm from http://noodle.med.yale.edu/~hdtag/notes/steifel_notes.pdf
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Stiefel_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Stiefel_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, grad)
                    if nesterov:
                        G = G.add(momentum, buf.double())
                    else:
                        G = buf.double()
                else:
                    G = p.grad.data.double()

                X = p.data.double()
                dtype = p.data.dtype
                if p.data.ndim == 2: 
                    n_dim = p.data.shape[0]
                    n_component = p.data.shape[1]

                    if 2*n_component < n_dim:
                        U = torch.cat((G, X), dim=1)
                        VT = torch.cat((X.T, -G.T), dim=0)
                        #p.data.add_(-group['lr'], (U@torch.pinverse(torch.eye(2*n_component, dtype=torch.double, device=X.device)+group['lr']/2.*VT@U)@VT@X).type(dtype))
                        p.data = (X - group['lr'] * U@torch.pinverse(torch.eye(2*n_component, dtype=torch.double, device=X.device)+group['lr']/2.*VT@U)@VT@X).type(dtype)
                    else:
                        A = G@X.T - X@G.T
                        p.data = (torch.pinverse(torch.eye(n_dim, dtype=torch.double, device=X.device)+group['lr']/2*A) @ (torch.eye(n_dim, dtype=torch.double, device=X.device)-group['lr']/2*A) @ X).type(dtype)
                elif p.data.ndim == 3:
                    n_dim = p.data.shape[1]
                    n_component = p.data.shape[2]

                    if 2*n_component < n_dim:
                        for i in range(len(p.data)):
                            U = torch.cat((G[i], X[i]), dim=1)
                            VT = torch.cat((X[i].T, -G[i].T), dim=0)
                            p.data[i] = (X[i] - group['lr'] * U@torch.pinverse(torch.eye(2*n_component, dtype=torch.double, device=X.device)+group['lr']/2.*VT@U)@VT@X[i]).type(dtype)
                    else:
                        for i in range(len(p.data)):
                            A = G[i]@X[i].T - X[i]@G[i].T
                            p.data[i] = (torch.pinverse(torch.eye(n_dim, dtype=torch.double, device=X.device)+group['lr']/2*A) @ (torch.eye(n_dim, dtype=torch.double, device=X.device)-group['lr']/2*A) @ X[i]).type(dtype)
                else:
                    raise ValueError('The dimensionality should be 2 or 3')

        return loss

