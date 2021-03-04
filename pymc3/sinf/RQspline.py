import torch
import torch.nn as nn
import math

def Percentile(input, percentiles):
    """
    Find the percentiles of a tensor along the last dimension.
    Adapted from https://github.com/aliutkus/torchpercentile/blob/master/torchpercentile/percentile.py
    """
    percentiles = percentiles.double()
    in_sorted, in_argsort = torch.sort(input, dim=-1)
    positions = percentiles * (input.shape[-1]-1) / 100
    floored = torch.floor(positions)
    ceiled = floored + 1
    ceiled[ceiled > input.shape[-1] - 1] = input.shape[-1] - 1
    weight_ceiled = positions-floored
    weight_floored = 1.0 - weight_ceiled
    d0 = in_sorted[..., floored.long()] * weight_floored
    d1 = in_sorted[..., ceiled.long()] * weight_ceiled
    result = d0+d1
    return result

class kde(object):
    """
    Adapted from Scipy's KDE estimator:
    https://github.com/scipy/scipy/blob/master/scipy/stats/kde.py
    """

    def __init__(self, dataset, bw_factor=None, weights=None, batchsize=None):
        if dataset.ndim == 1:
            self.dataset = dataset[:, None]
        elif dataset.ndim == 2:
            self.dataset = dataset
        else:
            raise ValueError("`dataset` should be a 1-d or 2-d array.")

        self.n, self.d = self.dataset.shape
        self.weights = weights

        if weights is not None:
            self.weights /= torch.sum(self.weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self.weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self.neff = 1 / torch.sum(self.weights**2)
            self.weights = self.weights.view(-1,1)
        else:
            self.neff = self.n
        
        self.bw_factor = bw_factor if bw_factor is not None else 1
        self.factor = self.neff ** (-1. / (self.d + 4)) * self.bw_factor
        
        if weights is None:
            data = self.dataset - torch.mean(self.dataset, dim=0)
            self._data_covariance = data.T @ data / (self.n - 1)
        else:
            data = self.dataset - torch.sum(self.dataset*self.weights, dim=0)
            self._data_covariance = (self.weights * data).T @ data / (1 - torch.sum(self.weights**2))
        
        self._data_inv_cov = torch.pinverse(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = torch.sqrt(torch.det(2 * math.pi * self.covariance))
        self.batchsize = batchsize


    def _diff(self, x, dataset):
        """Utility for evaluating pdf and cdf_1d."""
        points = x[:, None] if x.ndim == 1 else x

        m, d = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = points.view(1, self.d)
                d = self.d
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        return points[None, :, :] - dataset[:, None, :]
        # (# of data, # of points, # of dim)

    def pdf(self, x):
        """Evaluate the estimated pdf on a set of points.
        
        Parameters
        ----------
        x : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.
        
        Returns
        -------
        values : (# of points,)-array
            The values at each point.
        
        Raises
        ------
        ValueError
            If the dimensionality of the input points is different than
            the dimensionality of the KDE.
        
        """
        if self.batchsize is not None:
            result = torch.zeros_like(x)
            i = 0
            while i * self.batchsize < self.n:

                diff = self._diff(x, self.dataset[i*self.batchsize: (i+1)*self.batchsize])
                energy = torch.einsum("lmi,ij,lmj->lm", diff, self.inv_cov / 2, diff)
                if self.weights is None:
                    result += torch.sum(torch.exp(-energy), dim=0) / self._norm_factor / self.n
                else:
                    result += torch.sum(self.weights * torch.exp(-energy), dim=0) / self._norm_factor

                i += 1
        else:
            diff = self._diff(x, self.dataset)
            energy = torch.einsum("lmi,ij,lmj->lm", diff, self.inv_cov / 2, diff)
            if self.weights is None:
                result = torch.sum(torch.exp(-energy), dim=0) / self._norm_factor / self.n
            else:
                result = torch.sum(self.weights * torch.exp(-energy), dim=0) / self._norm_factor

        return result

    __call__ = pdf

    def cdf(self, x):
        """Evaluate the estimated cdf on a set of 1-d points.
        
        Parameters
        ----------
        x : (# of points)-array
            Alternatively, a scalar can be passed in and
            treated as a single point.
        
        Returns
        -------
        values : (# of points,)-array
            The values at each point.
        
        Raises
        ------
        NotImplementedError
            If KDE is not 1-d.
        ValueError
            If the dimensionality of the input points is different than
            the dimensionality of the KDE.
        
        """
        if self.d != 1:
            msg = "currently only supports cdf for 1-d kde"
            raise NotImplementedError(msg)
        
        if self.batchsize is not None:
            i = 0
            result = torch.zeros_like(x)

            while i * self.batchsize < self.n:

                diff = self._diff(x, self.dataset[i*self.batchsize: (i+1)*self.batchsize])[:, :, 0]
                diff_scaled = diff / self.covariance**0.5
                if self.weights is None:
                    result += torch.sum(0.5 * (1 + torch.erf(diff_scaled / 2**0.5)), dim=0) / self.n
                else:
                    result += torch.sum(self.weights * 0.5 * (1 + torch.erf(diff_scaled / 2**0.5)), dim=0)
                
                i += 1
        else:
            diff = self._diff(x, self.dataset)[:, :, 0]
            diff_scaled = diff / self.covariance**0.5
            if self.weights is None:
                result = torch.sum(0.5 * (1 + torch.erf(diff_scaled / 2**0.5)), dim=0) / self.n
            else:
                result = torch.sum(self.weights * 0.5 * (1 + torch.erf(diff_scaled / 2**0.5)), dim=0)

        return result


class RQspline(nn.Module):
    '''
    Ratianal quadratic spline.
    See appendix A.1 of https://arxiv.org/pdf/1906.04032.pdf
    The main advantage compared to cubic spline is that the
    inverse is analytical and does not require binary search

    x: (ndim, nknot) 2d array, each row should be monotonic increasing
    y: (ndim, nknot) 2d array, each row should be monotonic increasing
    deriv: (ndim, nknot) 2d array, should be positive
    '''

    def __init__(self, ndim, nknot):

        super().__init__()
        self.ndim = ndim
        self.nknot = nknot

        x0 = torch.rand(ndim, 1)-4.5
        logdx = torch.log(torch.abs(-2*x0 / (nknot-1)))

        #use log as parameters to make sure monotonicity
        self.x0 = nn.Parameter(x0)
        self.y0 = nn.Parameter(x0.clone())
        self.logdx = nn.Parameter(torch.ones(ndim, nknot-1)*logdx)
        self.logdy = nn.Parameter(torch.ones(ndim, nknot-1)*logdx)
        self.logderiv = nn.Parameter(torch.zeros(ndim, nknot))


    def set_param(self, x, y, deriv):

        dx = x[:,1:] - x[:,:-1]
        dy = y[:,1:] - y[:,:-1]
        assert (dx > 0).all()
        assert (dy > 0).all()
        assert (deriv > 0).all()

        self.x0[:] = x[:, 0].view(-1,1)
        self.y0[:] = y[:, 0].view(-1,1)
        self.logdx[:] = torch.log(dx)
        self.logdy[:] = torch.log(dy)
        self.logderiv[:] = torch.log(deriv)


    def _prepare(self):
        #return knot points and derivatives
        xx = torch.cumsum(torch.exp(self.logdx), dim=1)
        xx += self.x0
        xx = torch.cat((self.x0, xx), dim=1)
        yy = torch.cumsum(torch.exp(self.logdy), dim=1)
        yy += self.y0
        yy = torch.cat((self.y0, yy), dim=1)
        delta = torch.exp(self.logderiv)
        return xx, yy, delta

    def forward(self, x):
        # x: (ndata, ndim) 2d array
        xx, yy, delta = self._prepare() #(ndim, nknot)

        index = torch.searchsorted(xx.detach(), x.T.contiguous().detach()).T
        y = torch.zeros_like(x)
        logderiv = torch.zeros_like(x)

        #linear extrapolation
        select0 = index == 0
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[select0]
        y[select0] = yy[dim, 0] + (x[select0]-xx[dim, 0]) * delta[dim, 0]
        logderiv[select0] = self.logderiv[dim, 0]
        selectn = index == self.nknot
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[selectn]
        y[selectn] = yy[dim, -1] + (x[selectn]-xx[dim, -1]) * delta[dim, -1]
        logderiv[selectn] = self.logderiv[dim, -1]

        #rational quadratic spline
        select = ~(select0 | selectn)
        index = index[select]
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[select]
        xi = (x[select] - xx[dim, index-1]) / (xx[dim, index] - xx[dim, index-1])
        s = (yy[dim, index]-yy[dim, index-1]) / (xx[dim, index]-xx[dim, index-1])
        xi1_xi = xi*(1-xi)
        denominator = s + (delta[dim, index]+delta[dim, index-1]-2*s)*xi1_xi
        xi2 = xi**2

        y[select] = yy[dim, index-1] + ((yy[dim, index]-yy[dim, index-1]) * (s*xi2+delta[dim, index-1]*xi1_xi)) / denominator
        logderiv[select] = 2*torch.log(s) + torch.log(delta[dim, index]*xi2 + 2*s*xi1_xi + delta[dim, index-1]*(1-xi)**2) - 2 * torch.log(denominator)

        return y, logderiv

    def inverse(self, y):
        xx, yy, delta = self._prepare()

        index = torch.searchsorted(yy.detach(), y.T.contiguous().detach()).T
        x = torch.zeros_like(y)
        logderiv = torch.zeros_like(y)

        #linear extrapolation
        select0 = index == 0
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[select0]
        x[select0] = xx[dim, 0] + (y[select0]-yy[dim, 0]) / delta[dim, 0]
        logderiv[select0] = self.logderiv[dim, 0]
        selectn = index == self.nknot
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[selectn]
        x[selectn] = xx[dim, -1] + (y[selectn]-yy[dim, -1]) / delta[dim, -1]
        logderiv[selectn] = self.logderiv[dim, -1]

        #rational quadratic spline
        select = ~(select0 | selectn)
        index = index[select]
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[select]
        deltayy = yy[dim, index]-yy[dim, index-1]
        s = deltayy / (xx[dim, index]-xx[dim, index-1])
        delta_2s = delta[dim, index]+delta[dim, index-1]-2*s
        deltay_delta_2s = (y[select]-yy[dim, index-1]) * delta_2s

        a = deltayy * (s-delta[dim, index-1]) + deltay_delta_2s
        b = deltayy * delta[dim, index-1] - deltay_delta_2s
        c = - s * (y[select]-yy[dim, index-1])
        discriminant = b.pow(2) - 4 * a * c
        #discriminant[discriminant<0] = 0 
        assert (discriminant >= 0).all()
        xi = - 2*c / (b + torch.sqrt(discriminant))
        xi1_xi = xi * (1-xi)

        x[select] = xi * (xx[dim, index] - xx[dim, index-1]) + xx[dim, index-1]
        logderiv[select] = 2*torch.log(s) + torch.log(delta[dim, index]*xi**2 + 2*s*xi1_xi + delta[dim, index-1]*(1-xi)**2) - 2 * torch.log(s + delta_2s*xi1_xi)

        return x, logderiv


def estimate_knots_gaussian(data, interp_nbin, above_noise, weight=None, edge_bins=0, derivclip=None, extrapolate='regression', alpha=(0.9, 0.99), KDE=True, bw_factor=1, batchsize=None):

    if not KDE and weight is not None:
        raise NotImplementedError

    start = 100 / (interp_nbin-2*edge_bins+1)
    end = 100-start
    q1 = torch.linspace(start, end, interp_nbin-2*edge_bins, device=data.device)
    if edge_bins > 0:
        start = start / (edge_bins+1)
        end = q1[0]-start
        q0 = torch.linspace(start, end, edge_bins, device=data.device)
        end = 100-start
        start = q1[-1] + start
        q2 = torch.linspace(start, end, edge_bins, device=data.device)
        q = torch.cat((q0,q1,q2), dim=0)
    else:
        q = q1
    x = Percentile(data.T, q).to(torch.get_default_dtype())
    y = x.clone()
    deriv = torch.ones_like(x)

    eps = 1e-5
    for i in range(data.shape[1]):
        if above_noise[i]:
            if KDE:
                rho = kde(data[:,i], bw_factor=bw_factor, weights=weight, batchsize=batchsize)
                scale = (rho.covariance[0,0]+1)**0.5
                y[i] = 2**0.5 * scale * torch.erfinv(2*rho.cdf(x[i]).double()-1).to(torch.get_default_dtype())
                dy = y[i,1:] - y[i,:-1]
                dx = x[i,1:] - x[i,:-1]
                while (dy<=eps).any() or (dx<=eps).any() or torch.isnan(dy).any() or not torch.isfinite(y[i]).all():
                    select = torch.zeros(len(y[i]), dtype=bool, device=y.device)
                    select[1:] = dy <= eps 
                    select[1:] += dx <= eps 
                    select[1:] += torch.isnan(dy)
                    select += ~torch.isfinite(y[i])
                    x[i,select] = torch.rand(torch.sum(select).item(), device=x.device)*(torch.max(data[:,i])-torch.min(data[:,i])) + torch.min(data[:,i]) 
                    x[i] = torch.sort(x[i])[0]
                    y[i] = 2**0.5 * scale * torch.erfinv(2*rho.cdf(x[i]).double()-1).to(torch.get_default_dtype())
                    dy = y[i,1:] - y[i,:-1]
                    dx = x[i,1:] - x[i,:-1]
            else:
                y[i] = 2**0.5 * torch.erfinv(2*q.double()/100.-1).to(torch.get_default_dtype())
                dy = y[i,1:] - y[i,:-1]
                dx = x[i,1:] - x[i,:-1]
                q0 = q.clone()
                while (dy<=eps).any() or (dx<=eps).any() or torch.isnan(dy).any() or not torch.isfinite(y[i]).all():
                    select = torch.zeros(len(y[i]), dtype=bool, device=y.device)
                    select[1:] = dy <= eps 
                    select[1:] += dx <= eps 
                    select[1:] += torch.isnan(dy)
                    select += ~torch.isfinite(y[i])
                    q0[select] = torch.rand(torch.sum(select).item(), device=q.device)*100
                    q0 = torch.sort(q0)[0]
                    x[i] = Percentile(data[:,i], q0).to(torch.get_default_dtype())
                    y[i] = 2**0.5 * torch.erfinv(2*q0.double()/100.-1).to(torch.get_default_dtype())
                    dy = y[i,1:] - y[i,:-1]
                    dx = x[i,1:] - x[i,:-1]
            h = dx
            s = dy / dx
            deriv[i,1:-1] = (s[:-1]*h[1:] + s[1:]*h[:-1]) / (h[1:] + h[:-1])

            if derivclip == 1:
                deriv[i,0] = 1
                deriv[i,-1] = 1
            else:
                if extrapolate == 'endpoint':
                    endx1 = torch.min(data[:,i])
                    endx2 = torch.max(data[:,i])
                    if KDE:
                        deriv[i,0] = (2**0.5 * scale * torch.erfinv(2*rho.cdf(endx1).double()-1).to(torch.get_default_dtype()) - y[i,0]) / (endx1 - x[i,0])
                        deriv[i,-1] = (2**0.5 * scale * torch.erfinv(2*rho.cdf(endx2).double()-1).to(torch.get_default_dtype()) - y[i,-1]) / (endx2 - x[i,-1])
                    else:
                        deriv[i,0] = (2**0.5 * torch.erfinv(2*torch.tensor(0.5/len(data), device=data.device, dtype=torch.float64)-1).to(torch.get_default_dtype())- y[i,0]) / (endx1 - x[i,0])
                        deriv[i,-1] = (2**0.5 * torch.erfinv(2*torch.tensor(1-0.5/len(data), device=data.device, dtype=torch.float64)-1).to(torch.get_default_dtype()) - y[i,-1]) / (endx2 - x[i,-1])
                elif extrapolate == 'regression':
                    endx1 = torch.sort(data[data[:,i]<x[i,0],i])[0]
                    endx2 = torch.sort(data[data[:,i]>x[i,-1],i], descending=True)[0]
                    if KDE:
                        if len(endx1) > 10:
                            endx1 = Percentile(endx1, torch.linspace(0,100,11, device=endx1.device)[1:-1]).to(torch.get_default_dtype())
                        endy1 = 2**0.5 * scale * torch.erfinv(2*rho.cdf(endx1).double()-1).to(torch.get_default_dtype()) - y[i,0]
                        if len(endx2) > 10:
                            endx2 = Percentile(endx2, torch.linspace(0,100,11, device=endx2.device)[1:-1]).to(torch.get_default_dtype())
                        endy2 = 2**0.5 * scale * torch.erfinv(2*rho.cdf(endx2).double()-1).to(torch.get_default_dtype()) - y[i,-1]
                    else:
                        endy1 = 2**0.5 * torch.erfinv(2*torch.linspace(0.5,len(endx1)-0.5,len(endx1),device=data.device,dtype=torch.float64)/len(data)-1).to(torch.get_default_dtype()) - y[i,0]
                        endy2 = 2**0.5 * torch.erfinv(2*(1-torch.linspace(0.5,len(endx2)-0.5,len(endx2),device=data.device,dtype=torch.float64)/len(data))-1).to(torch.get_default_dtype()) - y[i,-1]
                    endx1 -= x[i,0]
                    select1 = torch.isfinite(endy1) & (endy1>0) & (endx1>0)
                    deriv[i,0] = torch.sum(endx1[select1]*endy1[select1]) / torch.sum(endx1[select1]*endx1[select1])
                    endx2 -= x[i,-1]
                    select2 = torch.isfinite(endy2) & (endy2>0) & (endx2>0)
                    deriv[i,-1] = torch.sum(endx2[select2]*endy2[select2]) / torch.sum(endx2[select2]*endx2[select2])
                    if torch.sum(select1) == 0:
                        deriv[i,0] = 1
                    if torch.sum(select2) == 0:
                        deriv[i,-1] = 1

            y[i] = (1-alpha[0]) * y[i] + alpha[0] * x[i]
            deriv[i,1:-1] = (1-alpha[0]) * deriv[i,1:-1] + alpha[0]
            deriv[i,0] = (1-alpha[1]) * deriv[i,0] + alpha[1]
            deriv[i,-1] = (1-alpha[1]) * deriv[i,-1] + alpha[1]

            if derivclip is not None and derivclip > 1:
                deriv[i,0] = torch.clamp(deriv[i,0], 1/derivclip, derivclip)
                deriv[i,-1] = torch.clamp(deriv[i,-1], 1/derivclip, derivclip)

        else:
            dx = x[i,1:] - x[i,:-1]
            while (dx<=eps).any():
                select = torch.zeros(len(x[i]), dtype=bool, device=x.device)
                select[1:] = dx <= eps 
                x[i,select] = torch.rand(torch.sum(select).item(), device=x.device)*(torch.max(data[:,i])-torch.min(data[:,i])) + torch.min(data[:,i])
                x[i] = torch.sort(x[i])[0]
                y[i] = x[i]
                dx = x[i,1:] - x[i,:-1]

    return x, y, deriv


def estimate_knots(data, sample, interp_nbin, above_noise, edge_bins=4, derivclip=1, extrapolate='regression', alpha=(0, 0), KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None):

    start = 100 / (interp_nbin-2*edge_bins+1)
    end = 100-start
    q1 = torch.linspace(start, end, interp_nbin-2*edge_bins, device=data.device)
    if edge_bins > 0:
        start = start / (edge_bins+1)
        end = q1[0]-start
        q0 = torch.linspace(start, end, edge_bins, device=data.device)
        end = 100-start
        start = q1[-1] + start
        q2 = torch.linspace(start, end, edge_bins, device=data.device)
        q = torch.cat((q0,q1,q2), dim=0)
    else:
        q = q1
    y = torch.zeros(sample.shape[1], interp_nbin, device=sample.device)
    for i in range(len(y)):
        y[i] = Percentile(sample[:,i], q)
    x = y.clone()
    deriv = torch.ones_like(y)
    
    if KDE:
        invq = torch.cat((torch.linspace(0, q[0], 5, device=data.device)[:-1], q, torch.linspace(q[-1], 100, 5, device=data.device)[1:]), dim=0)
    else:
        invq = q
    #invy = Percentile(data.T, invq).to(torch.get_default_dtype())
    invy = torch.zeros(data.shape[1], len(invq), device=data.device)
    for i in range(len(invy)):
        invy[i] = Percentile(data[:,i], invq)
    
    eps = 1e-5
    for i in range(data.shape[1]):
        if above_noise[i]:
            if KDE:
                rho = kde(data[:,i], bw_factor=bw_factor_data, batchsize=batchsize)
                rhos = kde(sample[:,i], bw_factor=bw_factor_sample, batchsize=batchsize)

                #inverse cdf
                invx = rho.cdf(invy[i])
                dx = invx[1:] - invx[:-1]
                dy = invy[i, 1:] - invy[i, :-1]
                select = torch.ones(len(invx), dtype=bool, device=invx.device)
                select[1:] = (dy > 0) & (dx > 0)
                invx = invx[select]
                invy1 = invy[i,select]
                h = invx[1:] - invx[:-1]
                s = (invy1[1:] - invy1[:-1]) / h
                invderiv = (s[:-1]*h[1:] + s[1:]*h[:-1]) / (h[1:] + h[:-1])
                invderiv = torch.cat((invderiv[0].view(-1), invderiv, invderiv[-1].view(-1)), dim=0)
                invcdf = RQspline(1, len(invx)).requires_grad_(False).to(data.device)
                invcdf.set_param(invx.view(1,-1), invy1.view(1,-1), invderiv.view(1,-1))

                x[i] = invcdf(rhos.cdf(y[i]).view(-1,1))[0].view(-1)

                dx = x[i,1:] - x[i,:-1]
                dy = y[i,1:] - y[i,:-1]
                while (dx<=eps).any() or (dy<=eps).any() or x[i,0]<=invy1[0] or x[i,-1]>=invy1[-1]:
                    select = torch.zeros(len(y[i]), dtype=bool, device=y.device)
                    select[1:] = dx <= eps 
                    select[1:] += dy <= eps 
                    select += x[i] <= invy1[0]
                    select += x[i] >= invy1[-1]
                    y[i,select] = torch.rand(torch.sum(select).item(), device=y.device)*(torch.max(sample[:,i])-torch.min(sample[:,i])) + torch.min(sample[:,i])
                    y[i] = torch.sort(y[i])[0]
                    x[i] = invcdf(rhos.cdf(y[i]).view(-1,1))[0].view(-1)
                    dx = x[i,1:] - x[i,:-1]
                    dy = y[i,1:] - y[i,:-1]
            else:
                x[i] = invy[i]
                dy = y[i,1:] - y[i,:-1]
                dx = x[i,1:] - x[i,:-1]
                q0 = q.clone()
                while (dy<=eps).any() or (dx<=eps).any():
                    select = torch.zeros(len(y[i]), dtype=bool, device=y.device)
                    select[1:] = dy <= eps 
                    select[1:] += dx <= eps 
                    q0[select] = torch.rand(torch.sum(select).item(), device=y.device)*100
                    q0 = torch.sort(q0)[0]
                    y[i] = Percentile(sample[:,i], q0).to(torch.get_default_dtype())
                    x[i] = Percentile(data[:,i], q0).to(torch.get_default_dtype())
                    dy = y[i,1:] - y[i,:-1]
                    dx = x[i,1:] - x[i,:-1]
            h = dx
            s = dy / dx
            deriv[i,1:-1] = (s[:-1]*h[1:] + s[1:]*h[:-1]) / (h[1:] + h[:-1])

            if derivclip == 1:
                deriv[i,0] = 1
                deriv[i,-1] = 1
            else:
                if extrapolate == 'endpoint':
                    try:
                        endx = torch.min(data[:,i])
                        endy = torch.min(sample[:,i])
                        deriv[i,0] = (endy - y[i,0]) / (endx - x[i,0])
                    except:
                        deriv[i,0] = 1
                    try:
                        endx = torch.max(data[:,i])
                        endy = torch.max(sample[:,i])
                        deriv[i,-1] = (endy - y[i,-1]) / (endx - x[i,-1])
                    except:
                        deriv[i,-1] = 1
                elif extrapolate == 'regression':
                    try:
                        endx = Percentile(data[data[:,i]<x[i,0],i], torch.linspace(0,90,10, device=data.device)) - x[i,0]
                        endy = Percentile(sample[sample[:,i]<y[i,0],i], torch.linspace(0,90,10, device=data.device)) - y[i,0]
                        deriv[i,0] = torch.sum(endy*endy) / torch.sum(endx*endy)
                    except:
                        deriv[i,0] = 1
                    try:
                        endx = Percentile(data[data[:,i]>x[i,-1],i], torch.linspace(10,100,10, device=data.device)) - x[i,-1]
                        endy = Percentile(sample[sample[:,i]>y[i,-1],i], torch.linspace(10,100,10, device=data.device)) - y[i,-1]
                        deriv[i,-1] = torch.sum(endy*endy) / torch.sum(endx*endy)
                    except:
                        deriv[i,-1] = 1

            y[i] = (1-alpha[0]) * y[i] + alpha[0] * x[i]
            deriv[i,1:-1] = (1-alpha[0]) * deriv[i,1:-1] + alpha[0]
            deriv[i,0] = (1-alpha[1]) * deriv[i,0] + alpha[1]
            deriv[i,-1] = (1-alpha[1]) * deriv[i,-1] + alpha[1]

            if derivclip is not None and derivclip > 1:
                deriv[i,0] = torch.clamp(deriv[i,0], 1/derivclip, derivclip)
                deriv[i,-1] = torch.clamp(deriv[i,-1], 1/derivclip, derivclip)
        else:
            dx = x[i,1:] - x[i,:-1]
            while (dx<=eps).any():
                select = torch.zeros(len(x[i]), dtype=bool, device=x.device)
                select[1:] = dx <= eps 
                x[i,select] = torch.rand(torch.sum(select).item(), device=x.device)*(torch.max(data[:,i])-torch.min(data[:,i])) + torch.min(data[:,i])
                x[i] = torch.sort(x[i])[0]
                y[i] = x[i]
                dx = x[i,1:] - x[i,:-1]

    return x, y, deriv
