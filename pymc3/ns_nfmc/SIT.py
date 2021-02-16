import torch
import torch.nn as nn
import numpy as np
import time
import math
from SlicedWasserstein import *
from RQspline import *
import torch.multiprocessing as mp
import copy


class SIT(nn.Module):

    #sliced iterative transport model
    
    def __init__(self, ndim):
        
        super().__init__()
        
        self.layer = nn.ModuleList([])
        self.ndim = ndim
    
    def forward(self, data, start=0, end=None, param=None):
        
        if data.ndim == 1:
            data = data.view(1,-1)
        if end is None:
            end = len(self.layer)
        elif end < 0:
            end += len(self.layer)
        if start < 0:
            start += len(self.layer)
        
        assert start >= 0 and end >= 0 and end >= start

        logj = torch.zeros(data.shape[0], device=data.device)
        
        for i in range(start, end):
            data, log_j = self.layer[i](data, param=param)
            logj += log_j

        return data, logj
    
    
    def inverse(self, data, start=None, end=0, d_dz=None, param=None):

        if data.ndim == 1:
            data = data.view(1,-1)
        if end < 0:
            end += len(self.layer)
        if start is None:
            start = len(self.layer)
        elif start < 0:
            start += len(self.layer)
        
        assert start >= 0 and end >= 0 and end <= start

        logj = torch.zeros(data.shape[0], device=data.device)
        
        for i in reversed(range(end, start)):
            if d_dz is None:
                data, log_j = self.layer[i].inverse(data, param=param)
            else:
                data, log_j, d_dz = self.layer[i].inverse(data, d_dz=d_dz, param=param)
            logj += log_j

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz


    def transform(self, data, start, end, param=None):

        if start is None:
            return self.inverse(data=data, start=start, end=end, param=param) 
        elif end is None:
            return self.forward(data=data, start=start, end=end, param=param) 
        elif start < 0:
            start += len(self.layer)
        elif end < 0:
            end += len(self.layer)
        
        if start < 0:
            start = 0
        elif start > len(self.layer):
            start = len(self.layer)
        if end < 0:
            end = 0
        elif end > len(self.layer):
            end = len(self.layer)

        if start <= end:
            return self.forward(data=data, start=start, end=end, param=param) 
        else:
            return self.inverse(data=data, start=start, end=end, param=param) 
    
    
    def add_layer(self, layer, position=None):
        
        if position is None or position == len(self.layer):
            self.layer.append(layer)
        else:
            if position < 0:
                position += len(self.layer)
            assert position >= 0 and position < len(self.layer)
            self.layer.insert(position, layer)
        
        return self
    
    
    def delete_layer(self, position=-1):
        
        if position == -1 or position == len(self.layer)-1:
            self.layer = self.layer[:-1]
        else:
            if position < 0:
                position += len(self.layer)
            assert position >= 0 and position < len(self.layer)-1
            
            for i in range(position, len(self.layer)-1):
                self.layer._modules[str(i)] = self.layer._modules[str(i + 1)]
            self.layer = self.layer[:-1]
        
        return self
    
    
    def evaluate_density(self, data, start=0, end=None, param=None):
        
        data, logj = self.forward(data, start=start, end=end, param=param)
        logq = -self.ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(data.reshape(len(data), self.ndim)**2,  dim=1)/2
        logp = logj + logq
        
        return logp


    def loss(self, data, start=0, end=None, param=None):
        return -torch.mean(self.evaluate_density(data, start=start, end=end, param=param))
    
    
    def sample(self, nsample, start=None, end=0, device=torch.device('cuda'), param=None):

        #device must be the same as the device of the model
        
        x = torch.randn(nsample, self.ndim, device=device)
        logq = -self.ndim/2.*torch.log(torch.tensor(2.*math.pi)) - torch.sum(x**2,  dim=1)/2
        x, logj = self.inverse(x, start=start, end=end, param=param)
        logp = logj + logq

        return x, logp



class logit(nn.Module):

    #logit transform

    def __init__(self, lambd=1e-5):

        super().__init__()
        self.lambd = lambd


    def forward(self, data, param=None):

        assert torch.min(data) >= 0 and torch.max(data) <= 1

        data = self.lambd + (1 - 2 * self.lambd) * data 
        logj = torch.sum(-torch.log(data*(1-data)) + math.log(1-2*self.lambd), axis=1)
        data = torch.log(data) - torch.log1p(-data)
        return data, logj


    def inverse(self, data, param=None):

        data = torch.sigmoid(data) 
        logj = torch.sum(-torch.log(data*(1-data)) + math.log(1-2*self.lambd), axis=1)
        data = (data - self.lambd) / (1. - 2 * self.lambd) 
        return data, logj



class whiten(nn.Module):

    #whiten layer

    def __init__(self, ndim_data, scale=True, ndim_latent=None):

        super().__init__()
        if ndim_latent is None:
            ndim_latent = ndim_data
        assert ndim_latent <= ndim_data
        self.ndim_data = ndim_data
        self.ndim_latent = ndim_latent
        self.scale = scale

        self.mean = nn.Parameter(torch.zeros(ndim_data))
        self.D = nn.Parameter(torch.ones(ndim_data))
        self.E = nn.Parameter(torch.eye(ndim_data))
        select = torch.zeros(ndim_data, dtype=torch.bool)
        select[:ndim_latent] = True
        self.register_buffer('select', select)


    def fit(self, data):

        assert data.ndim == 2 and data.shape[1] == self.ndim_data

        with torch.no_grad():
            self.mean[:] = torch.mean(data, dim=0)
            data0 = data - self.mean
            covariance = data0.T @ data0 / (data0.shape[0]-1)
            D, E = torch.symeig(covariance, eigenvectors=True)
            self.D[:] = torch.flip(D, dims=(0,))
            self.E[:] = torch.flip(E, dims=(1,))

            return self


    def forward(self, data, param=None):

        assert data.shape[1] == self.ndim_latent 
        data0 = data - self.mean

        if self.scale:
            D1 = self.D[self.select]**(-0.5)
            data0 = (torch.diag(D1) @ (self.E.T @ data0.T)[self.select]).T
            logj = torch.repeat_interleave(torch.sum(torch.log(D1)), len(data))
        else:
            data0 = (self.E.T @ data0.T)[self.select].T
            logj = torch.zeros(len(data), device=data.device)

        return data0, logj


    def inverse(self, data, d_dz=None, param=None):

        #d_dz: (len(data), self.ndim_latent, n_z)

        assert data.shape[1] == self.ndim_latent 
        if d_dz is not None:
            assert d_dz.shape[0] == data.shape[0] and data.shape[1] == self.ndim_latent and d_dz.shape[1] == self.ndim_latent

        data0 = torch.zeros([data.shape[0], self.ndim_data], device=data.device)
        data0[:, self.select] = data[:]
        if self.scale:
            D1 = self.D**0.5
            D1[~self.select] = 0.
            data0 = (self.E @ torch.diag(D1) @ data0.T).T
            logj = -torch.repeat_interleave(torch.sum(torch.log(D1[self.select])), len(data))
            if d_dz is not None:
                d_dz = torch.einsum('lj,j,ijk->ilk', self.E[:,self.select], D1[self.select], d_dz)
        else:
            data0 = (self.E @ data0.T).T
            logj = torch.zeros(len(data), device=data.device)
            if d_dz is not None:
                d_dz = torch.einsum('lj,ijk->ilk', self.E[:,self.select], d_dz)
        data0 += self.mean

        if d_dz is None:
            return data0, logj
        else:
            return data0, logj, d_dz



def start_timing():
    if torch.cuda.is_available():
        tstart = torch.cuda.Event(enable_timing=True)
        tstart.record()
    else:
        tstart = time.time()
    return tstart



def end_timing(tstart):
    if torch.cuda.is_available():
        tend = torch.cuda.Event(enable_timing=True)
        tend.record()
        torch.cuda.synchronize()
        t = tstart.elapsed_time(tend) / 1000.
    else:
        t = time.time() - tstart
    return t



def _transform_batch_layer(layer, data, logj, index, batchsize, start_index=0, end_index=None, direction='forward', param=None):

    if torch.cuda.is_available():
        gpu = index % torch.cuda.device_count()
        device = torch.device('cuda:%d'%gpu)
    else:
        device = torch.device('cpu')
    
    layer = layer.to(device)

    if end_index is None:
        end_index = len(data)

    i = 0
    while i * batchsize < end_index-start_index:
        start_index0 = start_index + i * batchsize 
        end_index0 = min(start_index + (i+1) * batchsize, end_index) 
        if direction == 'forward': 
            if param is None:
                data1, logj1 = layer.forward(data[start_index0:end_index0].to(device), param=param)
            else:
                data1, logj1 = layer.forward(data[start_index0:end_index0].to(device), param=param[start_index0:end_index0].to(device))
        else: 
            if param is None:
                data1, logj1 = layer.inverse(data[start_index0:end_index0].to(device), param=param)
            else:
                data1, logj1 = layer.inverse(data[start_index0:end_index0].to(device), param=param[start_index0:end_index0].to(device))
        data[start_index0:end_index0] = data1.to(data.device)
        logj[start_index0:end_index0] = logj[start_index0:end_index0] + logj1.to(logj.device)
        i += 1

    del data1, logj1, layer 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return


def transform_batch_layer(layer, data, batchsize, logj=None, direction='forward', param=None, pool=None):
    
    assert direction in ['forward', 'inverse']
    
    if logj is None:
        logj = torch.zeros(len(data), device=data.device)
    
    if pool is None: 
        _transform_batch_layer(layer, data, logj, 0, batchsize, direction=direction, param=param) 
    else:
        if torch.cuda.is_available():
            nprocess = torch.cuda.device_count()
        else:
            nprocess = mp.cpu_count()
        param0 = [(layer, data, logj, i, batchsize, len(data)*i//nprocess, len(data)*(i+1)//nprocess, direction, param) for i in range(nprocess)]
        pool.starmap(_transform_batch_layer, param0)
    
    return data, logj



def _transform_batch_model(model, data, logj, index, batchsize, start_index=0, end_index=None, start=0, end=None, param=None):

    if torch.cuda.is_available():
        gpu = index % torch.cuda.device_count()
        device = torch.device('cuda:%d'%gpu)
    else:
        device = torch.device('cpu')
    
    model = model.to(device)

    if end_index is None:
        end_index = len(data)

    i = 0
    while i * batchsize < end_index-start_index:
        start_index0 = start_index + i * batchsize 
        end_index0 = min(start_index + (i+1) * batchsize, end_index) 
        if param is None:
            data1, logj1 = model.transform(data[start_index0:end_index0].to(device), start=start, end=end, param=param)
        else:
            data1, logj1 = model.transform(data[start_index0:end_index0].to(device), start=start, end=end, param=param[start_index0:end_index0].to(device))
        data[start_index0:end_index0] = data1.to(data.device)
        logj[start_index0:end_index0] = logj[start_index0:end_index0] + logj1.to(logj.device)
        i += 1

    del data1, logj1, model 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return


def transform_batch_model(model, data, batchsize, logj=None, start=0, end=None, param=None, pool=None):
    
    if logj is None:
        logj = torch.zeros(len(data), device=data.device)
    
    if pool is None: 
        _transform_batch_model(model, data, logj, 0, batchsize, start=start, end=end, param=param) 
    else:
        if torch.cuda.is_available():
            nprocess = torch.cuda.device_count()
        else:
            nprocess = mp.cpu_count()
        param0 = [(model, data, logj, i, batchsize, len(data)*i//nprocess, len(data)*(i+1)//nprocess, start, end, param) for i in range(nprocess)]
        pool.starmap(_transform_batch_model, param0)
    
    return data, logj



class SlicedTransport(nn.Module):

    #1 layer of sliced transport
    def __init__(self, ndim, n_component=None, interp_nbin=200):

        super().__init__()
        self.ndim = ndim
        if n_component is None:
            self.n_component = ndim
        else:
            self.n_component = n_component
        self.interp_nbin = interp_nbin

        wi = torch.randn(self.ndim, self.n_component)
        Q, R = torch.qr(wi)
        L = torch.sign(torch.diag(R))
        wT = (Q * L)

        self.wT = nn.Parameter(wT)
        self.transform1D = RQspline(self.n_component, interp_nbin)


    def fit_wT(self, data, sample='gaussian', ndata_wT=None, MSWD_p=2, MSWD_max_iter=200, pool=None, verbose=True):

        #fit the directions to apply 1D transform

        if verbose:
            tstart = start_timing()

        if ndata_wT is None or ndata_wT > len(data):
            ndata_wT = len(data)
        if sample != 'gaussian':
            if ndata_wT > len(sample):
                ndata_wT = len(sample)
            if ndata_wT == len(sample):
                sample = sample.to(self.wT.device)
            else:
                sample = sample[torch.randperm(len(sample), device=sample.device)[:ndata_wT]].to(self.wT.device)
        if ndata_wT == len(data):
            data = data.to(self.wT.device)
        else:
            data = data[torch.randperm(len(data), device=data.device)[:ndata_wT]].to(self.wT.device)
        wT, SWD = maxSWDdirection(data, x2=sample, n_component=self.n_component, maxiter=MSWD_max_iter, p=MSWD_p)
        with torch.no_grad():
            SWD, indices = torch.sort(SWD, descending=True)
            wT = wT[:,indices]
            self.wT[:] = torch.qr(wT)[0] 

        if verbose:
            t = end_timing(tstart)
            print ('Fit wT:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())
        return self 


    def fit_spline(self, data, edge_bins=0, derivclip=None, extrapolate='regression', alpha=(0.9,0.99), noise_threshold=0, MSWD_p=2, KDE=True, bw_factor=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing()
            
            if noise_threshold > 0:
                SWD = SlicedWasserstein_direction(data, self.wT.to(data.device), second='gaussian', p=MSWD_p)
                above_noise = SWD > noise_threshold
            else:
                above_noise = torch.ones(self.wT.shape[1], dtype=bool, device=self.wT.device) 

            data0 = (data @ self.wT.to(data.device)).to(self.wT.device)

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots_gaussian(data0, interp_nbin=self.interp_nbin, above_noise=above_noise, edge_bins=edge_bins, 
                                                  derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor=bw_factor, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart)
                try:
                    print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())
                except:
                    print ('Fit spline Time:', t)

            return above_noise.any()


    def fit_spline_inverse(self, data, sample, edge_bins=4, derivclip=1, extrapolate='regression', alpha=(0,0), noise_threshold=0, MSWD_p=2, KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing()

            if noise_threshold > 0:
                SWD = SlicedWasserstein_direction(data, self.wT.to(data.device), second=sample, p=MSWD_p)
                above_noise = SWD > noise_threshold
            else:
                above_noise = torch.ones(self.wT.shape[1], dtype=bool, device=self.wT.device) 

            data0 = (data @ self.wT.to(data.device)).to(self.wT.device)
            sample0 = (sample @ self.wT.to(sample.device)).to(self.wT.device)

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots(data0, sample0, interp_nbin=self.interp_nbin, above_noise=above_noise, edge_bins=edge_bins, derivclip=derivclip, 
                                         extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart)
                try:
                    print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())
                except:
                    print ('Fit spline Time:', t)

            return above_noise.any() 


    def transform(self, data, mode='forward', d_dz=None, param=None):

        data0 = data @ self.wT
        remaining = data - data0 @ self.wT.T
        if mode is 'forward':
            data0, logj = self.transform1D(data0)
        elif mode is 'inverse':
            data0, logj = self.transform1D.inverse(data0)
            if d_dz is not None:
                d_dz0 = torch.einsum('ijk,jl->ilk', d_dz, self.wT)
                remaining_d_dz = d_dz - torch.einsum('ijk,lj->ilk', d_dz0, self.wT)
                d_dz0 /= torch.exp(logj[:,:,None])
                d_dz = remaining_d_dz + torch.einsum('ijk,lj->ilk', d_dz0, self.wT)
        logj = torch.sum(logj, dim=1)
        data = remaining + data0 @ self.wT.T

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz


    def forward(self, data, param=None):
        return self.transform(data, mode='forward', param=param)


    def inverse(self, data, d_dz=None, param=None):
        return self.transform(data, mode='inverse', d_dz=d_dz, param=param)




def Shift(data, shift):
    if shift[0] != 0:
        shiftx = shift[0]
        left = data.shape[1] - shiftx
        temp = torch.clone(data[:,:shiftx,:,:])
        data[:,:left,:,:] = torch.clone(data[:,shiftx:,:,:])
        data[:,left:,:,:] = temp
    if shift[1] != 0:
        shifty = shift[1]
        left = data.shape[2] - shifty
        temp = torch.clone(data[:,:,:shifty,:])
        data[:,:,:left,:] = torch.clone(data[:,:,shifty:,:])
        data[:,:,left:,:] = temp
    return data
     

def UnShift(data, shift):
    if shift[0] != 0:
        shiftx = shift[0]
        left = data.shape[1] - shiftx
        temp = torch.clone(data[:,left:,:,:])
        data[:,shiftx:,:,:] = torch.clone(data[:,:left,:,:])
        data[:,:shiftx,:,:] = temp
    if shift[1] != 0:
        shifty = shift[1]
        left = data.shape[2] - shifty
        temp = torch.clone(data[:,:,left:,:])
        data[:,:,shifty:,:] = torch.clone(data[:,:,:left,:])
        data[:,:,:shifty,:] = temp
    return data


class PatchSlicedTransport(nn.Module):

    #1 layer of patch based sliced transport 

    def __init__(self, shape=[28,28,1], kernel=[4,4,1], shift=[0,0], n_component=None, interp_nbin=200):

        assert shift[0] >= 0 and shift[0] < shape[0]
        assert shift[1] >= 0 and shift[1] < shape[1]
        assert len(shape) == 3 and len(kernel) == 3 and len(shift) == 2
        assert (kernel[0] <= shape[0]) and (kernel[1] <= shape[1])
        if shape[-1] == 1:
            assert kernel[-1] == 1
        else:
            assert (kernel[-1] == 1) or (kernel[-1] == shape[-1])

        super().__init__()
        self.register_buffer('shape', torch.tensor(shape))
        self.register_buffer('kernel', torch.tensor(kernel)) 
        self.register_buffer('shift', torch.tensor(shift))
        
        self.ndim_sub = (self.kernel[0]*self.kernel[1]*self.kernel[2]).item()

        if n_component is None:
            self.n_component = self.ndim_sub 
        else:
            self.n_component = n_component
            assert n_component <= self.ndim_sub
        self.interp_nbin = interp_nbin
        
        self.Nkernel_H = (self.shape[0] // self.kernel[0]).item()
        self.Nkernel_W = (self.shape[1] // self.kernel[1]).item()
        self.Nkernel_C = (self.shape[2] // self.kernel[2]).item()
        self.Nkernel = self.Nkernel_H * self.Nkernel_W * self.Nkernel_C 

        wT = torch.zeros(self.Nkernel, self.ndim_sub, self.n_component)
        for i in range(self.Nkernel):
            wi = torch.randn(self.ndim_sub, self.n_component)
            Q, R = torch.qr(wi)
            L = torch.sign(torch.diag(R))
            wT[i] = (Q * L)

        self.wT = nn.Parameter(wT)
        self.transform1D = RQspline(self.Nkernel*self.n_component, interp_nbin)


    @staticmethod
    def _fit_wT_patch(data, sample, wT, SWD, dim, index, HWC, kernel, n_component, ndata_wT, max_iter):

        if torch.cuda.is_available():
            gpu = index % torch.cuda.device_count()
            device = torch.device('cuda:%d'%gpu)
            device0 = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
            device0 = torch.device('cpu')
        H, W, C = HWC
        h = index // (W*C)
        w = (index-h*W*C) // C
        c = index - h*W*C - w*C
        if C == 1:
            dim0 = dim[h*kernel[0]:(h+1)*kernel[0], w*kernel[1]:(w+1)*kernel[1], :].reshape(-1).to(device)
        else:
            dim0 = dim[h*kernel[0]:(h+1)*kernel[0], w*kernel[1]:(w+1)*kernel[1], c].reshape(-1).to(device)
        if ndata_wT == len(data):
            data0 = data[:, dim0].to(device)
        else:
            data0 = data[torch.randperm(len(data), device=data.device)[:ndata_wT]][:, dim0].to(device)
        if sample is 'gaussian':
            sample0 = 'gaussian'
        elif ndata_wT == len(sample):
            sample0 = sample[:, dim0].to(device)
        else:
            sample0 = sample[torch.randperm(len(sample), device=sample.device)[:ndata_wT]][:, dim0].to(device)
        wT0, SWD0 = maxSWDdirection(data0, sample0, n_component=n_component, maxiter=max_iter)
        del data0, sample0, dim0
        with torch.no_grad():
            SWD0, indices = torch.sort(SWD0, descending=True)
            SWD[index] = SWD0.to(SWD.device)
            wT0 = wT0[:, indices]
            wT[index] = torch.qr(wT0)[0].to(wT.device)

        del SWD0, indices, wT0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def fit_wT(self, data, sample='gaussian', ndata_wT=None, MSWD_max_iter=200, pool=None, verbose=True):

        #fit the directions to apply 1D transform

        if verbose:
            tstart = start_timing()

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        dim = torch.arange(data.shape[1], device=data.device).reshape(1, *self.shape)
        dim = Shift(dim, self.shift)[0]

        if ndata_wT is None:
            ndata_wT = len(data)
        elif ndata_wT > len(data):
            ndata_wT = len(data)
        if sample != 'gaussian' and ndata_wT > len(sample):
            ndata_wT = len(sample)

        SWD = torch.zeros(self.Nkernel, self.n_component, device=device)

        HWC = (self.Nkernel_H, self.Nkernel_W, self.Nkernel_C)
        if pool is not None:
            param = [(data, sample, self.wT, SWD, dim, index, HWC, self.kernel, self.n_component, ndata_wT, MSWD_max_iter) for index in range(self.Nkernel)]
            pool.starmap(self._fit_wT_patch, param)

        else:
            for index in range(self.Nkernel):
                self._fit_wT_patch(data, sample, self.wT, SWD, dim, index, HWC, self.kernel, self.n_component, ndata_wT, MSWD_max_iter)

        if verbose:
            t = end_timing(tstart)
            print ('Fit wT:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())

        return self 
    

    def construct_wT(self):

        dim = torch.arange(torch.prod(self.shape), device=self.wT.device).reshape(1, *self.shape)
        dim = Shift(dim, self.shift)[0]
        Ntransform = self.Nkernel*self.n_component
        wT = torch.zeros(torch.prod(self.shape), Ntransform, device=self.wT.device)

        for h in range(self.Nkernel_H):
            for w in range(self.Nkernel_W):
                for c in range(self.Nkernel_C):
                    if self.Nkernel_C == 1:
                        dim0 = dim[h*self.kernel[0]:(h+1)*self.kernel[0], w*self.kernel[1]:(w+1)*self.kernel[1], :].reshape(-1)
                    else:
                        dim0 = dim[h*self.kernel[0]:(h+1)*self.kernel[0], w*self.kernel[1]:(w+1)*self.kernel[1], c].reshape(-1)
                    index = h*self.Nkernel_W*self.Nkernel_C + w*self.Nkernel_C + c
                    wT[dim0, self.n_component*index:self.n_component*(index+1)] = self.wT[index]
                
        return wT


    def fit_spline(self, data, edge_bins=0, derivclip=None, extrapolate='regression', alpha=(0.9,0.99), noise_threshold=0, KDE=True, bw_factor=1, batchsize=None, verbose=True):

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing()

            wT = self.construct_wT().to(data.device)

            if noise_threshold > 0:
                SWD = SlicedWasserstein_direction(data, wT, second='gaussian')
                above_noise = (SWD > noise_threshold).to(self.wT.device)
            else:
                above_noise = torch.ones(wT.shape[1], dtype=bool, device=self.wT.device)

            data0 = (data @ wT).to(self.wT.device)

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots_gaussian(data0, interp_nbin=self.interp_nbin, above_noise=above_noise, edge_bins=edge_bins, 
                                                  derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor=bw_factor, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart)
                try:
                    print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD.reshape(self.Nkernel, self.n_component).tolist())
                except:
                    print ('Fit spline Time:', t)
            return above_noise.any() 


    def fit_spline_inverse(self, data, sample, edge_bins=4, derivclip=1, extrapolate='regression', alpha=(0,0), noise_threshold=0, KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing()

            wT = self.construct_wT().to(data.device)

            if noise_threshold > 0:
                SWD = SlicedWasserstein_direction(data, wT, second=sample, batchsize=16)
                above_noise = (SWD > noise_threshold).to(self.wT.device)
            else:
                above_noise = torch.ones(wT.shape[1], dtype=bool, device=self.wT.device)
            data0 = (data @ wT).to(self.wT.device)
            sample0 = (sample @ wT).to(self.wT.device)

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots(data0, sample0, interp_nbin=self.interp_nbin, above_noise=above_noise, edge_bins=edge_bins, derivclip=derivclip,
                                         extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart)
                try:
                    print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD.reshape(self.Nkernel, self.n_component).tolist())
                except:
                    print ('Fit spline Time:', t)

            return above_noise.any()


    def transform(self, data, mode='forward', d_dz=None, param=None):

        wT = self.construct_wT()

        data0 = data @ wT
        remaining = data - data0 @ wT.T
        if mode is 'forward':
            data0, logj = self.transform1D(data0)
        elif mode is 'inverse':
            data0, logj = self.transform1D.inverse(data0)
            if d_dz is not None:
                d_dz0 = torch.einsum('ijk,jl->ilk', d_dz, wT)
                remaining_d_dz = d_dz - torch.einsum('ijk,lj->ilk', d_dz0, wT)
                d_dz0 /= torch.exp(logj[:,:,None])
                d_dz = remaining_d_dz + torch.einsum('ijk,lj->ilk', d_dz0, wT)
        logj = torch.sum(logj, dim=1)
        data = remaining + data0 @ wT.T

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz
    

    def forward(self, data, param=None):
        return self.transform(data, mode='forward', param=param)


    def inverse(self, data, d_dz=None, param=None):
        return self.transform(data, mode='inverse', d_dz=d_dz, param=param)



class ConditionalSlicedTransport_discrete(nn.Module):

    #1 layer of discrete conditional sliced transport
    def __init__(self, ndim, n_class, n_component=None, interp_nbin=100):

        super().__init__()
        self.ndim = ndim
        self.n_class = n_class
        if n_component is None:
            self.n_component = ndim
        else:
            self.n_component = n_component
        self.interp_nbin = interp_nbin

        wi = torch.randn(self.ndim, self.n_component)
        Q, R = torch.qr(wi)
        L = torch.sign(torch.diag(R))
        wT = (Q * L)

        self.wT = nn.Parameter(wT)
        self.transform1D = nn.ModuleList([RQspline(self.n_component, interp_nbin) for i in range(self.n_class)]) 


    def fit_wT(self, data, sample='gaussian', MSWD_p=2, MSWD_max_iter=200, verbose=True):

        #fit the directions to apply 1D transform

        if verbose:
            tstart = start_timing()

        wT, SWD = maxSWDdirection(data, x2=sample, n_component=self.n_component, maxiter=MSWD_max_iter, p=MSWD_p)
        with torch.no_grad():
            SWD, indices = torch.sort(SWD, descending=True)
            wT = wT[:,indices]
            self.wT[:] = torch.qr(wT)[0] 

        if verbose:
            t = end_timing(tstart)
            print ('Fit wT:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())
        return self 


    def fit_spline(self, data, label, edge_bins=0, derivclip=None, extrapolate='regression', alpha=(0.9, 0.99), noise_threshold=0, MSWD_p=2, KDE=True, bw_factor=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing()
            data0 = data @ self.wT

            SWD = []
            #build rational quadratic spline transform
            for binid in range(self.n_class):
                select = label == binid
                SWD1 = SlicedWasserstein_direction(data0[select], None, second='gaussian', p=MSWD_p)
                SWD.append(SWD1.tolist())
                x, y, deriv = estimate_knots_gaussian(data0[select], interp_nbin=self.interp_nbin, above_noise=(SWD1>noise_threshold), edge_bins=edge_bins, 
                                                      derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor=bw_factor, batchsize=batchsize)
                self.transform1D[binid].set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart)
                print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD)

            return SWD


    def fit_spline_inverse(self, data, sample, data_label, sample_label, edge_bins=4, derivclip=1, extrapolate='regression', alpha=(0, 0), noise_threshold=0, MSWD_p=2, KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing()
            data0 = data @ self.wT
            sample0 = sample @ self.wT

            SWD = []

            #build rational quadratic spline transform
            for binid in range(self.n_class):
                select_data = data_label == binid
                select_sample = sample_label == binid
                SWD1 = SlicedWasserstein_direction(data0[select_data], None, second=sample0[select_sample], p=MSWD_p)
                SWD.append(SWD1.tolist())

                x, y, deriv = estimate_knots(data0[select_data], sample0[select_sample], interp_nbin=self.interp_nbin, above_noise=(SWD1>noise_threshold), edge_bins=edge_bins, derivclip=derivclip, 
                                             extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize)
                self.transform1D[binid].set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart)
                print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD)

            return SWD


    def transform(self, data, label, mode='forward', d_dz=None):

        data0 = data @ self.wT
        remaining = data - data0 @ self.wT.T
        logj = torch.zeros(len(data), device=data.device)
        data1 = torch.zeros_like(data0)
        if mode is 'forward':
            for binid in range(self.n_class):
                select = label == binid
                data1[select], logj1 = self.transform1D[binid](data0[select])
                logj[select] = torch.sum(logj1, dim=1)
        elif mode is 'inverse':
            for binid in range(self.n_class):
                select = label == binid
                data1[select], logj1 = self.transform1D[binid].inverse(data0[select])
                logj[select] = torch.sum(logj1, dim=1)
            if d_dz is not None:
                d_dz0 = torch.einsum('ijk,jl->ilk', d_dz, self.wT)
                remaining_d_dz = d_dz - torch.einsum('ijk,lj->ilk', d_dz0, self.wT)
                d_dz0 /= torch.exp(logj[:,:,None])
                d_dz = remaining_d_dz + torch.einsum('ijk,lj->ilk', d_dz0, self.wT)
        data = remaining + data1 @ self.wT.T

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz


    def forward(self, data, param):
        return self.transform(data, param, mode='forward')


    def inverse(self, data, param, d_dz=None):
        return self.transform(data, param, mode='inverse', d_dz=d_dz)



class ConditionalPatchSlicedTransport_discrete(nn.Module):

    #1 layer of conditional patch based sliced transport 

    def __init__(self, n_class=10, shape=[28,28,1], kernel=[4,4,1], shift=[0,0], n_component=None, interp_nbin=200):

        assert shift[0] >= 0 and shift[0] < shape[0]
        assert shift[1] >= 0 and shift[1] < shape[1]
        assert len(shape) == 3 and len(kernel) == 3 and len(shift) == 2
        assert (kernel[0] <= shape[0]) and (kernel[1] <= shape[1])
        if shape[-1] == 1:
            assert kernel[-1] == 1
        else:
            assert (kernel[-1] == 1) or (kernel[-1] == shape[-1])

        super().__init__()
        self.register_buffer('shape', torch.tensor(shape))
        self.register_buffer('kernel', torch.tensor(kernel)) 
        self.register_buffer('shift', torch.tensor(shift))
        
        self.n_class = n_class
        self.ndim_sub = (self.kernel[0]*self.kernel[1]*self.kernel[2]).item()

        if n_component is None:
            self.n_component = self.ndim_sub 
        else:
            self.n_component = n_component
            assert n_component <= self.ndim_sub
        self.interp_nbin = interp_nbin
        
        self.Nkernel_H = (self.shape[0] // self.kernel[0]).item()
        self.Nkernel_W = (self.shape[1] // self.kernel[1]).item()
        self.Nkernel_C = (self.shape[2] // self.kernel[2]).item()
        self.Nkernel = self.Nkernel_H * self.Nkernel_W * self.Nkernel_C 

        wT = torch.zeros(self.Nkernel, self.ndim_sub, self.n_component)
        for i in range(self.Nkernel):
            wi = torch.randn(self.ndim_sub, self.n_component)
            Q, R = torch.qr(wi)
            L = torch.sign(torch.diag(R))
            wT[i] = (Q * L)

        self.wT = nn.Parameter(wT)
        self.transform1D = nn.ModuleList([RQspline(self.Nkernel*self.n_component, interp_nbin) for i in range(self.n_class)]) 

    
    def fit_wT(self, data, sample='gaussian', MSWD_p=2, MSWD_max_iter=200, verbose=True):

        #fit the directions to apply 1D transform

        if verbose:
            tstart = start_timing()

        dim = torch.arange(data.shape[1], device=data.device).reshape(1, *self.shape)
        dim = Shift(dim, self.shift)[0]

        SWD = torch.zeros(self.Nkernel, self.n_component, device=data.device)

        for h in range(self.Nkernel_H):
            for w in range(self.Nkernel_W):
                for c in range(self.Nkernel_C):
                    if self.Nkernel_C == 1:
                        dim0 = dim[h*self.kernel[0]:(h+1)*self.kernel[0], w*self.kernel[1]:(w+1)*self.kernel[1], :].reshape(-1)
                    else:
                        dim0 = dim[h*self.kernel[0]:(h+1)*self.kernel[0], w*self.kernel[1]:(w+1)*self.kernel[1], c].reshape(-1)
                    index = h*self.Nkernel_W*self.Nkernel_C + w*self.Nkernel_C + c
                    data0 = data[:, dim0]
                    if sample is 'gaussian':
                        sample0 = 'gaussian'
                    else:
                        sample0 = sample[:, dim0]
                    wT, SWD[index] = maxSWDdirection(data0, sample0, n_component=self.n_component, maxiter=MSWD_max_iter, p=MSWD_p)
                    del data0, sample0
                    with torch.no_grad():
                        SWD[index], indices = torch.sort(SWD[index], descending=True)
                        wT = wT[:, indices]
                        self.wT[index] = torch.qr(wT)[0]

        if verbose:
            t = end_timing(tstart)
            print ('Fit wT:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())

        return self 


    def construct_wT(self):

        dim = torch.arange(torch.prod(self.shape), device=self.wT.device).reshape(1, *self.shape)
        dim = Shift(dim, self.shift)[0]
        Ntransform = self.Nkernel*self.n_component
        wT = torch.zeros(torch.prod(self.shape), Ntransform, device=self.wT.device)

        for h in range(self.Nkernel_H):
            for w in range(self.Nkernel_W):
                for c in range(self.Nkernel_C):
                    if self.Nkernel_C == 1:
                        dim0 = dim[h*self.kernel[0]:(h+1)*self.kernel[0], w*self.kernel[1]:(w+1)*self.kernel[1], :].reshape(-1)
                    else:
                        dim0 = dim[h*self.kernel[0]:(h+1)*self.kernel[0], w*self.kernel[1]:(w+1)*self.kernel[1], c].reshape(-1)
                    index = h*self.Nkernel_W*self.Nkernel_C + w*self.Nkernel_C + c
                    wT[dim0, self.n_component*index:self.n_component*(index+1)] = self.wT[index]
                
        return wT


    def fit_spline(self, data, label, edge_bins=0, derivclip=None, extrapolate='regression', alpha=(0.9,0.99), noise_threshold=0, MSWD_p=2, KDE=True, bw_factor=1, batchsize=None, verbose=True):

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing()

            wT = self.construct_wT()
            data0 = data @ wT

            SWD = []
            #build rational quadratic spline transform
            for binid in range(self.n_class):
                select = label == binid
                SWD1 = SlicedWasserstein_direction(data0[select], None, second='gaussian', p=MSWD_p)
                SWD.append(SWD1.tolist())
                x, y, deriv = estimate_knots_gaussian(data0[select], interp_nbin=self.interp_nbin, above_noise=(SWD1>noise_threshold), edge_bins=edge_bins, 
                                                      derivclip=derivclip, extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor=bw_factor, batchsize=batchsize)
                self.transform1D[binid].set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart)
                print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD)

            return SWD


    def fit_spline_inverse(self, data, sample, data_label, sample_label, edge_bins=4, derivclip=1, extrapolate='regression', alpha=(0,0), noise_threshold=0, MSWD_p=2, KDE=True, bw_factor_data=1, bw_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.interp_nbin > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing()

            wT = self.construct_wT()
            data0 = data @ wT
            sample0 = sample @ wT

            SWD = []

            #build rational quadratic spline transform
            for binid in range(self.n_class):
                select_data = data_label == binid
                select_sample = sample_label == binid
                SWD1 = SlicedWasserstein_direction(data0[select_data], None, second=sample0[select_sample], p=MSWD_p)
                SWD.append(SWD1.tolist())

                x, y, deriv = estimate_knots(data0[select_data], sample0[select_sample], interp_nbin=self.interp_nbin, above_noise=(SWD1>noise_threshold), edge_bins=edge_bins, derivclip=derivclip, 
                                             extrapolate=extrapolate, alpha=alpha, KDE=KDE, bw_factor_data=bw_factor_data, bw_factor_sample=bw_factor_sample, batchsize=batchsize)
                self.transform1D[binid].set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart)
                print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD)

            return SWD


    def transform(self, data, label, mode='forward', d_dz=None):

        wT = self.construct_wT()

        data0 = data @ wT
        remaining = data - data0 @ wT.T
        logj = torch.zeros(len(data), device=data.device)
        data1 = torch.zeros_like(data0)
        if mode is 'forward':
            for binid in range(self.n_class):
                select = label == binid
                data1[select], logj1 = self.transform1D[binid](data0[select])
                logj[select] = torch.sum(logj1, dim=1)
        elif mode is 'inverse':
            for binid in range(self.n_class):
                select = label == binid
                data1[select], logj1 = self.transform1D[binid].inverse(data0[select])
                logj[select] = torch.sum(logj1, dim=1)
            if d_dz is not None:
                d_dz0 = torch.einsum('ijk,jl->ilk', d_dz, self.wT)
                remaining_d_dz = d_dz - torch.einsum('ijk,lj->ilk', d_dz0, self.wT)
                d_dz0 /= torch.exp(logj[:,:,None])
                d_dz = remaining_d_dz + torch.einsum('ijk,lj->ilk', d_dz0, self.wT)
        data = remaining + data1 @ wT.T

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz
    

    def forward(self, data, param=None):
        return self.transform(data, param, mode='forward')


    def inverse(self, data, d_dz=None, param=None):
        return self.transform(data, param, mode='inverse', d_dz=d_dz)

