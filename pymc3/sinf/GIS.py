from pymc3.sinf.SIT import *


def GIS(data_train, data_validate=None, iteration=None, weight_train=None, weight_validate=None, n_component=None, interp_nbin=None, KDE=True, bw_factor=0.5, alpha=None, edge_bins=None, 
        ndata_wT=None, MSWD_max_iter=None, logit=False, Whiten=False, batchsize=None, nocuda=False, patch=False, shape=[28,28,1], verbose=True):
    
    assert data_validate is not None or iteration is not None
 
    #hyperparameters
    ndim = data_train.shape[1]
    if interp_nbin is None:
        interp_nbin = min(200, int(len(data_train)**0.5))
    if alpha is None:
        alpha = (1-0.02*math.log10(len(data_train)), 1-0.001*math.log10(len(data_train)))
    if edge_bins is None:
        edge_bins = round(math.log10(len(data_train)))-1
    if batchsize is None:
        batchsize = len(data_train)
    if not patch:
        if n_component is None:
            if ndim <= 8 or len(data_train) / float(ndim) < 20:
                n_component = ndim
            else:
                n_component = 8
        if ndata_wT is None:
            ndata_wT = min(len(data_train), int(math.log10(ndim)*1e5))
        if MSWD_max_iter is None:
            MSWD_max_iter = min(len(data_train) // ndim, 200)
    else:
        assert shape[0] > 4 and shape[1] > 4
        n_component0 = n_component
        ndata_wT0 = ndata_wT
        MSWD_max_iter0 = MSWD_max_iter

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if nocuda:
        device = torch.device("cpu")
    data_train = data_train.to(device)
    if weight_train is not None:
        weight_train = weight_train.to(device)

    #define the model
    model = SIT(ndim=ndim).requires_grad_(False).to(device)
    logj_train = torch.zeros(len(data_train), device=device)
    if data_validate is not None:
        data_validate = data_validate.to(device)
        if weight_validate is not None:
            weight_validate = weight_validate.to(device)
        logj_validate = torch.zeros(len(data_validate), device=device)
        best_logp_validate = -1e10
        best_Nlayer = 0
        wait = 0
        maxwait = 10

    #logit transform
    if logit:
        layer = logit(lambd=1e-5).to(device)
        data_train, logj_train = layer(data_train)
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        if data_validate is not None:
            data_validate, logj_validate = layer(data_validate)
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            best_logp_validate = logp_validate
            best_Nlayer = 1

        model.add_layer(layer)
        if verbose:
            if data_validate is not None:
                print('After logit transform logp:', logp_train, logp_validate)
            else:
                print('After logit transform logp:', logp_train)
    
    #whiten
    if Whiten:
        layer = whiten(ndim_data=ndim, scale=True, ndim_latent=ndim).requires_grad_(False).to(device)
        layer.fit(data_train, weight_train)

        data_train, logj_train0 = layer(data_train)
        logj_train += logj_train0
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        if data_validate is not None:
            data_validate, logj_validate0 = layer(data_validate)
            logj_validate += logj_validate0
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            if logp_validate > best_logp_validate:
                best_logp_validate = logp_validate
                best_Nlayer = len(model.layer)

        model.add_layer(layer)
        if verbose:
            if data_validate is not None:
                print('After whiten logp:', logp_train, logp_validate)
            else:
                print('After whiten logp:', logp_train)

    #GIS iterations
    while True:
        t = time.time()
        if patch:
            #patch layers
            if len(model.layer) % 2 == 0:
                kernel = [4, 4, shape[-1]]
                shift = torch.randint(4, (2,)).tolist()
            else:
                kernel = [2, 2, shape[-1]]
                shift = torch.randint(2, (2,)).tolist()
            #hyperparameter
            ndim = np.prod(kernel)
            if n_component0 is None:
                if ndim <= 8 or len(data_train) / float(ndim) < 20:
                    n_component = ndim
                else:
                    n_component = 8
            elif n_component0 > ndim:
                n_component = ndim
            else:
                n_component = n_component0
            if ndata_wT0 is None:
                ndata_wT = min(len(data_train), int(math.log10(ndim)*1e5))
            if MSWD_max_iter0 is None:
                MSWD_max_iter = min(len(data_train) // ndim, 200)
            
            layer = PatchSlicedTransport(shape=shape, kernel_size=kernel, shift=shift, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)
        else:
            #regular GIS layer
            layer = SlicedTransport(ndim=ndim, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).to(device)
        
        #fit the layer
        layer.fit_wT(data=data_train, weight=weight_train, ndata_wT=ndata_wT, MSWD_max_iter=MSWD_max_iter, verbose=verbose)

        layer.fit_spline(data=data_train, weight=weight_train, edge_bins=edge_bins, alpha=alpha, KDE=KDE, bw_factor=bw_factor, batchsize=batchsize, verbose=verbose)

        #update the data
        data_train, logj_train = transform_batch_layer(layer, data_train, batchsize, logj=logj_train, direction='forward')
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        model.add_layer(layer)

        if data_validate is not None:
            data_validate, logj_validate = transform_batch_layer(layer, data_validate, batchsize, logj=logj_validate, direction='forward')
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            if logp_validate > best_logp_validate:
                best_logp_validate = logp_validate
                best_Nlayer = len(model.layer)
                wait = 0
            else:
                wait += 1
            if wait == maxwait:
                model.layer = model.layer[:best_Nlayer]
                break

        if verbose:
            if data_validate is not None: 
                print ('logp:', logp_train, logp_validate, 'time:', time.time()-t, 'iteration:', len(model.layer), 'best:', best_Nlayer)
            else:
                print ('logp:', logp_train, 'time:', time.time()-t, 'iteration:', len(model.layer))

        if iteration is not None and len(model.layer) >= iteration:
            break

    return model
