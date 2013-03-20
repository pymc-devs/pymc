from dist_math import * 

__all__ = ['log', 'simplex']


@quickclass
def transform(name, forward, backward, jacobian_det): 

    def apply(dist): 
        @quickclass
        def TransformedDistribtuion():
            support = dist.support 

            def logp(x): 
                return dist.logp(backward(x)) + jacobian_det(x)

            if hasattr(dist, "mode"): 
                mode = forward(dist.mode)
            if hasattr(dist, "median"): 
                mode = forward(dist.median)

            return locals()

        return TransformedDistribtuion()

    def __str__():
        return name + " transform"

    return locals()

log = transform("log", log, exp, idfn)

simplex = transform("simplex",
        lambda p: p[:-1],
        lambda p: concatenate([p, 1- sum(p, keepdims = True)]),
        lambda p: constant([0]))
