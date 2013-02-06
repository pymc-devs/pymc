from dist_math import * 

def AR1(a, tau) : 
    def like(x): 
        x_im1 = x[:-1]
        x_i = x[1:]
        boundary = Normal(0, tau)
        return (sum(Normal( x_im1*a, tau/(2*a))(x_i)) 
                + boundary(x[0]) 
                + boundary(x[-1]))
    return like
