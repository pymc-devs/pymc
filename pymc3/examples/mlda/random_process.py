import numpy as np
from numpy.linalg import inv, det

import matplotlib.pyplot as plt

from scipy.linalg import eigh
#from scipy.sparse.linalg.eigen.arpack import eigsh 
from scipy.spatial import distance_matrix


class RandomProcess:
    def __init__(self, dolfin_mesh, mkl, lamb):
        
        '''
        This class sets up a random process on a grid and generates
        a realisation of the process, given parameters or a random vector.
        '''
        
        # Internalise the grid and set number of vertices.
        self.mesh = dolfin_mesh
        self.n_points = self.mesh.num_vertices()
        
        # Save the coordinates in some vectors.
        self.x = self.mesh.coordinates()[:,0]; self.y = self.mesh.coordinates()[:,1]
        
        # Set some random field parameters.
        self.mkl = mkl
        self.lamb = lamb
        
        # Create a snazzy distance-matrix for rapid computation of the covariance matrix.
        dist = distance_matrix(self.mesh.coordinates(), self.mesh.coordinates())
        
        # Compute the covariance between all points in the space.
        self.cov =  np.exp(-0.5*dist**2/self.lamb**2)
    
    def plot_covariance_matrix(self):
        
        # Plot the covariance matrix.
        plt.figure(figsize = (10,8)); plt.imshow(self.cov, cmap = 'binary'); plt.colorbar(); plt.show()
    
    def compute_eigenpairs(self):
        
        # Find eigenvalues and eigenvectors using Arnoldi iteration.
        eigvals, eigvecs = eigh(self.cov, eigvals = (self.n_points - self.mkl, self.n_points - 1))
        #eigvals, eigvecs = eigsh(self.cov, self.mkl, which = 'LM')
        
        order = np.flip(np.argsort(eigvals))
        self.eigenvalues = eigvals[order]
        self.eigenvectors = eigvecs[:,order]
      
    def generate(self, parameters = None):
        
        # Generate a random field, see
        # Scarth, C., Adhikari, S., Cabral, P. H., Silva, G. H. C., & Prado, A. P. do. (2019). 
        # Random field simulation over curved surfaces: Applications to computational structural mechanics. 
        # Computer Methods in Applied Mechanics and Engineering, 345, 283–301. https://doi.org/10.1016/j.cma.2018.10.026
        
        if parameters is None:
            self.parameters = np.random.normal(size=self.mkl)
            
        else:
            self.parameters = np.array(parameters).flatten()
        
        self.random_field = np.linalg.multi_dot((self.eigenvectors, 
                                                 np.sqrt(np.diag(self.eigenvalues)), 
                                                 self.parameters))

    def plot(self, lognormal = False):
        
        # Plot the random field.
        if lognormal:
            random_field = np.exp(self.random_field)
            contour_levels = np.linspace(min(random_field), max(random_field), 20)
        else:
            random_field = self.random_field
            contour_levels = np.linspace(min(random_field), max(random_field), 20)

        plt.figure(figsize = (12,10))
        plt.tricontourf(self.x, self.y, random_field, levels = contour_levels, cmap = 'magma'); 
        plt.colorbar()
        plt.show()
