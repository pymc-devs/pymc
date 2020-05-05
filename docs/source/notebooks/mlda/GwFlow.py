import numpy as np
import matplotlib.pyplot as plt
from fenics import *
set_log_level(30)


class GwFlowSolver:
    def __init__(self, resolution,
                 field_mean, field_stdev):
        """
        This class solves the steady state
        groundwater flow equation (Darcy)
        on a unit square with some
        simple boundary conditions.
        """
        # Internatise conductivity field parameters.
        self.field_mean = field_mean
        self.field_stdev = field_stdev

        # Head at inflow and outflow.
        self.h_in = 1
        self.h_out = 0

        # Zero flow through boundaries.
        self.q_0 = Constant(0.0)
        
        # Create mesh and define function space
        self.nx = resolution[0]
        self.ny = resolution[1]
        self.mesh = UnitSquareMesh(self.nx, self.ny)

        self.V = FunctionSpace(self.mesh, 'Lagrange', 1)
        self.K = Function(self.V)
        self.n = FacetNormal(self.mesh)
        self.d2v = dof_to_vertex_map(self.V)
        
        # Define variational problem
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        self.data = None
        self.Q = None
        self.q = None
        self.h = None
        self.conductivity = None

        # Define the subdomains.
        sub_domains = MeshFunction('size_t', self.mesh,
                                   self.mesh.topology().dim() - 1)
        sub_domains.set_all(0)

        # Sub domain for no-flow (mark whole boundary,
        # inflow and outflow will later be overwritten)
        class Noflow(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        # Sub domain for inflow (left)
        class Inflow(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < DOLFIN_EPS

        # Sub domain for outflow (right)
        class Outflow(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > 1.0 - DOLFIN_EPS

        # Create instances of the boundaries
        # and mark them with some numbers.
        noflow = Noflow()
        noflow.mark(sub_domains, 0)

        inflow = Inflow()
        inflow.mark(sub_domains, 1)

        outflow = Outflow()
        outflow.mark(sub_domains, 2)

        # Impose the subdomain numbering on external boundary ds.
        self.ds = ds(subdomain_data=sub_domains)

        # Set Dirichlet BC's
        bc_left = DirichletBC(self.V, self.h_in, inflow)
        bc_right = DirichletBC(self.V, self.h_out, outflow)
        self.bcs = [bc_left, bc_right]
        
        # Define the variational problem and create an assembler.
        F = inner(grad(self.v), self.K * grad(self.u)) * dx \
            - self.v * self.q_0 * self.ds(0)
        a, l = lhs(F), rhs(F)
        self._A = PETScMatrix()
        self._b = PETScVector()
        self._assembler = SystemAssembler(a, l, self.bcs)
        self._solver = PETScKrylovSolver('gmres', 'ilu')
        self._solver.set_operator(self._A)
        
    def plot_mesh(self):
        """
        This method plots the mesh
        """
        plt.figure(figsize=(10, 10))
        plot(self.mesh)
        plt.show()
        
    def set_conductivity(self, random_field=False):
        self.conductivity = Function(self.V)
        
        # Set the conductivity
        if np.any(random_field):
            # Make it exponential
            self.conductivity.vector()[:] = \
                np.exp(self.field_mean +
                       self.field_stdev*random_field)

        # If no field is given, just set
        # the flow-field to the mean.
        else:
            self.conductivity.vector()[:] = \
                np.exp(self.field_mean *
                       np.ones(self.mesh.coordinates().shape[0]))
            
        # Map the random field vector to the domain.
        self.K.assign(self.conductivity)
    
    def solve(self):
        """
        Solve the variational problem
        """
        self.h = Function(self.V)
        self._assembler.assemble(self._A, self._b)
        self._solver.solve(self.h.vector(), self._b)
        
    def compute_flow(self):
        self.Q = VectorFunctionSpace(self.mesh,
                                     'Lagrange', 1)
        self.q = project(-self.K*grad(self.h),
                         self.Q)
        
    def get_data(self, datapoints):
        """
        Return data from a set of points.
        """
        self.data = np.zeros(len(datapoints))
        for i, datapoint in enumerate(datapoints):
            self.data[i] = self.h(datapoint[0],
                                  datapoint[1])
        return self.data

    def get_outflow(self):
        """
        This method works without computing the flow first.
        """
        return assemble(dot(- self.K * grad(self.h),
                            self.n) * self.ds(2))

    def plot_solution(self):
        """
        Plot the solution.
        """
        plt.figure(figsize=(12, 10))
        p = plot(self.h, cmap='plasma')
        plt.colorbar(p)
        plt.show()
