import numpy as np
import fenics as fe
import torch
import matplotlib.pyplot as plt


def bottom(x, on_boundary):
    return (on_boundary and fe.near(x[1], 0.0))

def left(x, on_boundary):
    return (on_boundary and fe.near(x[0], 0.0))

def top(x, on_boundary):
    return (on_boundary and fe.near(x[1], 1.0))

def right(x, on_boundary):
    return (on_boundary and fe.near(x[0], 1.0))


# Strain function
def epsilon(u):
    return 0.5*(fe.grad(u) + fe.grad(u).T)


# Stress function
def sigma(u, lmbda, mu):
    return lmbda*fe.div(u)*fe.Identity(2) + 2*mu*epsilon(u)

def sigma_NeoHook(u, lmbda, mu):
    # Kinematics
    d = 2
    I = fe.Identity(d)              # Identity tensor
    F = I + fe.grad(u)              # Deformation gradient
    B = F*F.T                       # Left Cauchy-Green tensor

    # Invariants of deformation tensors
    J  = fe.det(F)

    # stress (directly)
    sigma = mu/J * (B-I) + lmbda * (J-1) * I
    return sigma

def solve_hook_pde_fenics(E, nu, rhs, plotSol=False):
    # Define the mesh
    mesh = fe.UnitSquareMesh(E.shape[0]-1, E.shape[1]-1)

    # Define the function space
    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    W = fe.FunctionSpace(mesh, 'P', 1)

    # Define the trial and test functions
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    # Define the conductivity field as a FEniCS function
    c = fe.Function(W)

    # flatten to get it to the same shape as the mesh
    fl_E = E.flatten()
    # get the mesh to vertex map
    d2v = fe.dof_to_vertex_map(W)
    # sort the values from mesh-like to vertex-like
    sort_fl_E = fl_E[d2v.astype("int64")]

    # Map the nodal values to the Function object
    c.vector().set_local(np.asarray(sort_fl_E))

    # --------------------
    # Material fields
    # --------------------
    mu = c / (2*(1 + nu))
    lmbda = c*nu / ((1 + nu)*(1 - 2*nu))

    # --------------------
    # Boundary conditions
    # --------------------
    bc = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), bottom)

    # --------------------
    # Weak form
    # --------------------

    a = fe.inner(sigma(u, lmbda, mu), epsilon(v))*fe.dx
    L = fe.dot(fe.Constant(rhs), v)*fe.dx #- fe.inner(g, u_test)*ds(1)

    # Solve the PDE
    u = fe.Function(V)
    fe.solve(a == L, u, bc)
    #fe.plot(u)
    if plotSol:
        fe.plot(u, mode="displacement")
        plt.show()
        fe.plot(c)
        plt.show()

    u_array = u.compute_vertex_values(mesh)

    # Reshape the numpy array to a 2D grid
    u_grid = np.reshape(u_array, (2, E.shape[0], E.shape[1]))

    u_torch = torch.from_numpy(u_grid)

    return u_torch


def solve_hook_pde_fenics_2(E, nu, rhs, f=[0.0, 0.0], plotSol=False):
        # Define the mesh
    mesh = fe.UnitSquareMesh(E.shape[0]-1, E.shape[1]-1)

    # Define the function space
    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    W = fe.FunctionSpace(mesh, 'P', 1)

    # Define the trial and test functions
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    # Define the conductivity field as a FEniCS function
    c = fe.Function(W)

    # flatten to get it to the same shape as the mesh
    fl_E = E.flatten()
    # get the mesh to vertex map
    d2v = fe.dof_to_vertex_map(W)
    # sort the values from mesh-like to vertex-like
    sort_fl_E = fl_E[d2v.astype("int64")]

    # Map the nodal values to the Function object
    c.vector().set_local(np.asarray(sort_fl_E))

    # --------------------
    # Material fields
    # --------------------
    mu = c / (2*(1 + nu))
    lmbda = c*nu / ((1 + nu)*(1 - 2*nu))

    # --------------------
    # Boundary conditions (Dirichlet)
    # --------------------
    bc1 = fe.DirichletBC(V.sub(1), fe.Constant(0.0), bottom)
    bc2 = fe.DirichletBC(V.sub(0), fe.Constant(0.0), left)
    bcs = [bc1, bc2]

    # --------------------
    # Boundary conditions (Neumann)
    # --------------------
    # create a mesh function which assigns an unsigned integer (size_t) to each edge
    mf = fe.MeshFunction("size_t", mesh, 1) # 3rd argument is dimension of an edge
    mf.set_all(0) # initialize the function to zero
    # create a SubDomain subclass, specifying the portion of the boundary with x[0] < 1/2
    class Boundary_right(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and fe.near(x[0], 1.0))
    class Boundary_top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and fe.near(x[1], 1.0))
    B_right = Boundary_right() # instantiate it
    B_top = Boundary_top() # instantiate it
    # use this lefthalf object to set values of the mesh function to 1 in the subdomain
    B_right.mark(mf, 1)
    B_top.mark(mf, 2)
    # define a new measure ds based on this mesh function
    ds = fe.Measure('ds', domain=mesh, subdomain_data=mf)

    r1, r2 = rhs["neumann_right"][0].to("cpu").numpy(), rhs["neumann_right"][1].to("cpu").numpy()
    t1, t2 = rhs["neumann_top"][0].to("cpu").numpy(), rhs["neumann_top"][1].to("cpu").numpy()
    r1 = fe.Constant(r1)
    r2 = fe.Constant(r2)
    t1 = fe.Constant(t1)
    t2 = fe.Constant(t2)
    g_right = fe.Expression(('r1', 'r2'), r1=r1, r2=r2, degree=1)
    g_top = fe.Expression(('t1', 't2'), t1=t1, t2=t2, degree=1)
    # g = fe.Expression(('x[0] > 1.0-1e-14 ? rhs : 0', 'x[1] > 1.0-1e-14 ? rhs : 0'), rhs=rhs, degree=1)

    f1 = fe.Constant(f[0])
    f2 = fe.Constant(f[1])
    f = fe.Expression(('f1', 'f2'), f1=f1, f2=f2, degree=1)

    # --------------------
    # Weak form
    # --------------------

    a = fe.inner(sigma(u, lmbda, mu), epsilon(v))*fe.dx
    # L = fe.inner(g, v)*fe.ds # fe.dot(fe.Constant(rhs), v)*fe.dx -
    L = fe.inner(g_right, v)*ds(1) + fe.inner(g_top, v)*ds(2) + fe.inner(f, v)*fe.dx

    # Solve the PDE
    u = fe.Function(V)
    fe.solve(a == L, u, bcs)
    #fe.plot(u)
    if plotSol:
        fe.plot(u, mode="displacement")
        plt.show()
        fe.plot(c)
        plt.show()

    u_array = u.compute_vertex_values(mesh)

    # Reshape the numpy array to a 2D grid
    u_grid = np.reshape(u_array, (2, E.shape[0], E.shape[1]))

    u_torch = torch.from_numpy(u_grid)

    return u_torch


def solve_hook_pde_fenics_NeoHook(E, nu, rhs, f=[0.0, 0.0], plotSol=False):
        # Define the mesh
    mesh = fe.UnitSquareMesh(E.shape[0]-1, E.shape[1]-1)

    # Define the function space
    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    W = fe.FunctionSpace(mesh, 'P', 1)

    # Define the trial and test functions
    u = fe.Function(V)
    v = fe.TestFunction(V)

    # Define the conductivity field as a FEniCS function
    c = fe.Function(W)

    # flatten to get it to the same shape as the mesh
    fl_E = E.flatten()
    # get the mesh to vertex map
    d2v = fe.dof_to_vertex_map(W)
    # sort the values from mesh-like to vertex-like
    sort_fl_E = fl_E[d2v.astype("int64")]

    # Map the nodal values to the Function object
    c.vector().set_local(np.asarray(sort_fl_E))

    # --------------------
    # Material fields
    # --------------------
    mu = c / (2*(1 + nu))
    lmbda = c*nu / ((1 + nu)*(1 - 2*nu))

    # --------------------
    # Boundary conditions (Dirichlet)
    # --------------------
    bc1 = fe.DirichletBC(V.sub(1), fe.Constant(0.0), bottom)
    bc2 = fe.DirichletBC(V.sub(0), fe.Constant(0.0), left)
    bcs = [bc1, bc2]

    # --------------------
    # Boundary conditions (Neumann)
    # --------------------
    # create a mesh function which assigns an unsigned integer (size_t) to each edge
    mf = fe.MeshFunction("size_t", mesh, 1) # 3rd argument is dimension of an edge
    mf.set_all(0) # initialize the function to zero
    # create a SubDomain subclass, specifying the portion of the boundary with x[0] < 1/2
    class Boundary_right(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and fe.near(x[0], 1.0))
    class Boundary_top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and fe.near(x[1], 1.0))
    B_right = Boundary_right() # instantiate it
    B_top = Boundary_top() # instantiate it
    # use this lefthalf object to set values of the mesh function to 1 in the subdomain
    B_right.mark(mf, 1)
    B_top.mark(mf, 2)
    # define a new measure ds based on this mesh function
    ds = fe.Measure('ds', domain=mesh, subdomain_data=mf)


    rhs = fe.Constant(rhs)
    g_right = fe.Expression(('rhs', '0'), rhs=rhs, degree=1)
    g_top = fe.Expression(('0', 'rhs'), rhs=rhs, degree=1)



    f1 = fe.Constant(f[0])
    f2 = fe.Constant(f[1])
    f = fe.Expression(('f1', 'f2'), f1=f1, f2=f2, degree=1)
    
    # --------------------
    # Weak form
    # --------------------

    a = fe.inner(sigma_NeoHook(u, lmbda, mu), epsilon(v))*fe.dx
    # L = fe.inner(g, v)*fe.ds # fe.dot(fe.Constant(rhs), v)*fe.dx -
    L = fe.inner(g_right, v)*ds(1) + fe.inner(g_top, v)*ds(2) + fe.inner(f, v)*fe.dx

    # Solve the PDE
    fe.solve(a-L == 0, u, bcs)
    #fe.plot(u)
    if plotSol:
        fe.plot(u, mode="displacement")
        plt.show()
        fe.plot(c)
        plt.show()

    u_array = u.compute_vertex_values(mesh)

    # Reshape the numpy array to a 2D grid
    u_grid = np.reshape(u_array, (2, E.shape[0], E.shape[1]))

    u_torch = torch.from_numpy(u_grid)

    return u_torch