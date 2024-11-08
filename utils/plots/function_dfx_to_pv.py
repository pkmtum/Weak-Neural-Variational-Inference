import dolfinx as dfx
import pyvista as pv
import numpy as np


def dfx_to_pv(fenics_fun, name_quantity: str, fun_values=None):
    """
    :param fenics_fun: dfx.fem.Function (with or without values)
    :param name_quantity: string; what is displayed on the color-bar
    :param fun_values: optional; Function values for fenics_fun
    :return: grid for PyVista plotting
    """
    # convert dfx FunctionSpace to PyVista

    # this absolute abomination of code called FEniCSX with PyVista does not support cell-wise constant (i.e. degree 0)
    # plots. So I do a semi-dirty work around, by creating a new FunctionSpace of degree 1 and then setting cell-wise
    # constant values (instead of point wise constant).
    real_fun_space = fenics_fun.function_space
    bs = real_fun_space.dofmap.bs
    flag_deg_0 = (real_fun_space.ufl_element().degree() == 0)
    if flag_deg_0:
        # here I create a new fun space with degree 1
        fun_space = dfx.fem.FunctionSpace(real_fun_space.mesh, (real_fun_space.ufl_element().family(), 1))
    else:
        fun_space = real_fun_space

    # create grid
    u_topology, u_cell_types, u_geometry = dfx.plot.create_vtk_mesh(fun_space)
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

    # Set value for function (optional)
    if fun_values is None:
        fun_values = np.asarray(fenics_fun[:])

    # Make grid ready to plot
    if flag_deg_0:
        u_grid.cell_data[name_quantity] = fun_values.reshape((u_cell_types.shape[0], bs))
    else:
        u_grid.point_data[name_quantity] = fun_values.reshape((u_geometry.shape[0], bs))
    # set name of quantity
    u_grid.set_active_scalars(name_quantity)

    # return
    return u_grid
