import torch
from utils.Integrator.function_integrator_2D import trapzInt2D

def inverseBasisStiffness(shape_func_target):
    """
    :return: The inverse of the stiffness matrix of the basis functions. int_{s} phi_j*phi_k ds
    """
    num_shape_func_target = shape_func_target.size(0)
    a = torch.zeros(num_shape_func_target, num_shape_func_target)
    for i in range(num_shape_func_target):
        a[i, :] = trapzInt2D(torch.einsum('ij, kij->kij', shape_func_target[i], shape_func_target[:]))
    print("For a^(-1)@B matrix for the basis functions, cond(a) is: ", torch.linalg.cond(a))
    aInv = torch.inverse(a)
    return aInv

def fixedMatrixB(shape_func_target, shape_func_source):
    """
    :return: The inverse of the stiffness matrix of the basis functions. int_{s} phi_j*phi_k ds
    """
    num_shape_func_target = shape_func_target.size(0)
    num_shape_func_source = shape_func_source.size(0)
    B = torch.zeros(num_shape_func_target, num_shape_func_source)
    for i in range(num_shape_func_target):
            B[i, :] = trapzInt2D(torch.einsum('ij, kij->kij', shape_func_target[i], shape_func_source[:]))
    return B

def a_inv_B(shape_func_target, shape_func_source):
    aInv = inverseBasisStiffness(shape_func_target)
    B = fixedMatrixB(shape_func_target, shape_func_source)
    aInvB = torch.einsum('ij,jk->ik', aInv, B)
    return aInvB
