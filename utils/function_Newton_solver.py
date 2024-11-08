import torch


def Newton_solver(Val, x, step_size, eps):
    """
    Solve nonlinear system F=0 by Newton's method.
    J is the Jacobian of F which is calculated by torch.autograd.functional.jacobian.
    F must be functions of x and has to be traceable with torch.autograd.
    At input, x holds the start value and has to require_grad.
    The iteration continues until ||F|| < eps.
    """
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(">> Start Newton-Raphson solver")
    # calculate starting values + grads
    F_value = Val(x)
    J_value = torch.autograd.functional.jacobian(Val, x)

    # check the residual
    F_norm = torch.linalg.norm(F_value, ord=2)  # l2 norm of vector

    iteration_counter = 0

    print(">> Iteration 0. Error: %g" % F_norm)
    while abs(F_norm) > eps and iteration_counter < 10:
        # solve linearized system to get change of x
        delta = torch.linalg.solve(J_value, -F_value)

        # get new x and decouple from old x
        x = x + delta * step_size
        x = x.clone().detach()
        x.requires_grad_(True)

        # calculate new val + grad
        F_value = Val(x)
        J_value = torch.autograd.functional.jacobian(Val, x)

        # calculate new error
        F_norm = torch.linalg.norm(F_value, ord=2)
        iteration_counter += 1
        print(">> Iteration %i finished. Error: %g" % (iteration_counter, F_norm))

    # Here, either a solution is found, or too many iterations
    assert abs(F_norm) < eps, "Maximum number of iterations where reach. Consider using load steps."

    print(">> End Newton-Raphson solver")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("")

    return x, iteration_counter


def least_square_solver(Val, x, step_size, eps):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(">> Start least_square_solver")
    # calculate starting values
    # residuals -> dim = [1, 2 N_\phi^2]
    F_value = Val(x)

    # check the residual
    F_norm = torch.linalg.norm(F_value, ord=2)  # l2 norm of vector

    iteration_counter = 0

    print(">> Iteration 0. Error: %g" % F_norm)
    while abs(F_norm) > eps and iteration_counter < 10:
        # d residuals / d y -> dim = [2 N_\phi^2, 2 N_y^2]
        J_value = torch.autograd.functional.jacobian(Val, x)

        # solve linearized system to get change of x
        delta = torch.linalg.lstsq(J_value, -F_value).solution

        # get new x and decouple from old x
        x = x + delta * step_size
        x = x.clone().detach()
        x.requires_grad_(True)

        # calculate new val
        F_value = Val(x)

        # calculate new error
        F_norm = torch.linalg.norm(F_value, ord=2)
        iteration_counter += 1
        print(">> Iteration %i finished. Error: %g" % (iteration_counter, F_norm))

    # Here, either a solution is found, or too many iterations
    assert abs(F_norm) < eps, "Maximum number of iterations where reach. Consider using load steps."

    print(">> End least_square_solver solver")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("")

    return x, iteration_counter

def gradient_descent_solver(Val, x, step_size, eps):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(">> Start Gradient Descent solver")
    # initilize iteration counter
    # calculate starting values
    # residuals -> dim = [1, 2 N_\phi^2]
    optim = torch.optim.SGD(params=x, lr=step_size)

    for i in range(100):
        # reset optimizer 
        optim.zero_grad()

        # calculate new val
        F_value = Val(x)

        # check the residual
        F_norm = torch.linalg.norm(F_value, ord=2)  # l2 norm of vector

        if F_norm < eps:
            return x, i

        F_norm.backward()
        optim.step()
    
    # Here, either a solution is found, or too many iterations
    assert abs(F_norm) < eps, "Maximum number of iterations where reach. Consider using load steps."

    print(">> End least_square_solver solver")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("")