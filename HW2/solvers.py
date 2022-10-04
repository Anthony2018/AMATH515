# this file contains collection of solver we learned in the class
from numpy.linalg import norm
from numpy.linalg import solve
import numpy as np

# =============================================================================
# TODO Complete the following optimization algorithms
#	* proximal gradient desecnt
#	* accelerated gradient descent
#	* accelerated proximal gradient descent
# =============================================================================

# Proximal gradient descent
# -----------------------------------------------------------------------------
def optimizeWithPGD(x0, func_f, func_g, grad_f, prox_g, beta_f, tol=1e-6, max_iter=1000):
    """
    Optimize with Proximal Gradient Descent Method
        min_x f(x) + g(x)
    where f is beta smooth and g is proxiable.
    
    input
    -----
    x0 : array_like
        Starting point for the solver
    func_f : function
        Input x and return the function value of f
    func_g : function
        Input x and return the function value of g
    grad_f : function
        Input x and return the gradient of f
    prox_g : function
        Input x and a constant float number and return the prox solution
    beta_f : float
        beta smoothness constant for f
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = x0.copy()
    g = grad_f(x)
    #
    step_size = 1.0/beta_f
    # not recording the initial point since we do not have measure of the optimality
    obj_his = np.zeros(max_iter)
    err_his = np.zeros(max_iter)
    
    # start iteration
    iter_count = 0
    err = tol + 1.0
    while err >= tol:
        #####
        x_new = prox_g(x - step_size * g, step_size)
        #####
        #
        # update information
        obj = func_f(x_new) + func_g(x_new)
        err = norm(x - x_new)/step_size
        #
        np.copyto(x, x_new)
        g = grad_f(x)
        #
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        iter_count += 1
        if iter_count >= max_iter:
            print('Proximal gradient descent reach maximum of iteration')
            return x, obj_his[:iter_count], err_his[:iter_count], 1
    #
    return x, obj_his[:iter_count], err_his[:iter_count], 0

# Accelerated gradient descent
# -----------------------------------------------------------------------------
def optimizeWithAGD(x0, func, grad, beta, tol=1e-6, max_iter=1000):
    """
    Optimize with Accelerated Gradient Descent Method
        min_x f(x)
    where f is beta smooth.
    
    input
    -----
    x0 : array_like
        Starting point for the solver
    func : function
        Input x and return the function value
    grad : function
        Input x and return the gradient
    beta : float
        beta smoothness constant for the function
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = x0.copy()
    y = x0.copy()
    g = grad(y)
    t = 1.0
    #
    step_size = 1.0/beta
    # not recording the initial point since we do not have measure of the optimality
    obj_his = np.zeros(max_iter+1)
    err_his = np.zeros(max_iter+1)
    #
    obj_his[0] = func(x)
    err_his[0] = norm(g)
    
    # start iteration
    iter_count = 0
    err = tol + 1.0
    while err >= tol:
        # proximal gradient descent step
        #####
        # TODO: complete the accelerate gradient step
        x_new = y - step_size * g
        t_new = t + iter_count 
        y_new = x_new + (iter_count-1)/(iter_count+2)*(x_new - x)
        #####
        #
        # update information
        np.copyto(x, x_new)
        np.copyto(y, y_new)
        t = t_new
        g = grad(y)
        #
        obj = func(x_new)
        err = norm(g)
        #
        obj_his[iter_count + 1] = obj
        err_his[iter_count + 1] = err
        #
        # check if exceed maximum number of iteration
        iter_count += 1
        if iter_count >= max_iter:
            print('Proximal gradient descent reach maximum of iteration')
            return x, obj_his[:iter_count+1], err_his[:iter_count+1], 1
    #
    return x, obj_his[:iter_count+1], err_his[:iter_count+1], 0

# Accelerated proximal gradient descent
# -----------------------------------------------------------------------------
def optimizeWithAPGD(x0, func_f, func_g, grad_f, prox_g, beta_f, tol=1e-6, max_iter=1000):
    """
    Optimize with Accelerated Proximal Gradient Descent Method
        min_x f(x) + g(x)
    where f is beta smooth and g is proxiable.
    
    input
    -----
    x0 : array_like
        Starting point for the solver
    func_f : function
        Input x and return the function value of f
    func_g : function
        Input x and return the function value of g
    grad_f : function
        Input x and return the gradient of f
    prox_g : function
        Input x and a constant float number and return the prox solution
    beta_f : float
        beta smoothness constant for f
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = x0.copy()
    y = x0.copy()
    g = grad_f(y)
    t = 1.0
    #
    step_size = 1.0/beta_f
    # not recording the initial point since we do not have measure of the optimality
    obj_his = np.zeros(max_iter)
    err_his = np.zeros(max_iter)
    
    # start iteration
    iter_count = 0
    err = tol + 1.0
    while err >= tol:
        #####
        # TODO: complete the accelerate proximal gradient step
        x_new = prox_g(y - step_size *g, step_size)
        t_new = (1+np.sqrt(1+4 * t))/2
        y_new = x_new + (t-1)/(t_new)*(x_new - x)
        #####
        #
        # update information
        obj = func_f(x_new) + func_g(x_new)
        err = norm(x - x_new)
        #
        np.copyto(x, x_new)
        np.copyto(y, y_new)
        t = t_new
        g = grad_f(y)
        #
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        iter_count += 1
        if iter_count >= max_iter:
            print('Proximal gradient descent reach maximum of iteration')
            return x, obj_his[:iter_count], err_his[:iter_count], 1
    #
    return x, obj_his[:iter_count], err_his[:iter_count], 0

# =============================================================================
# From previous homeworks:
#	* gradient descent
#	* Newton's method
# =============================================================================

# Gradient descent
# -----------------------------------------------------------------------------
def optimizeWithGD(x0, func, grad, beta, tol=1e-6, max_iter=1000):
    """
    Optimize with Gradient Descent
    	min_x f(x)
    where f is beta smooth.

    input
    -----
    x0 : array_like
        Starting point for the solver.
    func : function
        Input x and return the function value.
    grad : function
        Input x and return the gradient.
    beta : float
        beta smoothness constant
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = np.copy(x0)
    g = grad(x)
    step_size = 1.0/beta
    #
    obj = func(x)
    err = norm(g)
    #
    obj_his = np.zeros(max_iter + 1)
    err_his = np.zeros(max_iter + 1)
    #
    obj_his[0] = obj
    err_his[0] = err
    
    # start iterations
    iter_count = 0
    while err >= tol:
        # gradient descent step
        x -= step_size*g
        #
        # update function and gradient
        g = grad(x)
        #
        obj = func(x)
        err = norm(g)
        #
        iter_count += 1
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        if iter_count >= max_iter:
            print('Gradient descent reach maximum number of iteration.')
            return x, obj_his[:iter_count+1], err_his[:iter_count+1], 1
    #
    return x, obj_his[:iter_count+1], err_his[:iter_count+1], 0

# Newton's Method
# -----------------------------------------------------------------------------
def optimizeWithNT(x0, func, grad, hess, tol=1e-6, max_iter=100):
    """
    Optimize with Newton's Method
    
    input
    -----
    x0 : array_like
        Starting point for the solver.
    func : function
        Input x and return the function value.
    grad : function
        Input x and return the gradient.
    hess : function
        Input x and return the Hessian matrix.
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial step
    x = np.copy(x0)
    g = grad(x)
    H = hess(x)
    #
    obj = func(x)
    err = norm(g)
    #
    obj_his = np.zeros(max_iter + 1)
    err_his = np.zeros(max_iter + 1)
    #
    obj_his[0] = obj
    err_his[0] = err
    
    # start iteration
    iter_count = 0
    while err >= tol:
        # Newton's step
        x -= solve(H, g)
        #
        # update function, gradient and Hessian
        g = grad(x)
        H = hess(x)
        #
        obj = func(x)
        err = norm(g)
        #
        iter_count += 1
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        #
        # check if exceed maximum number of iteration
        if iter_count >= max_iter:
            print('Gradient descent reach maximum number of iteration.')
            return x, obj_his[:iter_count+1], err_his[:iter_count+1], 1
    #
    return x, obj_his[:iter_count+1], err_his[:iter_count+1], 0

# if __name__ == '__main__':
#     # create synthetic dataset
#     np.random.seed(123)
#     m = 100  # number of measurements
#     n = 500  # number of variables
#     k = 10  # number of nonzero variables
#     s = 0.05  # measurements noise level
#     #
#     A_cs = np.random.randn(m, n)
#     x_cs = np.zeros(n)
#     x_cs[np.random.choice(range(n), k, replace=False)] = np.random.choice([-1.0, 1.0], k)
#     b_cs = A_cs.dot(x_cs) + s * np.random.randn(m)
#     #
#     lam_cs = 0.1 * norm(A_cs.T.dot(b_cs), np.inf)
#     # apply the proximal gradient descent solver
#     x0_cs_pgd = np.zeros(x_cs.size)
#
#
#     # define the function, prox and the beta constant
#     def func_f_cs(x):
#         # TODO: complete the function
#         return 0.5 * np.linalg.norm(A_cs.dot(x) - b_cs) ** 2
#
#
#     def func_g_cs(x):
#         # TODO: complete the function
#         return lam_cs * np.linalg.norm(x, ord=1)
#
#
#     def grad_f_cs(x):
#         # TODO: complete the gradient of the smooth part
#         return A_cs.T.dot(A_cs.dot(x) - b_cs)
#
#
#     def prox_g_cs(x, t):
#         # TODO: complete the prox of 1 norm
#         x_copy = x.copy()
#         for i in range(len(x)):
#             if x[i] > t:
#                 x_copy[i] = x[i] + t
#             elif x[i] < -t:
#                 x_copy[i] = x[i] + t
#             else:
#                 x_copy[i] = 0
#         return x_copy
#
#
#     ##==GRADED==##
#     # TODO: what is the beta value for the smooth part
#     beta_f_cs = np.max(np.linalg.eig(A_cs.T.dot(A_cs))[0])
#
#     ##==GRADED==##
#     # x_cs_pgd should be (mostly empty) numpy vector of shape (n, )
#     # x_cs_pgd, obj_his_cs_pgd, err_his_cs_pgd, exit_flag_cs_pgd = \
#     #     optimizeWithPGD(x0_cs_pgd, func_f_cs, func_g_cs, grad_f_cs, prox_g_cs, beta_f_cs)
#
#     # apply the proximal gradient descent solver
#     x0_cs_apgd = np.zeros(x_cs.size)
#     ##==GRADED==##
#     # x_cs_apgd should be (mostly empty) numpy vector of shape (n, )
#     x_cs_apgd, obj_his_cs_apgd, err_his_cs_apgd, exit_flag_cs_apgd = \
#         optimizeWithAPGD(x0_cs_apgd, func_f_cs, func_g_cs, grad_f_cs, prox_g_cs, beta_f_cs)