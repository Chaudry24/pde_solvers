# WE SOLVE TRANSPORT EQUATION IN 2D

# packages
import matplotlib.pyplot as plt

# modules
import scipy.linalg
from second_order_methods import *

# TODO: 1. add unit tests
# TODO: 2. m is not changing

# setting norm gradient_0 and norm gradient to be something stupid to progress loop
norm_gradient_0 = 1
norm_gradient = 1

# Second order methods
for ii in range(20):
    # solve state equation
    m = state_eq_sol(v)
    # store gradient of m
    for i in range(nt):
        temp = grad(m[:, :, i])
        grad_m[:, :, 0, i] = temp[:, :, 0]
        grad_m[:, :, 1, i] = temp[:, :, 1]
    # compute obj functional
    J1 = obj_functional(m[:, :, -1], v)
    # solve adjoint equation
    lam = adjoint_eq_sol(v)
    # tolerance for the convergence
    tol = 1e-2
    if norm_gradient / norm_gradient_0 < tol:
        break
    else:
        print(f"{ii}: Updating Velocity")
        lam_times_grad_m[:, :, 0, :] = lam * grad_m[:, :, 0, :]
        lam_times_grad_m[:, :, 1, :] = lam * grad_m[:, :, 1, :]
        time_int_lam_m = np.sum(lam_times_grad_m * dt, 3)
        # evaluate the incremental control equation as Hessian operator
        # Hessian * update = -gradient
        gradient = (beta * (lap(v) + v) + time_int_lam_m).reshape([-1, 1])
        # gradient of initial guess to check stopping condition
        if ii == 0:
            gradient_0 = gradient
            norm_gradient_0 = scipy.linalg.norm(gradient_0)
        # norm of gradient at current iteration
        norm_gradient = scipy.linalg.norm(gradient)
        # returns a tuple with entry 0 being array of the sol
        update = linalg.cg(hessian_operator, -gradient)[0].reshape([nx, ny, 2])
        # update velocity
        alpha = 1
        vnew = v - alpha * update
        m = state_eq_sol(vnew)
        J2 = obj_functional(m[:, :, -1], vnew)
        if J2 < J1:
            v = vnew
            J1 = J2
        else:
            k = 0
            while J1 < J2 and k < 4:
                alpha = alpha / 2
                vnew = v - alpha * update
                m = state_eq_sol(vnew)
                J2 = obj_functional(m[:, :, -1], vnew)
                k += 1
            v = vnew

plt.figure()
plt.title("Initial Image")
plt.imshow(mT)
plt.show()
plt.figure()
plt.title("Final Image")
plt.imshow(mR)
for i in range(nt):
    plt.show()
    plt.figure()
    plt.title(f"Solution of final image at time {i / (nt - 1)}")
    plt.imshow(m[:, :, i])
plt.show()
