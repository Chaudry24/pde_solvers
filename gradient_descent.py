from functions import *

# SOLVING OPTIMIZATION PROBLEM WITH GRADIENT DESCENT ON VEL

for j in range(15):

    # solve state equation
    m = state_eq_sol(v)
    # store gradient of m
    for iiiiii in range(nt):
        grad_m[:, :, :, iiiiii] = grad(m[:, :, iiiiii])
    # compute obj functional
    J1 = obj_functional(m[:, :, nt - 1], v)
    # final condition of adjoint eq
    lam1 = mR - m[:, :, nt - 1]
    lam[:, :, nt - 1] = lam1
    # solve adjoint equation
    lam = adjoint_eq_sol(v)
    # tolerance for the convergence
    tol = 0.01
    if np.abs(J1) <= tol:
        break
    else:
        # print("updating velocity")
        alpha = 1
        lam_times_grad_m[:, :, 0, :] = np.multiply(lam, grad_m[:, :, 0, :])
        lam_times_grad_m[:, :, 1, :] = np.multiply(lam, grad_m[:, :, 1, :])
        time_int_lam_m = np.sum(lam_times_grad_m * dt)
        update = (beta * (lap(v) + v) + time_int_lam_m)
        vnew = v - alpha * update
        # solve state equation
        m = state_eq_sol(vnew)
        # compute obj functional
        J2 = obj_functional(m[:, :, nt - 1], vnew)
        if np.abs(J2) < np.abs(J1):
            J1 = J2
            v = vnew
        else:
            k = 0
            while np.abs(J2) >= np.abs(J1) and k < 5:
                print("updating alpha")
                alpha = alpha / 2
                vnew = v - alpha * update
                m = state_eq_sol(v)
                J2 = obj_functional(m[:, :, nt - 1], vnew)
                k = k + 1
            v = vnew
