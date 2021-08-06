# SOLVING TRANSPORT EQUATION ITERATIVELY

# import packages
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import linalg

# no. of spatial points
nx = 64
# Interval Length
L = 2 * np.pi
# spatial step
dx = L / nx
# spatial grid
x = np.arange(dx - 1 / 2 * dx, L, dx)  # (0, 2pi] grid
# x = np.arange(-L/2 + 1 / 2 * dx, L/2, dx) # (-pi, pi] grid

# CFL condition: dt <= C(dx).
# Courant number
C = 1
# time step
dt = C * dx
# no. of time points
nt = int(np.ceil(1 / dt)) + 1
# time domain
t = np.arange(0, 1, dt)

# initialize memory
m = np.zeros([nx, nt], dtype=np.float64)  # state variable
lam = np.zeros([nx, nt], dtype=np.float64)  # adjoint variable
v = 0 * np.ones(nx, dtype=np.float64)  # control variable
grad_m = np.zeros([nx, nt], dtype=np.float64)  # needed to store gradient of m in update velocity step
mtill = np.zeros([nx, nt], dtype=np.float64)  # incremental state variable
lamtill = np.zeros([nx, nt], dtype=np.float64)  # incremental adjoint variable
vtill = 0 * np.ones(nx, dtype=np.float64)  # incremental control variable

# initial condition (template image)
mT = np.sin(x)
m[:, 0] = mT
# final condition (reference image)
mR = np.sin(x - 0.1)

# beta parameter for regularizing velocity
beta = 0.1


# FUNCTIONS FOR THE PROBLEM
# Gradient function
def grad(array):
    # INPUT
    # 1 dimensional spatial array

    # OUTPUT
    # return gradient using FFT

    uhat = np.fft.fft(array)
    kappa = (2 * np.pi / L) * np.arange(-nx / 2, nx / 2)
    # Re-order fft frequencies
    kappa = np.fft.fftshift(kappa)
    # Obtain real part of the function for plotting
    duhat = kappa * uhat * 1j
    # Inverse Fourier Transform
    dufft = np.real(np.fft.ifft(duhat))

    return dufft


# State equation
def state_eq_sol(vel):
    # INPUT
    # vel the velocity field

    # OUTPUT
    # Evaluates state equation: p_t m + grad(m)*v

    # SOLVING USING HEUN'S METHOD
    for ii in range(nt - 1):
        dmfft = grad(m[:, ii])
        mtil = m[:, ii] - dt * np.multiply(dmfft, vel)
        dmffttil = grad(mtil)

        # solving state equation forward in time
        m[:, ii + 1] = m[:, ii] - dt / 2 * np.multiply(vel, dmfft + dmffttil)

    return m


# Adjoint equation
def adjoint_eq_sol(vel):
    # INPUT
    # vel is velocity field

    # OUTPUT
    # evaluates and returns lambda from adjoint equation given velocity and final condition:
    # p_t (lambda) = -grad(lambda * vel)

    # SOLVING USING HEUN'S METHOD
    for iii in range(nt, 1, -1):
        dlamvfft = grad(np.multiply(vel, lam[:, iii - 1]))
        lamtil = lam[:, iii - 1] - dt * grad(np.multiply(vel, lam[:, iii - 1]))
        dlamtil = grad(np.multiply(lamtil, vel))

        # solving adjoint equation backward in time
        lam[:, iii - 2] = lam[:, iii - 1] - dt / 2 * (dlamvfft + dlamtil)

    return lam


# Laplace operator
def lap(array):
    # INPUT
    # 1 dimensional array

    # OUTPUT
    # returns lap operator applied to array

    uhat = np.fft.fft(array)
    kappa = (2 * np.pi / L) * np.arange(-nx / 2, nx / 2)
    # Re-order fft frequencies
    kappa = np.fft.fftshift(kappa)
    # Laplacian operator in fourier domain
    lap_op = (kappa * 1j) ** 4
    # Apply one over Laplacian operator to array
    uhat = lap_op * uhat
    # Apply inverse fft to uhat and make uhat real
    uhat = np.real(np.fft.ifft(uhat))

    return uhat


# Laplace + Id operator inverse
def lap_plus_id_inv(array):
    # INPUT
    # 1 dimensional array

    # OUTPUT
    # returns inverse lap + id operator applied to array

    uhat = np.fft.fft(array)
    kappa = (2 * np.pi / L) * np.arange(-nx / 2, nx / 2)
    # Re-order fft frequencies
    kappa = np.fft.fftshift(kappa)
    # Laplacian operator
    lap_op = (kappa * 1j) ** 4
    # Apply one over (Laplacian + identity) operator to array
    step1 = 2 * (lap_op + np.ones(nx)) ** -1
    uhat = step1 * uhat
    # Apply inverse fft to uhat and make uhat real
    uhat = np.real(np.fft.ifft(uhat))

    return uhat


# Objective Functional
def obj_functional(m1, vel):
    # INPUT
    # m1 the solution from state equation
    # vel the velocity field

    # OUTPUT
    # evaluates and returns the value of the objective functional

    dist_between_mR_and_m1 = 1 / 2 * np.sum((mR - m1) ** 2 * dx)
    lap_of_vel = lap(vel)
    lap_plus_id_times_vel_times_vel = np.multiply(lap_of_vel + vel, vel)
    sum_of_above = np.sum(lap_plus_id_times_vel_times_vel)
    reg_term = beta / 2 * sum_of_above

    return dist_between_mR_and_m1 + reg_term


# Incremental state equation
def inc_state(m, vtill):
    """Evaluates incremental state equation"""
    # INPUT
    # state variable m
    # incremental control variable v

    # OUTPUT
    # Returns mtill after evaluating incremental state

    for ii in range(nt):
        grad_m[:, ii] = grad(m[:, ii])  # save gradient of m
    for ii in range(nt - 1):
        rhs1 = -grad(mtill[:, ii]) * v - grad_m[:, ii] * vtill
        temp_array = mtill[:, ii] + dt * rhs1
        rhs2 = -grad(temp_array) * v - grad_m[:, ii + 1] * vtill
        mtill[:, ii + 1] = mtill[:, ii] + dt / 2 * (rhs1 + rhs2)

    return mtill


# Incremental adjoint equation
def inc_adj(v, lam, vtill):
    """Evaluates incremental adjoint equation"""
    # INPUTS
    # v veloctiy field
    # lam solution to adjoint equation
    # vtill incremental control variable

    lamtill[:, -1] = -mtill[:, -1]  # final condition on lamtill

    for ii in range(nt, 1, -1):
        rhs1 = -grad(lamtill[:, ii - 1] * v + lam[:, ii - 1] * vtill)
        temp_array = lamtill[:, ii - 1] + dt * rhs1
        rhs2 = -grad(temp_array * v + lam[:, ii - 2] * vtill)
        lamtill[:, ii - 2] = lamtill[:, ii - 1] - dt / 2 * (rhs1 + rhs2)

    return lamtill


# Incremental control equation
def inc_control(lam, m, mtill, vtill):
    # INPUTS
    # lam is the adjoint variable
    # m is the state variable
    # mtill is the incremental state variable
    # vtill is the incremental control variable

    # OUTPUTS
    # applies the Hessian to vtill (evaluates incremental control equation)

    grad_mtill = np.zeros([nx, nt], dtype=np.float64)
    for ii in range(nt):
        grad_m[:, ii] = grad(m[:, ii])
    for ii in range(nt):
        grad_mtill[:, ii] = grad(mtill[:, ii])

    return beta * (lap(vtill) + vtill) + np.sum(lam * grad_mtill + lamtill * grad_m, axis=1)


# SOLVING OPTIMIZATION PROBLEM WITH GRADIENT DESCENT ON VEL

# for j in range(1000):
#
#     # show iterate
#     # print(j)
#     # solve state equation
#     m = state_eq_sol(v)
#     # store gradient of m
#     for iiiiii in range(nt):
#         grad_m[:, iiiiii] = grad(m[:, iiiiii])
#     # compute obj functional
#     J1 = obj_functional(m[:, nt - 1], v)
#     # show objective functional val
#     # print(J1)
#     # final condition of adjoint eq
#     lam1 = mR - m[:, nt - 1]
#     lam[:, nt - 1] = lam1
#     # solve adjoint equation
#     lam = adjoint_eq_sol(v)
#     # tolerance for the convergence
#     tol = 1
#     # plot velocity field
#     # plt.plot(v)
#     # plt.show()
#     # update velocity field
#     if np.abs(J1) <= tol:
#         break
#     else:
#         # print("updating velocity")
#         alpha = 1
#         lam_times_grad_m = np.multiply(lam, grad_m)
#         time_int_lam_m = np.sum(lam_times_grad_m * dt, 1)
#         vnew = v - alpha * (beta * (lap(v) + v) + time_int_lam_m)
#         # solve state equation
#         m = state_eq_sol(vnew)
#         # compute obj functional
#         J2 = obj_functional(m[:, nt - 1], vnew)
#         if np.abs(J2) < np.abs(J1):
#             J1 = J2
#             v = vnew
#         else:
#             k = 0
#             while np.abs(J2) >= np.abs(J1) and k < 1000:
#                 # print("updating alpha")
#                 alpha = alpha / 2
#                 vnew = v - alpha * (beta * (lap(v) + np.ones(nx) * v) + time_int_lam_m)
#                 m = state_eq_sol(vnew)
#                 J2 = obj_functional(m[:, nt - 1], vnew)
#                 k = k + 1
#             v = vnew

# SOLVING OPTIMIZATION PROBLEM WITH PICKARD ITERATION ON VEL

# for j in range(1000):
#
#     # show iterate
#     # print(j)
#     # solve state equation
#     m = state_eq_sol(v)
#     # store gradient of m
#     for iiiiii in range(nt):
#         grad_m[:, iiiiii] = grad(m[:, iiiiii])
#     # compute obj functional
#     J1 = obj_functional(m[:, nt - 1], v)
#     # show objective functional val
#     # print(J1)
#     # final condition of adjoint eq
#     lam1 = mR - m[:, nt - 1]
#     lam[:, nt - 1] = lam1
#     # solve adjoint equation
#     lam = adjoint_eq_sol(v)
#     # tolerance for the convergence
#     tol = 1
#     # plot velocity field
#     # plt.plot(v)
#     # plt.show()
#     # update velocity field
#     if np.abs(J1) <= tol:
#         break
#     else:
#         # print("updating velocity")
#         alpha = 1
#         lam_times_grad_m = np.multiply(lam, grad_m)
#         time_int_lam_m = np.sum(lam_times_grad_m * dt, 1)
#         update = (1 / beta * lap_plus_id_inv(time_int_lam_m) - v)
#         vnew = v - alpha * update
#         # solve state equation
#         m = state_eq_sol(vnew)
#         # compute obj functional
#         J2 = obj_functional(m[:, nt - 1], vnew)
#         if np.abs(J2) < np.abs(J1):
#             J1 = J2
#             v = vnew
#         else:
#             k = 0
#             while np.abs(J2) >= np.abs(J1) and k < 1000:
#                 # print("updating alpha")
#                 alpha = alpha / 2
#                 vnew = v - alpha * update
#                 m = state_eq_sol(vnew)
#                 J2 = obj_functional(m[:, nt - 1], vnew)
#                 k = k + 1
#             v = vnew

# SOLVING OPTIMIZATION PROBLEM WITH PCG

# Matrix vector multiplication
def mat_vec(vector):
    return inc_control(lam, m, mtill, vector)


# hessian_operator as linear operator
hessian_operator = linalg.LinearOperator(shape=(nx, nx), matvec=mat_vec)
# hessian_operator = lambda v_tilde: inc_control(lam, m, mtill, v_tilde)
# hessian_operator = linalg.aslinearoperator(hessian_operator)

for ii in range(30):
    # solve state equation
    m = state_eq_sol(v)
    # store gradient of m
    for ii in range(nt):
        grad_m[:, ii] = grad(m[:, ii])
    # compute obj functional
    J1 = obj_functional(m[:, nt - 1], v)
    # final condition of adjoint eq
    lam1 = mR - m[:, -1]
    lam[:, nt - 1] = lam1
    # solve adjoint equation
    lam = adjoint_eq_sol(v)
    # tolerance for the convergence
    tol = 0.001
    if np.abs(J1) <= tol:
        break
    else:
        print("Updating Veloctiy")
        # initial condition for incremental state variable
        mtill[:, 0] = np.zeros(nx, dtype=np.float64)
        # solve incremental state equation
        mtill = inc_state(m, vtill)
        # final condition for lamtill
        lamtill[:, -1] = -mtill[:, -1]
        # solve incremental adjoint equation
        lamtill = inc_adj(v, lam, vtill)
        # time integral lamda time grad m
        lam_times_grad_m = np.multiply(lam, grad_m)
        time_int_lam_m = np.sum(lam_times_grad_m * dt, 1)
        # evaluate the incremental control equation as Hessian operator
        # Hessian * update = -gradient
        gradient = beta * (lap(v) + v) + time_int_lam_m
        update = linalg.cg(hessian_operator, -gradient)[0]
        # update velocity
        alpha = 1
        vnew = v - alpha * update
        m = state_eq_sol(vnew)
        J2 = obj_functional(m[:, -1], vnew)
        if J2 < J1:
            v = vnew
            J1 = J2
        else:
            k = 0
            while J1 < J2 and k < 4:
                alpha = alpha / 3
                vnew = v - alpha * update
                m = state_eq_sol(vnew)
                J2 = obj_functional(m[:, -1], vnew)
                k += 1
            v = vnew


for i in range(12):
    plt.plot(x, m[:, i], 'b-')
plt.plot(x, mR, 'r+')
plt.plot(x, v)
plt.show()

# SOME NOTES ABOUT NEXT MEETING WITH ANDREAS
# for mR "close" to mT, the final velocity tends to be periodic
# and not constant
