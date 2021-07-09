# SOLVING TRANSPORT EQUATION ITERATIVELY

# import packages
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import time
import numba as nb

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
m = np.zeros([nx, nt], dtype=np.float64)
lam = np.zeros([nx, nt], dtype=np.float64)
v = 0*np.ones(nx, dtype=np.float64)
grad_m = np.zeros([nx, nt], dtype=np.float64)  # needed to store gradient of m in update velocity step

# initial condition (template image)
mT = np.sin(x)
m[:, 0] = mT
# final condition (reference image)
mR = np.cos(x)

# beta parameter for regularizing velocity
beta = 0.1


# FUNCTIONS FOR THE PROBLEM
# Gradient function
# @nb.jit(fastmath=True)
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
# @nb.jit(fastmath=True)
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
# @nb.jit(fastmath=True)
def adjoint_eq_sol(vel):
    # INPUT
    # vel is velocity field

    # OUTPUT
    # evaluates and returns lambda from adjoint equation given velocity and final condition:
    # p_t (lambda) = -grad(lambda * vel)

    # SOLVING USING HEUN'S METHOD
    for iii in range(nt):
        dlamvfft = grad(np.multiply(vel, lam[:, nt - 1 - iii]))
        lamtil = lam[:, nt - 1 - iii] - dt * grad(np.multiply(vel, lam[:, nt - 1 - iii]))
        dlamtil = grad(np.multiply(lamtil, vel))

        # solving adjoint equation backward in time
        lam[:, nt - 1 - iii - 1] = lam[:, nt - 1 - iii] - dt / 2 * (dlamvfft + dlamtil)

    return lam


# Laplace operator
# @nb.jit(fastmath=True)
def lap(array):
    # INPUT
    # 1 dimensional array

    # OUTPUT
    # returns inverse lap + id operator applied to array

    uhat = np.fft.fft(array)
    kappa = (2 * np.pi / L) * np.arange(-nx / 2, nx / 2)
    # Re-order fft frequencies
    kappa = np.fft.fftshift(kappa)
    # Laplacian operator in fourier domain
    lap_op = (kappa * 1j)**4
    # Apply one over Laplacian operator to array
    uhat = lap_op*uhat
    # Apply inverse fft to uhat and make uhat real
    uhat = np.real(np.fft.ifft(uhat))

    return uhat


# Laplace + Id operator inverse
# @nb.jit(fastmath=True)
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
    lap_op = (kappa * 1j)**4
    # Apply one over (Laplacian + identity) operator to array
    step1 = 2*(lap_op + np.ones(nx))**-1
    uhat = step1*uhat
    # Apply inverse fft to uhat and make uhat real
    uhat = np.real(np.fft.ifft(uhat))

    return uhat


# Objective Functional
# @nb.jit(fastmath=True)
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

timing = time.time() # time code with and without numba

# SOLVING OPTIMIZATION PROBLEM WITH GRADIENT DESCENT ON VEL

for j in range(1000):

    # show iterate
    # print(j)
    # solve state equation
    m = state_eq_sol(v)
    # store gradient of m
    for iiiiii in range(nt):
        grad_m[:, iiiiii] = grad(m[:, iiiiii])
    # compute obj functional
    J1 = obj_functional(m[:, nt - 1], v)
    # show objective functional val
    # print(J1)
    # final condition of adjoint eq
    lam1 = mR - m[:, nt - 1]
    lam[:, nt - 1] = lam1
    # solve adjoint equation
    lam = adjoint_eq_sol(v)
    # tolerance for the convergence
    tol = 1
    # plot velocity field
    # plt.plot(v)
    # plt.show()
    # update velocity field
    if np.abs(J1) <= tol:
        break
    else:
        # print("updating velocity")
        alpha = 1
        lam_times_grad_m = np.multiply(lam, grad_m)
        time_int_lam_m = np.sum(lam_times_grad_m * dt, 1)
        vnew = v - alpha * (beta * (lap(v) + v) + time_int_lam_m)
        # solve state equation
        m = state_eq_sol(vnew)
        # compute obj functional
        J2 = obj_functional(m[:, nt - 1], vnew)
        if np.abs(J2) < np.abs(J1):
            J1 = J2
            v = vnew
        else:
            k = 0
            while np.abs(J2) >= np.abs(J1) and k < 1000:
                # print("updating alpha")
                alpha = alpha / 2
                vnew = v - alpha * (beta * (lap(v) + np.ones(nx)*v) + time_int_lam_m)
                m = state_eq_sol(vnew)
                J2 = obj_functional(m[:, nt - 1], vnew)
                k = k + 1
            v = vnew


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


# timing = time.time()-timing
# print(timing)

# plotting velocity field and final m

for i in range(12):
    plt.plot(m[:, i], 'b-')
plt.plot(mR, 'r+')
plt.plot(v)
plt.show()

# SOME NOTES ABOUT NEXT MEETING WITH ANDREAS
# for mR "close" to mT, the final velocity tends to be periodic
# and not constant
