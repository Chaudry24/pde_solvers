# WE SOLVE TRANSPORT EQUATION IN 2D

# packages
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
# import seaborn

# spatial points
nx = 4
ny = nx
# interval length
L = 2 * np.pi
# step size in space
dx = L / nx
dy = dx
# spatial grid
omega = np.arange(dx - 1 / 2 * dx, L, dx)
x, y = np.meshgrid(omega, omega)
# temporal points
nt = 1024
# temporal stepsize
dt = 1 / nt
# temporal grid
t = np.arange(0, 1, dt)

# initialize memory (operator for objective functional)
temp = np.zeros([nx, ny, 2])
# save 2 dimensional identity into temp
temp[:, :, 0] = np.eye(nx)
temp[:, :, 1] = np.eye(nx)

# initialize memory
m = np.zeros([nx, ny, nt], dtype=np.float64)
lam = np.zeros([nx, ny, nt], dtype=np.float64)
v = np.ones([nx, ny, 2], dtype=np.float64)
grad_m = np.zeros([nx, ny, 2, nt], dtype=np.float64)  # needed to store gradient of m in update velocity step
lam_times_grad_m = np.zeros([nx, ny, 2, nt], dtype=np.float64)  # needed to store lam \times grad(m)

# initial condition (template image)
mT = np.sin(30*x*y)
m[:, :, 0] = mT
# final condition (reference image)
mR = np.sin(30*x*y-1)

# plot initial and final images
# plt.imshow(mT)
# plt.imshow(mR)

# beta parameter for regularizing velocity
beta = 0.1


# Gradient function
def grad(array):
    # INPUT
    # 2 dimensional spatial array of size [nx, ny]

    # OUTPUT
    # return gradient array of size [nx, ny, 2] using FFT

    # initialize memory
    dufft = np.zeros([nx, ny, 2])
    # take 2d fft of the arrray
    uhat = np.fft.fft2(array)
    # find u and v direction frequencies in fourier domain
    kappa_u = (2 * np.pi / L) * np.arange(-nx / 2, nx / 2)
    kappa_v = (2 * np.pi / L) * np.arange(-ny / 2, ny / 2)
    # Re-order fft frequencies
    kappa_u = np.fft.fftshift(kappa_u)
    kappa_v = np.fft.fftshift(kappa_v)
    # put frequencies in a grid
    kappa_u, kappa_v = np.meshgrid(kappa_u, kappa_v)
    # Find partial derivatives
    duhat_x = kappa_u * uhat * 1j
    duhat_y = kappa_v * uhat * 1j
    # Take inverse FFT and make it real
    duhat_x = np.real(np.fft.ifft2(duhat_x))
    duhat_y = np.real(np.fft.ifft2(duhat_y))
    # Save partial derivatives in array
    dufft[:, :, 0] = duhat_x
    dufft[:, :, 1] = duhat_y

    return dufft


# Divergence
def div(array):
    # INPUT
    # array of size [nx, ny, 2]

    # OUTPUT
    # returns the divergence of the array
    # as a [nx, ny] array

    # find u and v direction frequencies in fourier domain
    kappa_u = (2 * np.pi / L) * np.arange(-nx / 2, nx / 2)
    kappa_v = (2 * np.pi / L) * np.arange(-ny / 2, ny / 2)
    # Re-order fft frequencies
    kappa_u = np.fft.fftshift(kappa_u)
    kappa_v = np.fft.fftshift(kappa_v)
    # put frequencies in a grid
    kappa_u, kappa_v = np.meshgrid(kappa_u, kappa_v)

    # COMPUTE PARTIAL DERIVATIVES

    # FINDING PARTIAL_X
    # take 2d fft of the first coordinate
    uhat = np.fft.fft2(array[:, :, 0])
    # Find partial derivative
    duhat_x = kappa_u * uhat * 1j
    # Take inverse FFT and make it real
    duhat_x = np.real(np.fft.ifft2(duhat_x))

    # FINDING PARTIAL_Y
    # take 2d fft of the second coordinate
    uhat = np.fft.fft2(array[:, :, 1])
    # Find partial derivative
    duhat_y = kappa_v * uhat * 1j
    # Take inverse FFT and make it real
    duhat_y = np.real(np.fft.ifft2(duhat_y))

    # sum partial derivatives to get divergence
    divergence = duhat_x + duhat_y

    return divergence


# State equation
def state_eq_sol(vel):
    # INPUT
    # vel 2d velocity field (array of size [nx, ny])

    # OUTPUT
    # Evaluates state equation: p_t m + grad(m)*v
    # and returns array of size [nx, ny, nt]

    # initialize array to say v \cdot grad(m)
    # v_dot_grad_m = np.zeros([nx, ny])
    # SOLVING USING HEUN'S METHOD
    for ii in range(nt - 1):
        dmfft = grad(m[:, :, ii])
        # saving grad(m) \cdot v
        v_dot_grad_m = np.tensordot(dmfft[:, :, 0], vel[:, :, 0], 1)
        v_dot_grad_m = np.tensordot(dmfft[:, :, 1], vel[:, :, 1], 1) + v_dot_grad_m
        # continue Huen algorithm
        mtil = m[:, :, ii] - dt * v_dot_grad_m
        dmffttil = grad(mtil)

        # solving state equation forward in time
        # saving grad(mtil) \cdot v
        v_dot_grad_m = np.tensordot(dmfft[:, :, 0] + dmffttil[:, :, 0], vel[:, :, 0], 1)
        v_dot_grad_m = np.tensordot(dmfft[:, :, 1] + dmffttil[:, :, 1], vel[:, :, 1], 1) + v_dot_grad_m
        # continue solving state equation
        m[:, :, ii + 1] = m[:, :, ii] - dt / 2 * v_dot_grad_m

    return m


# Adjoint equation
def adjoint_eq_sol(vel):
    # INPUT
    # vel 2d velocity field (array of size [nx, ny])

    # OUTPUT
    # evaluates and returns lambda from
    # adjoint equation given velocity and final condition:
    # p_t (lambda) = -grad(lambda * vel)
    # returns lambda array of size [nx, ny, nt]

    # initialize array to save lam times vel product
    vel_times_lam = np.zeros([nx, ny, 2])
    # SOLVING USING HEUN'S METHOD
    for iii in range(nt):
        vel_times_lam[:, :, 0] = np.multiply(vel[:, :, 0], lam[:, :, nt - 1 - iii])
        vel_times_lam[:, :, 1] = np.multiply(vel[:, :, 1], lam[:, :, nt - 1 - iii])
        dlamvfft = div(vel_times_lam)
        lamtil = lam[:, :, nt - 1 - iii] - \
                 dt * dlamvfft
        vel_times_lam[:, :, 0] = np.multiply(lamtil, vel[:, :, 0])
        vel_times_lam[:, :, 1] = np.multiply(lamtil, vel[:, :, 1])
        dlamtil = div(vel_times_lam)

        # solving adjoint equation backward in time
        lam[:, :, nt - 1 - iii - 1] = lam[:, :, nt - 1 - iii] - dt / 2 * (dlamvfft + dlamtil)

    return lam


# Laplace operator
def lap(array):
    # INPUT
    # [nx, ny, 2] dimensional array

    # OUTPUT
    # returns the laplacian as [nx, ny, 2] array

    # initialize memory
    lap_array = np.zeros([nx, ny, 2])
    # compute gradient of the array
    grad_x = grad(array[:, :, 0])
    grad_y = grad(array[:, :, 1])
    # compute divergence of the gradient aka laplacian
    temp_array1 = div(grad_x)  # divergence of first component
    # temp_array2 = div(grad_x[:, :, 1])
    lap_array[:, :, 0] = temp_array1
    temp_array1 = div(grad_y)  # divergence of second component
    # temp_array2 = div(grad_y[:, :, 1])
    lap_array[:, :, 1] = temp_array1

    return lap_array


# Inner product
def innerprod(array1, array2):
    # INPUT
    # array1, array2 [nx, ny, 2] dimensional arrays

    # OUTPUT
    # returns the inner product between two arrays as [nx, ny] array
    return np.tensordot(array1, array2)


# Objective Functional
def obj_functional(m1, vel):
    # INPUT
    # mRef the final reference image
    # m1 the solution from state equation
    # vel the velocity field

    # OUTPUT
    # evaluates and returns the value of the objective functional

    # need temp array to save Lv \cdot v
    # temp1 = np.zeros([nx, ny])
    lap_id_op = lap(vel) + temp * vel
    temp1 = np.tensordot(lap_id_op[:, :, 0], vel[:, :, 0], 1)
    temp1 = np.tensordot(lap_id_op[:, :, 1], vel[:, :, 1], 1) + temp1
    return 1 / 2 * np.sum(np.sum((mR - m1) ** 2 * dx) * dy) + beta / 2 * np.sum(temp1)


# SOLVING OPTIMIZATION PROBLEM WITH GRADIENT DESCENT ON VEL

for j in range(1000):

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
    tol = 1
    if np.abs(J1) <= tol:
        break
    else:
        # print("updating velocity")
        alpha = 1
        lam_times_grad_m[:, :, 0, :] = np.multiply(lam, grad_m[:, :, 0, :])
        lam_times_grad_m[:, :, 1, :] = np.multiply(lam, grad_m[:, :, 1, :])
        time_int_lam_m = np.sum(lam_times_grad_m * dt)
        vnew = v - alpha * (beta * (lap(v) + temp * v) + time_int_lam_m + np.random.randn(nx, ny, 2))
        # solve state equation
        m = state_eq_sol(vnew)
        # compute obj functional
        J2 = obj_functional(m[:, :, nt - 1], vnew)
        if np.abs(J2) < np.abs(J1):
            J1 = J2
            v = vnew
        else:
            k = 0
            while np.abs(J2) >= np.abs(J1) and k < 80:
                # print("updating alpha")
                alpha = alpha / 2
                vnew = v - alpha * (beta * (lap(v) + temp * v) + time_int_lam_m + np.random.randn(nx, ny, 2))
                m = state_eq_sol(v)
                J2 = obj_functional(m[:, :, nt - 1], vnew)
                k = k + 1
            v = vnew

# plt.imshow(mT)
# plt.show()
# plt.imshow(m[:, :, 0])
# plt.show()
plt.imshow(mR)
# plt.show()
for i in range(10):
    plt.imshow(m[:, :, 2**i])
plt.show()

