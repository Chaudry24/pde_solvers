from initialize import *


# Inner product
def innerprod(array1, array2, axis=2):
    # INPUT
    # array1, array2 [nx, ny, 2] dimensional arrays

    # OUTPUT
    # returns the inner product between two arrays as [nx, ny] array
    point_wise_prod = array1 * array2
    sum_of_prod = np.sum(point_wise_prod, axis)
    return sum_of_prod


# Gradient function
def grad(array):
    # INPUT
    # array of size [nx, ny]

    # OUTPUT
    # returns gradient of array of size [nx, ny, 2] using FFT

    # initialize memory
    dufft = np.zeros([nx, ny, 2], dtype=np.float64)
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
    # returns the divergence of the array as [nx, ny] array

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
    # vel 2d velocity field (array of size [nx, ny, 2])

    # OUTPUT
    # Evaluates state equation: p_t m + grad(m) \cdot v
    # and returns array of size [nx, ny, nt]

    # initial condiiton on state variable
    m[:, :, 0] = mT

    # SOLVING USING HEUN'S METHOD
    for i in range(nt - 1):
        dm = grad(m[:, :, i])
        # saving grad(m) \cdot v
        v_dot_array = innerprod(dm, vel)
        # continue Huen algorithm
        rhs1 = m[:, :, i] - dt * v_dot_array
        dmrhs1 = grad(rhs1)

        # solving state equation forward in time
        v_dot_array = innerprod(dm + dmrhs1, vel)
        # continue solving state equation
        m[:, :, i + 1] = m[:, :, i] - dt / 2 * v_dot_array

    return m


# Adjoint equation
def adjoint_eq_sol(vel):
    # INPUT
    # vel 2d velocity field (array of size [nx, ny, 2])

    # OUTPUT
    # evaluates and returns lambda from
    # adjoint equation given velocity and final condition:
    # p_t (lambda) = -grad(lambda * vel)
    # returns lambda array of size [nx, ny, nt]

    # final condition on adjoint variable
    lam[:, :, -1] = mR - m[:, :, 0]

    # initialize array to save lam times vel product
    temp = np.zeros([nx, ny, 2])
    # SOLVING USING HEUN'S METHOD
    for i in range(nt - 1, 0, -1):
        temp[:, :, 0] = vel[:, :, 0] * lam[:, :, i]
        temp[:, :, 1] = vel[:, :, 1] * lam[:, :, i]
        divergence_of_lam_times_vel = div(temp)
        rhs1 = lam[:, :, i] - dt * divergence_of_lam_times_vel
        temp[:, :, 0] = vel[:, :, 0] * rhs1
        temp[:, :, 1] = vel[:, :, 1] * rhs1
        rhs2 = div(temp)

        # solving adjoint equation backward in time
        lam[:, :, i - 1] = lam[:, :, i] - dt / 2 * (divergence_of_lam_times_vel + rhs2)

    return lam


# Laplace operator
def lap(array):
    # INPUT
    # [nx, ny, 2] dimensional array

    # OUTPUT
    # returns the negative laplacian as [nx, ny, 2] array

    # initialize memory
    lap_array = np.zeros([nx, ny, 2])
    # compute gradient of the array
    grad_x = grad(array[:, :, 0])
    grad_y = grad(array[:, :, 1])
    # compute divergence of the gradient aka laplacian
    temp_array = div(grad_x)  # divergence of first component
    lap_array[:, :, 0] = temp_array  # save divergence of first component
    temp_array = div(grad_y)  # divergence of second component
    lap_array[:, :, 1] = temp_array  # save divergence of second component

    return -lap_array


# Laplacian + ID inverse operator
def lap_plus_id_inv(array):
    # INPUT
    # [nx, ny, 2] dimensional array; array =  (Lap + Id) of something

    # OUTPUT
    # returns (Lap + Id)^{-1} of array

    # laplacian operator
    uhat = np.fft.fft2(array)
    # find u and v direction frequencies in fourier domain
    kappa_u = (2 * np.pi / L) * np.arange(-nx / 2, nx / 2)
    kappa_v = (2 * np.pi / L) * np.arange(-ny / 2, ny / 2)
    # Re-order fft frequencies
    kappa_u = np.fft.fftshift(kappa_u)
    kappa_v = np.fft.fftshift(kappa_v)
    # put frequencies in a grid
    kappa_u, kappa_v = np.meshgrid(kappa_u, kappa_v)
    # laplacian operator (negative)
    lap_op = np.zeros([nx, ny, 2])  # initialize memory
    lap_op[:, :, 0] = np.real((-kappa_u * 1j) ** 4)
    lap_op[:, :, 1] = np.real((-kappa_v * 1j) ** 4)
    # Apply one over (Laplacian + identity) operator to array
    step1 = (lap_op + 1) ** -1
    uhat = step1 * uhat
    # Apply inverse fft to uhat and make uhat real
    uhat = np.real(np.fft.ifft2(uhat))

    return uhat


# Objective Functional
def obj_functional(m1, vel):
    # INPUT
    # m1 the solution from state equation at last time point
    # vel the velocity field

    # OUTPUT
    # evaluates and returns the value of the objective functional

    lap_id_op = lap(vel) + vel
    # temp array to save Lv \cdot v
    temp = innerprod(lap_id_op, vel)
    return 1 / 2 * np.sum((mR - m1) ** 2 * dx * dy) + beta / 2 * np.sum(temp)


# Incremental state equation
def inc_state(state_var, inc_vel):
    # INPUT
    # state variable m
    # incremental velocity field

    # OUTPUT
    # Returns mtill

    # we reshape inc_vel because of conjugate gradient
    inc_vel = inc_vel.reshape([nx, ny, 2])

    # initial condition on mtill
    mtill[:, :, -1] = 0

    for i in range(nt):
        # save gradient of state_var
        temp1 = grad(state_var[:, :, i])
        grad_m[:, :, 0, i] = temp1[:, :, 0]  # partial_x state_var
        grad_m[:, :, 1, i] = temp1[:, :, 1]  # partial_y state_var
    for i in range(nt - 1):
        # save gradient of inc_state_var
        grad_m_till = np.zeros([nx, ny, 2, nt], dtype=np.float64)
        temp2 = grad(mtill[:, :, i])
        grad_m_till[:, :, 0, i] = temp2[:, :, 0]  # partial_x inc_state_var
        grad_m_till[:, :, 1, i] = temp2[:, :, 1]  # partial_y inc_state_var
        rhs1 = -innerprod(grad_m_till[:, :, :, i], v) - innerprod(grad_m[:, :, :, i], inc_vel)
        temp_array = mtill[:, :, i] + dt * rhs1
        temp_array_grad = grad(temp_array)
        rhs2 = -innerprod(temp_array_grad, v) - innerprod(grad_m[:, :, :, i + 1], inc_vel)
        mtill[:, :, i + 1] = mtill[:, :, i] + dt / 2 * (rhs1 + rhs2)

    return mtill


# Incremental adjoint equation
def inc_adj(vel, adjoint_var, inc_vel):
    # INPUT
    # vel is veloctiy field
    # adjoint_var solution to adjoint equation
    # inc_vel incremental control variable

    # OUTPUT
    # returns lamtill

    # reshape inc_vel for pcg
    inc_vel = inc_vel.reshape([nx, ny, 2])

    # final condition on lamtill
    lamtill[:, :, -1] = -mtill[:, :, -1]

    lamtill_times_v = np.zeros([nx, ny, 2])
    lam_times_vtill = np.zeros([nx, ny, 2])
    temp_array_times_v = np.zeros([nx, ny, 2])

    for i in range(nt - 1, 0, -1):

        lamtill_times_v[:, :, 0] = lamtill[:, :, i] * vel[:, :, 0]
        lamtill_times_v[:, :, 1] = lamtill[:, :, i] * vel[:, :, 1]
        lam_times_vtill[:, :, 0] = adjoint_var[:, :, i] * inc_vel[:, :, 0]
        lam_times_vtill[:, :, 1] = adjoint_var[:, :, i] * inc_vel[:, :, 1]
        rhs1 = -div(lamtill_times_v + lam_times_vtill)
        temp_array = lamtill[:, :, i - 1] + dt * rhs1
        temp_array_times_v[:, :, 0] = temp_array * vel[:, :, 0]
        temp_array_times_v[:, :, 1] = temp_array * vel[:, :, 1]
        lam_times_vtill[:, :, 0] = adjoint_var[:, :, i - 1] * inc_vel[:, :, 0]
        lam_times_vtill[:, :, 1] = adjoint_var[:, :, i - 1] * inc_vel[:, :, 1]
        rhs2 = -div(temp_array_times_v + lam_times_vtill)
        lamtill[:, :, i - 1] = lamtill[:, :, i] + dt / 2 * (rhs1 + rhs2)

    return lamtill
