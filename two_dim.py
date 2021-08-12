# WE SOLVE TRANSPORT EQUATION IN 2D

# packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg

# TODO: (AM) General comments:
# [ ] add a benchmark for the differential operators and
#       move them to a separate file
# [ ] add benchmarks for the PDE solvers (forward and adjoint solvers);
#       to make sure there are no bugs and to measure the runtime
#       the CPU time seems really bad (even ignoring
#       convergence due to some bugs)
# [ ] add print statement so you can follow the course of the solver
#       and debug it (check if there are issues in the parameters
#       as well as the computed variables_

# import matplotlib
# import seaborn

# spatial points
nx = 16
ny = nx
# interval length
L = 2.0*np.pi
# step size in space
dx = L / nx
dy = dx
# spatial grid
omega = np.arange(dx - 1.0 / 2.0 * dx, L, dx)
x, y = np.meshgrid(omega, omega)
# temporal points
nt = 4*nx
# temporal stepsize
dt = 1.0 / nt
# temporal grid
t = np.arange(0, 1, dt)

print( "nt = ", nt );
print( "ht = ", dt );

# initialize memory (operator for objective functional)
temp = np.zeros([nx, ny, 2])
# save 2 dimensional identity into temp
temp[:, :, 0] = np.eye(nx)
temp[:, :, 1] = np.eye(nx)

# initialize memory
m = np.zeros([nx, ny, nt], dtype=np.float64)  # state variable
lam = np.zeros([nx, ny, nt], dtype=np.float64)  # adjoint variable
v = np.ones([nx, ny, 2], dtype=np.float64)  # control variable
grad_m = np.zeros([nx, ny, 2, nt], dtype=np.float64)  # needed to store gradient of m in update velocity step
lam_times_grad_m = np.zeros([nx, ny, 2, nt], dtype=np.float64)  # needed to store lam \times grad(m)
mtill = np.zeros([nx, ny, nt], dtype=np.float64)  # incremental state variable
lamtill = np.zeros([nx, ny, nt], dtype=np.float64)  # incremental adjoint variable
vtill = 0 * np.ones([nx, ny, 2], dtype=np.float64)  # incremental control variable
lam_times_grad_m = np.zeros([nx, ny, 2, nt], dtype=np.float64) # needed to store lam times grad_m

# initial condition (template image)
mT = np.sin(30*x*y)
m[:, :, 0] = mT

# TODO: (AM) use synthetic velocity field and solve forward problem
# to create synthetic reference image

# final condition (reference image)
mR = np.sin(30*x*y-0.7)

# plot initial and final images
#plt.imshow(mT)
#plt.imshow(mR)
#plt.show()

# beta parameter for regularizing velocity
beta = 0.1


# TODO: (AM) add divergence and gradient to
# tensor calc python file and import it
# write benchmark file to test these operators
# throw flag if results are wrong

# Inner product
def innerprod(array1, array2, axis=2):
    # INPUT
    # array1, array2 [nx, ny, 2] dimensional arrays

    # OUTPUT
    # returns the inner product between two arrays as [nx, ny] array
    prod = array1 * array2
    temp1 = np.sum(prod, axis)
    return temp1


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

    # TODO: (AM) while the initial condition for this equation
    # does not change, you should still set it explicitly
    # for readibility of the code

    # initialize array to say v \cdot grad(m)
    # v_dot_grad_m = np.zeros([nx, ny])
    # SOLVING USING HEUN'S METHOD
    for ii in range(nt - 1):
        dmfft = grad(m[:, :, ii])
        # saving grad(m) \cdot v
        v_dot_grad_m = innerprod(dmfft, vel)
        # continue Heun algorithm
        mtil = m[:, :, ii] - dt * v_dot_grad_m
        dmffttil = grad(mtil)

        # solving state equation forward in time
        # saving grad(mtil) \cdot v
        v_dot_grad_m = innerprod(dmfft + dmffttil, vel)
        # continue solving state equation
        m[:, :, ii + 1] = m[:, :, ii] - dt / 2.0 * v_dot_grad_m

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

    # TODO: (AM) I do not see the setup for the final condition
    # here; where do you set lam(t=1) ? it changes
    # during the course of the iterations

    # initialize array to save lam times vel product
    vel_times_lam = np.zeros([nx, ny, 2])
    # SOLVING USING HEUN'S METHOD
    for iii in range(nt, 1, -1):
        # TODO: (AM) why iii-1? should it not be iii instead; also
        # these variables are local; so, you should be able to use i instead of iii
        vel_times_lam[:, :, 0] = np.multiply(vel[:, :, 0], lam[:, :, iii - 1])
        vel_times_lam[:, :, 1] = np.multiply(vel[:, :, 1], lam[:, :, iii - 1])
        dlamvfft = div(vel_times_lam)
        lamtil = lam[:, :, iii - 1] - \
                 dt * dlamvfft
        vel_times_lam[:, :, 0] = np.multiply(lamtil, vel[:, :, 0])
        vel_times_lam[:, :, 1] = np.multiply(lamtil, vel[:, :, 1])
        dlamtil = div(vel_times_lam)

        # solving adjoint equation backward in time
        lam[:, :, iii - 2] = lam[:, :, iii - 1] - dt / 2.0 * (dlamvfft + dlamtil)

    return lam


# TODO: (AM) add laplacian to another file; add benchamrk
# Laplace operator
def lap(array):
    # INPUT
    # [nx, ny, 2] dimensional array

    # OUTPUT
    # returns the laplacian as [nx, ny, 2] array
    # TODO: (AM) the way you compute the laplacian is wrong;
    # integration by parts results in
    # < grad v , grad v > = < -div grad v, v> = < -lap v, v >
    # the laplacian is - lap (depending on how you define it in the
    # spectral domain; it has to be a positive semi-definite operator)

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


# Objective Functional
def obj_functional(m1, vel):
    # INPUT
    # mRef the final reference image
    # m1 the solution from state equation
    # vel the velocity field

    # OUTPUT
    # evaluates and returns the value of the objective functional

    # TODO: (AM) SUGGESTION: the evaluation of the objective function
    # should include the state solve (needs to be done
    # anyway, and this way it's cleaner (you will not forget
    # to update m1; if you want, you can then have two functions
    # to evaluate the distance and the regularizer
    # which will then take the input m1

    print("evaluating objective functional")

    # need temp array to save Lv \cdot v
    # temp1 = np.zeros([nx, ny])
    # TODO: there should be a weight between lap and identity
    # i.e., op = (-lap + gamma*id)

    lap_id_op = lap(vel) + vel
    temp1 = innerprod(lap_id_op, vel)
    return 1.0 / 2.0 * np.sum(np.sum((mR - m1) ** 2 * dx) * dy) + beta / 2.0 * np.sum(temp1)


# Incremental state equation
def inc_state(m, vtill):
    """Evaluates incremental state equation"""
    # INPUT
    # state variable m
    # incremental control variable v

    # OUTPUT
    # Returns mtill after evaluating incremental state

    # TODO: (AM) General comment: variables that are local (and not global)
    # can have the same names; local variables only exist
    # in the current context (inside loops or functions); so, no there
    # should be no need to use temp1 and temp2, unless
    # there are issues with pointers (you need to be careful about
    # assigments between arrays)

    # TODO: (AM) if you are doing this for speedup reasons
    # then what you can do is add a temp array to the second loop
    # that stores the current gradient (you only compute the gradient once)
    # would require less memory; otherwise, you can also pre-compute it
    # when evaluating the state equation (then you really get a speed-up
    # at the cost of memory)
    print("incremental state solve")
    for ii in range(nt):
        # save gradient of m
        temp1 = grad(m[:, :, ii])
        grad_m[:, :, 0, ii] = temp1[:, :, 0]  # partial_x m
        grad_m[:, :, 1, ii] = temp1[:, :, 1]  # partial_y m

    for ii in range(nt - 1):
        # TODO: NEVER (!!!!) allocate memory in a foor loop
        grad_m_till = np.zeros([nx, ny, 2, nt], dtype=np.float64)

        temp2 = grad(mtill[:, :, ii])
        grad_m_till[:, :, 0, ii] = temp2[:, :, 0]  # partial_x mtil
        grad_m_till[:, :, 1, ii] = temp2[:, :, 1]  # partial_y mtil
        rhs1 = -innerprod(grad_m_till[:, :, :, ii], v) - innerprod(grad_m[:, :, :, ii], vtill)
        temp_array = mtill[:, :, ii] + dt * rhs1
        temp_array_grad = grad(temp_array)
        rhs2 = -innerprod(temp_array_grad, v) - innerprod(grad_m[:, :, :, ii + 1], vtill)

        # TODO: if you are sure it's executed as (dt/2.0)*(rhs1 + rhs2) [ which I think
        # it is) this should be fine; for readibility I would suggest to add brackets
        # or a variable dthalf = 0.5*dt;
        mtill[:, :, ii + 1] = mtill[:, :, ii] + dt / 2.0 * (rhs1 + rhs2)

    return mtill


# Incremental adjoint equation
def inc_adj(v, lam, vtill):
    """Evaluates incremental adjoint equation"""
    # INPUTS
    # v veloctiy field
    # lam solution to adjoint equation
    # vtill incremental control variable

    print("incremental adjoint solve")
    lamtill[:, :, -1] = -mtill[:, :, -1]  # final condition on lamtill
    lamtill_times_v = np.zeros([nx, ny, 2])
    lam_times_vtill = np.zeros([nx, ny, 2])
    temp_array_times_v = np.zeros([nx, ny, 2])

    for ii in range(nt, 1, -1):
        # TODO: (AM) above your are using np.multipy, here you are not;
        # are both of these equivalent / a hadamard product?

        # TODO: (AM) why ii-1? should it not be ii instead; the number of
        # entries should be nt (i.e., the last entry is at index nt)
        lamtill_times_v[:, :, 0] = lamtill[:, :, ii - 1] * v[:, :, 0]  # lamtill times v in x
        lamtill_times_v[:, :, 1] = lamtill[:, :, ii - 1] * v[:, :, 1]  # lamtill times v in y
        lam_times_vtill[:, :, 0] = lam[:, :, ii - 1] * vtill[:, :, 0]  # lam times vtill in x
        lam_times_vtill[:, :, 1] = lam[:, :, ii - 1] * vtill[:, :, 1]  # lam times vtill in y
        rhs1 = -div(lamtill_times_v + lam_times_vtill)
        temp_array = lamtill[:, :, ii - 1] + dt * rhs1
        temp_array_times_v[:, :, 0] = temp_array * v[:, :, 0]  # temp_array * v in x
        temp_array_times_v[:, :, 1] = temp_array * v[:, :, 1]  # temp_array * v in y
        lam_times_vtill[:, :, 0] = lam[:, :, ii - 2] * vtill[:, :, 0]  # lam times vtill in x
        lam_times_vtill[:, :, 1] = lam[:, :, ii - 2] * vtill[:, :, 1]  # lam times vtill in y
        rhs2 = -div(temp_array_times_v + lam_times_vtill)
        lamtill[:, :, ii - 2] = lamtill[:, :, ii - 1] - dt / 2.0 * (rhs1 + rhs2)

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
    print("evaluating incremental control equation")

    # TODO: (AM) incremental state and adjoint solves go here
    # mtilde and lambda change as a function of vtill

    grad_mtill = np.zeros([nx, ny, 2, nt], dtype=np.float64)
    lam_times_grad_m_till = np.zeros([nx, ny, 2, nt])
    lam_till_times_grad_m = np.zeros([nx, ny, 2, nt])
    for ii in range(nt):
        temp1 = grad(m[:, :, ii])
        grad_m[:, :, :, ii] = temp1  # saving gradient m for all time
    for ii in range(nt):
        temp2 = grad(mtill[:, :, ii])
        grad_mtill[:, :, :, ii] = temp2  # saving gradient mtill for all time
    lam_times_grad_m_till[:, :, 0, :] = grad_mtill[:, :, 0, :] * lam
    lam_times_grad_m_till[:, :, 1, :] = grad_mtill[:, :, 1, :] * lam
    lam_till_times_grad_m[:, :, 0, :] = grad_m[:, :, 0, :] * lamtill
    lam_till_times_grad_m[:, :, 1, :] = grad_m[:, :, 1, :] * lamtill

    # TODO: (AM) there should be a parameter that controls the identity, i.e.,
    # (-lap + gamma*id)
    # TODO: you need to discretize a time integral here, i.e., not just
    # simply sum up but scale it with the time step size; depending on
    # which trapezoidal rule you use, you will have to use different weights
    # easiest is the trapezoidal rule, where first and last time step get a weight
    # of 0.5*dt and
    return beta * (lap(vtill) + vtill) + np.sum(lam_times_grad_m_till + lam_till_times_grad_m, axis=3)


def inc_control2(lam, m, mtill, vtill):
    # INPUTS
    # lam is the adjoint variable
    # m is the state variable
    # mtill is the incremental state variable
    # vtill is the incremental control variable

    # OUTPUTS
    # applies the Hessian to vtill (evaluates incremental control equation)
    print("evaluating incremental control equation")

    # TODO: (AM) incremental state and adjoint solves go here
    # mtilde and lambda change as a function of vtill

    grad_mtill = np.zeros([nx, ny, 2, nt], dtype=np.float64)
    lam_times_grad_m_till = np.zeros([nx, ny, 2, nt])
    lam_till_times_grad_m = np.zeros([nx, ny, 2, nt])
    for ii in range(nt):
        temp1 = grad(m[:, :, ii])
        grad_m[:, :, :, ii] = temp1  # saving gradient m for all time
    for ii in range(nt):
        temp2 = grad(mtill[:, :, ii])
        grad_mtill[:, :, :, ii] = temp2  # saving gradient mtill for all time

    # TODO: (AM) again, question would be if the product is the Hadamard product
    # i.e., an elementwise product)
    lam_times_grad_m_till[:, :, 0, :] = grad_mtill[:, :, 0, :] * lam
    lam_times_grad_m_till[:, :, 1, :] = grad_mtill[:, :, 1, :] * lam
    lam_till_times_grad_m[:, :, 0, :] = grad_m[:, :, 0, :] * lamtill
    lam_till_times_grad_m[:, :, 1, :] = grad_m[:, :, 1, :] * lamtill

    # reshape v till from vector to tensor
    vtill = vtill.reshape((nx, ny, 2))

    # TODO: (AM) there should be a parameter that controls the identity, i.e.,
    # (-lap + gamma*id)
    # TODO: you need to discretize a time integral here, i.e., not just
    # simply sum up but scale it with the time step size; depending on
    # which trapezoidal rule you use, you will have to use different weights
    # easiest is the trapezoidal rule, where first and last time step get a weight
    # of 0.5*dt and
    increment_control = beta * (lap(vtill) + vtill) + np.sum(lam_times_grad_m_till + lam_till_times_grad_m, axis=3)
    # increment_control = increment_control.reshape((nx * ny * 2, nx * ny * 2))
    return increment_control






# # SOLVING OPTIMIZATION PROBLEM WITH GRADIENT DESCENT ON VEL
#
# for j in range(15):
#
#     # solve state equation
#     m = state_eq_sol(v)
#     # store gradient of m
#     for iiiiii in range(nt):
#         grad_m[:, :, :, iiiiii] = grad(m[:, :, iiiiii])
#     # compute obj functional
#     J1 = obj_functional(m[:, :, nt - 1], v)
#     # final condition of adjoint eq
#     lam1 = mR - m[:, :, nt - 1]
#     lam[:, :, nt - 1] = lam1
#     # solve adjoint equation
#     lam = adjoint_eq_sol(v)
#     # tolerance for the convergence
#     tol = 0.01
#     if np.abs(J1) <= tol:
#         break
#     else:
#         # print("updating velocity")
#         alpha = 1
#         lam_times_grad_m[:, :, 0, :] = np.multiply(lam, grad_m[:, :, 0, :])
#         lam_times_grad_m[:, :, 1, :] = np.multiply(lam, grad_m[:, :, 1, :])
#         time_int_lam_m = np.sum(lam_times_grad_m * dt)
#         vnew = v - alpha * (beta * (lap(v) + temp * v) + time_int_lam_m + np.random.randn(nx, ny, 2))
#         # solve state equation
#         m = state_eq_sol(vnew)
#         # compute obj functional
#         J2 = obj_functional(m[:, :, nt - 1], vnew)
#         if np.abs(J2) < np.abs(J1):
#             J1 = J2
#             v = vnew
#         else:
#             k = 0
#             while np.abs(J2) >= np.abs(J1) and k < 5:
#                 print("updating alpha")
#                 alpha = alpha / 3
#                 vnew = v - alpha * (beta * (lap(v) + temp * v) + time_int_lam_m + np.random.randn(nx, ny, 2))
#                 m = state_eq_sol(v)
#                 J2 = obj_functional(m[:, :, nt - 1], vnew)
#                 k = k + 1
#             v = vnew


# SOLVING OPTIMIZATION PROBLEM USING PCG

def mat_vec(vector):
    # INPUT
    # vector of size [nx * ny * 2, 1]

    # TODO: you need to solve the incremental state and adjoint
    # equations when evaluating the matvec; they
    # change as a function of "vector"; you can also do this in
    # inc_control (see comments above)
    return inc_control2(lam, m, mtill, vector)

hessian_operator = linalg.LinearOperator(shape=(2*nx**2, 2*nx**2), matvec=mat_vec)


for ii in range(30):
    # solve state equation
    m = state_eq_sol(v)

    # TODO: (AM) you store the gradient here but do
    # also recompute it inside the solves; if you
    # precompute it after solving the state equation, you can skip
    # the computation of the gradient of m in the PDE solves;
    # otherwise you are just wasting CPU time and memory for no
    # reason

    # store gradient of m
    for ii in range(nt):
        temp = grad(m[:, :, ii])
        grad_m[:, :, 0, ii] = temp[:, :, 0]
        grad_m[:, :, 1, ii] = temp[:, :, 1]
    # compute obj functional
    J1 = obj_functional(m[:, :, -1], v)
    # final condition of adjoint eq

    # TODO: (AM) add the computation of the initial condition
    # to the PDE solves for readibility

    lam1 = mR - m[:, :, -1]

    # TODO: (AM) again, are you sure it's nt-1 and not nt?
    lam[:, :, nt - 1] = lam1

    # solve adjoint equation
    lam = adjoint_eq_sol(v)

    # tolerance for the convergence
    tol = 0.001
    # TODO: (AM) the stopping condition should be the relative
    # reduction fo the gradient and not the objective value (there
    # are other options, but from an optimization point of view
    # that would be the best for now): ||g(v_k)|| / ||g(v_0)||, where
    # v_k is the current iterate and v_0 is the initial guess for
    # the optimizer
    if np.abs(J1) <= tol:
        break
    else:
        print("updating velocity")

        # TODO: (AM) the incremental solves need to go into the matvec
        # what is updated in this main loop is only the gradient and the
        # objective

        # initial condition for incremental state variable
        mtill[:, :, 0] = np.zeros(nx, dtype=np.float64)
        # solve incremental state equation
        mtill = inc_state(m, vtill)
        # final condition for lamtill
        lamtill[:, :, -1] = -mtill[:, :, -1]
        # solve incremental adjoint equation
        lamtill = inc_adj(v, lam, vtill)
        # time integral lamda time grad m
        lam_times_grad_m[:, :, 0, :] = lam * grad_m[:, :, 0, :]
        lam_times_grad_m[:, :, 1, :] = lam * grad_m[:, :, 1, :]
        time_int_lam_m = np.sum(lam_times_grad_m * dt, 3)
        # evaluate the incremental control equation as Hessian operator
        # Hessian * update = -gradient
        gradient = (beta * (lap(v) + v) + time_int_lam_m).reshape([-1, 1])

        # TODO: (AM) in general ill-advised, but I would suggest to implement
        #       your own CG (should be quick) and add the following:
        #           - print the residual
        #           - implement a negative curvature test
        #           - check for blow ups of the variables (neg curvature
        #               will also make things unstable; we can subsequently)
        #               return to the linalg implementation, but for now to
        #               debug this seems a better choice

        # TODO: (AM) do the reshape afterward (for readiblity);
        update = linalg.cg(hessian_operator, -gradient)[0].reshape([nx, ny, 2]) # returns a tuple with entry 0 being array of the sol
        # update velocity

        # TODO: add the line search to a function
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
plt.figure()
plt.title("Final Image")
plt.imshow(mR)
plt.figure()
plt.title("Solution of final image")
plt.imshow(m[:, :, -1])
x = 1
# for i in range(10):
#     plt.figure()
#     plt.imshow(m[:, :, 2**i])

