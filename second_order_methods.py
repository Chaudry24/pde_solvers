from functions import *
from scipy.sparse import linalg


# Incremental control equation for BETA * (-LAP + ID)*VTILL _ \INT_0^1 (GRAD(M)*LAMDATILL + GRAD(MTILL)*LAMBDA)DT
def inc_control(adjoint_var, state_var, inc_vel, preconditioner):
    # INPUTS
    # adjoint_var is the adjoint variable
    # state_var is the state variable
    # inc_vel is the incremental control variable
    # preconditioner flag determines if preconditioned hessian returned or not

    # OUTPUTS
    # applies the Hessian to vtill (evaluates incremental control equation)

    # solve incremental state equation
    inc_state_var = inc_state(state_var, inc_vel)
    # solve incremental adjoint equation
    inc_adjoint_var = inc_adj(v, adjoint_var, inc_vel)
    # initialize memory to save some things
    grad_mtill = np.zeros([nx, ny, 2, nt], dtype=np.float64)
    lam_times_grad_m_till = np.zeros([nx, ny, 2, nt])
    lam_till_times_grad_m = np.zeros([nx, ny, 2, nt])
    for ii in range(nt):
        temp1 = grad(state_var[:, :, ii])
        grad_m[:, :, :, ii] = temp1  # saving gradient m for all time
    for ii in range(nt):
        temp2 = grad(inc_state_var[:, :, ii])
        grad_mtill[:, :, :, ii] = temp2  # saving gradient inc_state_var for all time
    lam_times_grad_m_till[:, :, 0, :] = grad_mtill[:, :, 0, :] * adjoint_var
    lam_times_grad_m_till[:, :, 1, :] = grad_mtill[:, :, 1, :] * adjoint_var
    lam_till_times_grad_m[:, :, 0, :] = grad_m[:, :, 0, :] * inc_adjoint_var
    lam_till_times_grad_m[:, :, 1, :] = grad_m[:, :, 1, :] * inc_adjoint_var

    # reshape inc_vel from vector to tensor
    inc_vel = inc_vel.reshape((nx, ny, 2))

    if not preconditioner:
        increment_control = beta * (lap(inc_vel) + inc_vel) + np.sum((lam_times_grad_m_till +
                                                                      lam_till_times_grad_m) * dt, axis=3)
        return increment_control
    else:
        increment_control_preconditioned = -1 / beta * lap_plus_id_inv(np.sum((lam_times_grad_m_till +
                                                                               lam_till_times_grad_m) * dt, axis=3))
        return increment_control_preconditioned


def mat_vec(vector, preconditioner=True):
    # INPUT
    # vector of size [nx * ny * 2, 1]

    return inc_control(lam, m, vector, preconditioner)

hessian_operator = linalg.LinearOperator(shape=(2*nx**2, 2*nx**2), matvec=mat_vec,
                                         dtype=np.float64)
