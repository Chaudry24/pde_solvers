import numpy as np

# FILE CONTAINS SETTINGS FOR INITIALIZATIONS

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
nt = 4 * nx
# temporal stepsize
dt = 1 / nt
# temporal grid
t = np.arange(0, 1, dt)


# initialize memory
m = np.zeros([nx, ny, nt], dtype=np.float64)  # state variable
lam = np.zeros([nx, ny, nt], dtype=np.float64)  # adjoint variable
v = 0 * np.ones([nx, ny, 2], dtype=np.float64)  # control variable
grad_m = np.zeros([nx, ny, 2, nt], dtype=np.float64)  # needed to store gradient of m in update velocity step
mtill = np.zeros([nx, ny, nt], dtype=np.float64)  # incremental state variable
lamtill = np.zeros([nx, ny, nt], dtype=np.float64)  # incremental adjoint variable
vtill = np.ones([nx, ny, 2], dtype=np.float64)  # incremental control variable
lam_times_grad_m = np.zeros([nx, ny, 2, nt], dtype=np.float64)  # to save lam time grad m

# initial condition (template image)
mT = np.sin(30*x - 0.5) + np.cos(30*y - 0.5)
# final condition (reference image)
mR = np.sin(30*x - 1) + np.cos(30*y - 1)

# beta parameter for regularizing velocity
beta = 1e-2

