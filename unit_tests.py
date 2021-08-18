from functions import *
import matplotlib.pyplot as plt


# UNIT TEST: INNER PRODUCT
def unit_test_ip():

    array1 = np.zeros([nx, ny, 2])
    array2 = np.zeros([nx, ny, 2])

    # vectors pointing at (0,1) on [nx, ny] grid
    array1[:, :, 1] = np.ones([nx, ny])
    # vectors pointing at (1,0) on [nx, ny] grid
    array2[:, :, 0] = np.ones([nx, ny])

    # the true inner product
    true_ip = np.zeros([nx, ny])
    # computed inner product
    computed_ip = innerprod(array1, array2)
    # check solution
    print(true_ip - computed_ip)


# UNIT TEST: GRADIENT
def unit_test_grad():

    test_function = np.sin(x)

    # the true gradient
    true_gradient = np.zeros([nx, ny, 2])
    true_gradient[:, :, 0] = np.cos(x)
    true_gradient[:, :, 1] = 0

    # computed gradients
    computed_gradient = grad(test_function)

    # Check solutions for partial_x
    plt.figure()
    plt.title("True Solution (partial_x)")
    plt.imshow(true_gradient[:, :, 0])
    plt.show()
    plt.figure()
    plt.title("Computed Solution (partial_x)")
    plt.imshow(computed_gradient[:, :, 0])
    plt.show()

    # Check solutions for partial_y
    plt.figure()
    plt.title("True Solution (partial_y)")
    plt.imshow(true_gradient[:, :, 1])
    plt.show()
    plt.figure()
    plt.title("Computed Solution (partial_y)")
    plt.imshow(computed_gradient[:, :, 1])
    plt.show()


# UNIT TEST: DIVERGENCE
def unit_test_div():

    test_function = np.zeros([nx, ny, 2])
    test_function[:, :, 0] = -np.cos(x**2)
    test_function[:, :, 1] = np.sin(y**2)

    # the true divergence
    true_divergence = 2*x*np.sin(x**2) - 2*y*np.sin(y**2)

    # computed divergence
    computed_divergence = div(test_function)

    # check solution
    plt.figure()
    plt.title("True Divergence")
    plt.imshow(true_divergence)
    plt.show()
    plt.figure()
    plt.title("Computed Divergence")
    plt.imshow(computed_divergence)
    plt.show()



