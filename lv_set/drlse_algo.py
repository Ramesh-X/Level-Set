import numpy as np
import scipy.ndimage.filters as filters


def del2(M):
    dx = 1
    dy = 1
    rows, cols = M.shape
    dx = dx * np.ones((1, cols - 1))
    dy = dy * np.ones((rows - 1, 1))

    mr, mc = M.shape
    D = np.zeros((mr, mc))

    if (mr >= 3):
        ## x direction
        ## left and right boundary
        D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:, 0] * dx[:, 1])
        D[:, mc - 1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc - 1]) \
                       / (dx[:, mc - 3] * dx[:, mc - 2])

        ## interior points
        tmp1 = D[:, 1:mc - 1]
        tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
        tmp3 = np.kron(dx[:, 0:mc - 2] * dx[:, 1:mc - 1], np.ones((mr, 1)))
        D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3

    if (mr >= 3):
        ## y direction
        ## top and bottom boundary
        D[0, :] = D[0, :] + \
                  (M[0, :] - 2 * M[1, :] + M[2, :]) / (dy[0, :] * dy[1, :])

        D[mr - 1, :] = D[mr - 1, :] \
                       + (M[mr - 3, :] - 2 * M[mr - 2, :] + M[mr - 1, :]) \
                         / (dy[mr - 3, :] * dx[:, mr - 2])

        ## interior points
        tmp1 = D[1:mr - 1, :]
        tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr - 2, :])
        tmp3 = np.kron(dy[0:mr - 2, :] * dy[1:mr - 1, :], np.ones((1, mc)))
        D[1:mr - 1, :] = tmp1 + tmp2 / tmp3

    return D / 4


def drlse_edge(phi_0, g, lmda, mu, alfa, epsilon, timestep, iter, potentialFunction):  # Updated Level Set Function
    """

    :param phi_0: level set function to be updated by level set evolution
    :param g: edge indicator function
    :param lmda: weight of the weighted length term
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param timestep: time step
    :param iter: number of iterations
    :param potentialFunction: choice of potential function in distance regularization term.
%              As mentioned in the above paper, two choices are provided: potentialFunction='single-well' or
%              potentialFunction='double-well', which correspond to the potential functions p1 (single-well)
%              and p2 (double-well), respectively.
    """
    phi = phi_0.copy()
    [vy, vx] = np.gradient(g)
    for k in range(iter):
        phi = NeumannBoundCond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        smallNumber = 1e-10
        Nx = phi_x / (s + smallNumber)  # add a small positive number to avoid division by zero
        Ny = phi_y / (s + smallNumber)
        curvature = div(Nx, Ny)
        if potentialFunction == 'single-well':
            distRegTerm = filters.laplace(phi, mode='wrap') - curvature  # compute distance regularization term in equation (13) with the single-well potential p1.
        elif potentialFunction == 'double-well':
            distRegTerm = distReg_p2(phi)  # compute the distance regularization term in eqaution (13) with the double-well potential p2.
        else:
            print('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.')
        diracPhi = Dirac(phi, epsilon)
        areaTerm = diracPhi * g  # balloon/pressure force
        edgeTerm = diracPhi * (vx * Nx + vy * Ny) + diracPhi * g * curvature
        phi = phi + timestep * (mu * distRegTerm + lmda * edgeTerm + alfa * areaTerm)
    return phi


def distReg_p2(phi: np.ndarray) -> np.ndarray:
    """
        compute the distance regularization term with the double-well potential p2 in equation (16)
    """
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)  # compute first order derivative of the double-well potential p2 in equation (16)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))  # compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + filters.laplace(phi, mode='wrap')


def div(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    [junk, nxx] = np.gradient(nx)
    [nyy, junk] = np.gradient(ny)
    return nxx + nyy


def Dirac(x: np.ndarray, sigma: float) -> np.ndarray:
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def NeumannBoundCond(f: np.ndarray) -> np.ndarray:
    """
        Make a function satisfy Neumann boundary condition
    """
    [nrow, ncol] = f.shape
    g = f.copy()
    g[[0, nrow - 1], [0, ncol - 1]] = g[[2, nrow - 3], [2, ncol - 3]]
    g[[0, nrow - 1], 1: ncol - 1] = g[[2, nrow - 3], 1: ncol - 1]
    g[1: nrow - 1, [0, ncol - 1]] = g[1: nrow - 1, [2, ncol - 3]]
    return g
