"""
This python code demonstrates an edge-based active contour model as an application of the
Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:

  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation",
     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.

Author: Ramesh Pramuditha Rathnayake
E-mail: rsoft.ramesh@gmail.com

Released Under MIT License
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from lv_set.drlse_algo import drlse_edge
from lv_set.potential_func import DOUBLE_WELL, SINGLE_WELL
from lv_set.show_fig import show_fig1, show_fig2, draw_all


def find_lsf(img: np.ndarray, initial_lsf: np.ndarray, timestep=1, iter_inner=10, iter_outer=30, lmda=5,
             alfa=-3, epsilon=1.5, sigma=0.8, potential_function=DOUBLE_WELL):
    """
    :param img: Input image as a grey scale uint8 array (0-255)
    :param initial_lsf: Array as same size as the img that contains the seed points for the LSF.
    :param timestep: Time Step
    :param iter_inner: How many iterations to run drlse before showing the output
    :param iter_outer: How many iterations to run the iter_inner
    :param lmda: coefficient of the weighted length term L(phi)
    :param alfa: coefficient of the weighted area term A(phi)
    :param epsilon: parameter that specifies the width of the DiracDelta function
    :param sigma: scale parameter in Gaussian kernal
    :param potential_function: The potential function to use in drlse algorithm. Should be SINGLE_WELL or DOUBLE_WELL
    """
    if len(img.shape) != 2:
        raise Exception("Input image should be a gray scale one")

    if len(img.shape) != len(initial_lsf.shape):
        raise Exception("Input image and the initial LSF should be in the same shape")

    if np.max(img) <= 1:
        raise Exception("Please make sure the image data is in the range [0, 255]")

    # parameters
    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)

    img = np.array(img, dtype='float32')
    img_smooth = gaussian_filter(img, sigma)  # smooth image by Gaussian convolution
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    g = 1 / (1 + f)  # edge indicator function.

    # initialize LSF as binary step function
    phi = initial_lsf.copy()

    show_fig1(phi)
    show_fig2(phi, img)
    print('show fig 2 first time')

    if potential_function != SINGLE_WELL:
        potential_function = DOUBLE_WELL  # default choice of potential function

    # start level set evolution
    for n in range(iter_outer):
        phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potential_function)
        print('show fig 2 for %i time' % n)
        draw_all(phi, img)

    # refine the zero level contour by further level set evolution with alfa=0
    alfa = 0
    iter_refine = 10
    phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potential_function)
    return phi
