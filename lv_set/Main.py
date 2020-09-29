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
from skimage.io import imread

from lv_set.find_lsf import find_lsf
from lv_set.potential_func import *
from lv_set.show_fig import draw_all


def gourd_params():
    img = imread('gourd.bmp', True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[24:35, 19:25] = -c0
    initial_lsf[24:35, 39:50] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 10,
        'iter_outer': 30,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 0.8,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }


def two_cells_params():
    img = imread('twocells.bmp', True)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[9:55, 9:75] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 5,
        'iter_outer': 40,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1.5,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }


params = gourd_params()
# params = two_cells_params()
phi = find_lsf(**params)

print('Show final output')
draw_all(phi, params['img'], 10)
