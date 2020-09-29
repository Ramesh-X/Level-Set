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

img = imread('gourd.bmp', True)

# parameters
timestep = 1  # time step
iter_inner = 10
iter_outer = 30
lmda = 5  # coefficient of the weighted length term L(phi)
alfa = -3  # coefficient of the weighted area term A(phi)
epsilon = 1.5  # parameter that specifies the width of the DiracDelta function
sigma = 0.8  # scale parameter in Gaussian kernel

# initialize LSF as binary step function
c0 = 2
initialLSF = c0 * np.ones(img.shape)
# generate the initial region R0 as two rectangles
initialLSF[25:34, 20:24] = -c0
initialLSF[25:34, 40:49] = -c0

phi = find_lsf(img=img, initial_lsf=initialLSF, timestep=timestep, iter_inner=iter_inner, iter_outer=iter_outer,
               lmda=lmda, alfa=alfa, epsilon=epsilon, sigma=sigma, potential_function=DOUBLE_WELL)

print('Show final output')
draw_all(phi, img, 10)
