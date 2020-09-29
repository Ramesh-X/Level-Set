import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread

import lv_set.drlse_algo as drlse
from lv_set.show_fig import show_fig1, show_fig2, draw_all

img = np.array(imread('gourd.bmp', True), dtype='float32')

# parameters
timestep = 1        # time step
mu = 0.2/timestep   # coefficient of the distance regularization term R(phi)
iter_inner = 10
iter_outer = 30
lmda = 5            # coefficient of the weighted length term L(phi)
alfa = -3           # coefficient of the weighted area term A(phi)
epsilon = 1.5       # parameter that specifies the width of the DiracDelta function

sigma = 0.8         # scale parameter in Gaussian kernel
img_smooth = gaussian_filter(img, sigma)    # smooth image by Gaussian convolution
[Iy, Ix] = np.gradient(img_smooth)
f = np.square(Ix) + np.square(Iy)
g = 1 / (1+f)    # edge indicator function.

# initialize LSF as binary step function
c0 = 2
initialLSF = c0 * np.ones(img.shape)
# generate the initial region R0 as two rectangles
initialLSF[25:34, 20:24] = -c0
initialLSF[25:34, 40:49] = -c0
phi = initialLSF.copy()


show_fig1(phi)
show_fig2(phi, img)
print('show fig 2 first time')

potential = 2
if potential == 1:
    potentialFunction = 'single-well'  # use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
elif potential == 2:
    potentialFunction = 'double-well'  # use double-well potential in Eq. (16), which is good for both edge and region based models
else:
    potentialFunction = 'double-well'  # default choice of potential function

# start level set evolution
for n in range(iter_outer):
    phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
    if np.mod(n, 2) == 0:
        print('show fig 2 for %i time' % n)
        draw_all(phi, img)

# refine the zero level contour by further level set evolution with alfa=0
alfa = 0
iter_refine = 10
phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)

finalLSF = phi.copy()
print('show final fig 2')
draw_all(phi, img, 10)
