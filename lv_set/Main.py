from scipy.misc import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage import measure
import lv_set.drlse_algo as drlse
import numpy as np


img = np.array(imread('../gourd.bmp', True), dtype='float32')
# im_t = img[:, :, 1]

# parameters
timestep = 1        # time step
mu = 0.2/timestep   # coefficient of the distance regularization term R(phi)
iter_inner = 4
iter_outer = 25
lmda = 2            # coefficient of the weighted length term L(phi)
alfa = -9           # coefficient of the weighted area term A(phi)
epsilon = 2.0       # parameter that specifies the width of the DiracDelta function

sigma = 0.8         # scale parameter in Gaussian kernel
img_smooth = filters.gaussian_filter(img, sigma)    # smooth image by Gaussian convolution
[Iy, Ix] = np.gradient(img_smooth)
f = np.square(Ix) + np.square(Iy)
g = 1 / (1+f)    # edge indicator function.

# initialize LSF as binary step function
c0 = 2
initialLSF = c0 * np.ones(img.shape)
# generate the initial region R0 as two rectangles
# initialLSF[24:35, 19:25] = -c0
print(initialLSF.shape)
initialLSF[24:35, 20:26] = -c0
phi = initialLSF.copy()

plt.ion()
fig1 = plt.figure(1)

def show_fig1():
    ax1 = fig1.add_subplot(111, projection='3d')
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)

show_fig1()
fig2 = plt.figure(2)


def show_fig2():
    contours = measure.find_contours(phi, 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)


show_fig2()
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
        fig2.clf()
        show_fig2()
        fig1.clf()
        show_fig1()
        plt.pause(0.3)

# refine the zero level contour by further level set evolution with alfa=0
alfa = 0
iter_refine = 10
phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)

finalLSF = phi.copy()
print('show final fig 2')
fig2.clf()
show_fig2()
fig1.clf()
show_fig1()


'''
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')
y, x = finalLSF.shape
x = np.arange(0, x, 1)
y = np.arange(0, y, 1)
X, Y = np.meshgrid(x, y)
ax3.plot_surface(X, Y, -finalLSF, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
ax3.contour(X, Y, finalLSF, 0, colors='g', linewidths=2)
'''

plt.pause(10)
plt.show()

