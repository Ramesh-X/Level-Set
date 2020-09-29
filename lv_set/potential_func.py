"""
This python code demonstrates an edge-based active contour model as an application of the
Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:

  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation",
     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.

Author: Ramesh Pramuditha Rathnayake
E-mail: rsoft.ramesh@gmail.com

Released Under MIT License
"""

# use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
DOUBLE_WELL = 'double-well'

# use double-well potential in Eq. (16), which is good for both edge and region based models
SINGLE_WELL = 'single-well'
