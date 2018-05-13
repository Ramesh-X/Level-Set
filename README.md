# Level Set Image Segmentation using Python

## Introduction

This python code implements a new level set formulation, called distance regularized level set evolution (DRLSE), proposed by 
**Chunming Li et al's** in the paper ["Distance Regularized Level Set Evolution and its Application to Image Segmentation", 
IEEE Trans. Image Processing, vol. 19 (12), 2010](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5557813 "Link to the original paper")

The main advantages of DRLSE over conventional level set formulations include the following:
1) it completely eliminates the need for reinitialization
2) it allows the use of large time steps to significantly speed up curve evolution, while ensuring numerical accuracy
3) Very easy to implement and computationally more efficient than conventional level set formulations.
<br></br>

This package only implements an edge-based active contour model as one application of DRLSE.
More applications of DRLSE can be found in other published papers in the following website:

http://www.imagecomputing.org/~cmli/

## Execution

You can run the project using: <br>
<code>python -m lv_set.Main</code>
