import numpy as np
from scipy import signal,ndimage
import cv2
import matplotlib.pyplot as plt

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    
    #compute kernels for gradient detection
    xKernel = (1/2)*np.array([[1. ,0. ,-1.]])
    yKernel = xKernel.T

    #compute gradients of image I
    Ix = signal.convolve2d(img, xKernel, 'same')
    Iy = signal.convolve2d(img, yKernel, 'same')

    #squares of derivatives for later use
    Ix_2 = Ix * Ix
    Iy_2 = Iy * Iy
    Ixy  = Ix * Iy
    
    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    gw_Ixx = cv2.GaussianBlur(Ix_2, ksize = (5, 5), sigmaX = sigma, borderType = cv2.BORDER_REPLICATE)
    gw_Iyy = cv2.GaussianBlur(Iy_2, ksize = (5, 5), sigmaX = sigma, borderType = cv2.BORDER_REPLICATE)
    gw_Ixy = cv2.GaussianBlur(Ixy, ksize = (5, 5), sigmaX = sigma, borderType = cv2.BORDER_REPLICATE)

    # Compute Harris response function
    # TODO: compute the Harris response function C here

    #compute det and trace of M
    det_M = gw_Ixx*gw_Iyy - gw_Ixy**2
    trace_M =  gw_Ixx + gw_Iyy
    
    #response function C
    C = det_M - k * (trace_M**2)
    
    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    
    #find points with a surrounding window that gives large corner response
    #  (C > threshold) 
    strength = C > thresh
    #take the points of local maxima 
    local_maximality = (C == ndimage.maximum_filter(C, size=3))

    #take the points that fulfill the above criteria 
    col,row = np.asarray(strength & local_maximality).nonzero()
    corners = np.array([row, col]).T

    return corners, C

   