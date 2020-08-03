'''
We will inspect various options for image thresholding provided by OpenCV
Thresholding can be used as a preprocessing step for other operations on images
Such as finding shapes, segmentation...
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_image, plot_images_list

def global_thresholding():
    #important is to work with grayscale images
    image_src = cv2.imread("img/opencv.png", cv2.IMREAD_GRAYSCALE)
    (image_src_h, image_src_w,) = image_src.shape

    threshold = 20

    #every pixel with value >= threshold set to 255, all other to 0
    ret, thresh1 = cv2.threshold(image_src, threshold, 255, cv2.THRESH_BINARY)

    #every pixel with value >= threshold set to 0, all other to 255
    ret, thresh2 = cv2.threshold(image_src, threshold, 255, cv2.THRESH_BINARY_INV)
    
    #every pixel with value >= threshold set to 255, all other stay the same
    ret, thresh3 = cv2.threshold(image_src, threshold, 255, cv2.THRESH_TRUNC)
    
    #every pixel with value >= threshold stays the same, all other are 0
    ret, thresh4 = cv2.threshold(image_src, threshold, 255, cv2.THRESH_TOZERO)

    #every pixel with value >= threshold set to 0, all other stay the same
    ret, thresh5 = cv2.threshold(image_src, threshold, 255, cv2.THRESH_TOZERO_INV)

    plot_images_list([[image_src, thresh1, thresh2], [thresh3, thresh4, thresh5]])

def adaptive_thresholding():
    '''
    Instead of using one set value for image thresholding we can use an algorithm that will adjust threshold value for small regions of image
    This can help us when there are different lighting conditions in an image
    '''

    image_src = cv2.imread("img/adaptive2.png", cv2.IMREAD_GRAYSCALE)
    
    threshold = 127

    ret, thresh1 = cv2.threshold(image_src, threshold, 255, cv2.THRESH_BINARY)

    '''
    For every pixel in the input image look in the neighbouthood of size 11x11 pixels
    Calculate mean value of pixels in the block and subtract constant 2 from that mean
    Set the pixel value to 255 if it's value in original image is greater then subtracted mean
    Othervise set it to 0
    '''
    thresh2 = cv2.adaptiveThreshold(image_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY, 11, 2)
    
    '''
    Similar approach
    For pixels in the block calculate weighted sum, where weights are Gaussian distributed
    Pixel in the center of the block has the largest weight, pixels on the edge of the block have the smallest weight
    Devide the weighted sum by pixel count and use that value as threshold
    '''
    thresh3 = cv2.adaptiveThreshold(image_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 11, 2)

    plot_images_list([[image_src, thresh1], [thresh2, thresh3]]),


