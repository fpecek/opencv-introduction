'''
In this chapter we will inspect OpenCV functions for adding images with and without weighting and image interpolation.
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

def adding_images_simple():
    '''
    Adding two images can be performe using numpy alone or with OpenCV functions.
    When working with images we need to be aware of the shape of images participating in addition and of the type of nupy array.
    Images are usually represented as UINT8 data, with values from 0-255 so addition can overflow the pixel value.
    cv.add function will perform the addition and take care of the overflow issue
    Also it will saturate the values of resulting image to the closest value available for the type of original images
    One of the parameters in the add function call can also be a scalar
    '''

    x = np.uint8([250])
    y = np.uint8([10])
    print(f"OpenCV add: {cv2.add(x,y)}, Numpy add: {np.add(x, y)}")

    logo1 = cv2.imread("img/logo_1.jpg")
    logo2 = cv2.imread("img/logo_2.jpg")
    logo_add = cv2.add(logo1, logo2)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title("Logo 1")
    ax2.set_title("Logo 2")
    ax3.set_title("Logo 1+2")
    ax1.imshow(logo1[:,:,::-1])
    ax2.imshow(logo2[:,:,::-1])
    ax3.imshow(logo_add[:,:,::-1])
    plt.show()


def image_blendig():
    #Blening is usefull operation when you want to add an imag to another iage with a degree of transparency
    #g(x)=(1−α)f0(x)+αf1(x)
    
    #alpha = 0.7, beta = 0.3, gamma = 0
    logo1 = cv2.imread("img/logo_1.jpg")
    logo2 = cv2.imread("img/logo_2.jpg")
    logo_add = cv2.addWeighted(logo1, 0.7, logo2, 0.3, 0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title("Logo 1")
    ax2.set_title("Logo 2")
    ax3.set_title("Logo blend")
    ax1.imshow(logo1[:,:,::-1])
    ax2.imshow(logo2[:,:,::-1])
    ax3.imshow(logo_add[:,:,::-1])
    plt.show()

def image_add_with_mask():
    "We want to add the hulk logo to the CA logo, but without the background"

    #we will load two logo images
    logo1 = cv2.imread("img/logo_1.jpg")
    logo2 = cv2.imread("img/logo_2.jpg")

    #convert second logo to grayscale
    logo2_gray = cv2.cvtColor(logo2, cv2.COLOR_BGR2GRAY)
    #set all pixels with value higher than 200 to 255 and rest to 0
    ret, mask = cv2.threshold(logo2_gray, 200, 255, cv2.THRESH_BINARY)
    #create inverse mask
    mask_inv = cv2.bitwise_not(mask)

    logo1_bg = cv2.bitwise_and(logo1, logo1, mask=mask)
    logo2_fg = cv2.bitwise_and(logo2, logo2, mask=mask_inv)

    plt.imshow(logo1_bg[:, :, ::-1]); plt.show()
    plt.imshow(logo2_fg[:, :, ::-1]); plt.show()

    logo3 = cv2.add(logo1_bg, logo2_fg)

    plt.imshow(logo3[:, :, ::-1]); plt.show()
