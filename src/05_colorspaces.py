'''
We will inspect an effect of choosing the right colorspace for
a problem of color segmentation under different lighting conditions
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_images_list

dark_bgr = cv2.imread("img/cube_dark.png")
light_bgr = cv2.imread("img/cube_light.png")

def check_cube_rgb_channels():

    '''
    In the B channel blues are similar to whites
    In the G channel greens, yellows and whites are similar
    In the R channel reds, oranges, and yellows are similar
    Darker image has lower intensities accross all channels when compared to lighner image
    '''

    plot_images_list([light_bgr, dark_bgr])

    light_channels = cv2.split(light_bgr)
    dark_channels = cv2.split(dark_bgr)

    plot_images_list([[light_bgr]+light_channels, [dark_bgr]+dark_channels])

def check_cube_lab_channels():
    '''
    L - Lightness (intensity)
    A - Color component with range Green to Magenta
    B - Color component with range Blue to Yellow
    We have a dedicated channel for intensity of color in this colorspace
    '''

    #convert images from bgr to lab color space

    light_lab = cv2.cvtColor(light_bgr, cv2.COLOR_BGR2LAB)
    dark_lab = cv2.cvtColor(dark_bgr, cv2.COLOR_BGR2LAB)

    light_channels = cv2.split(light_lab)
    dark_channels = cv2.split(dark_lab)

    plot_images_list([[light_bgr]+light_channels, [dark_bgr]+dark_channels])

    '''
    We can see that the main difference between light and dark image is in the Lightness channel
    Other channels have the same intensities for same cube pieces
    '''

def check_cube_hsv_channels():
    '''
    H – Hue ( Dominant Wavelength ).
    S – Saturation ( how vivid is the color)
    V – Value ( Intensity - lightness or darkness of color )
    We have a dedicated channel for intensity of color in this colorspace
    '''

    #convert images from bgr to lab color space

    light_hsv = cv2.cvtColor(light_bgr, cv2.COLOR_BGR2HSV)
    dark_hsv = cv2.cvtColor(dark_bgr, cv2.COLOR_BGR2HSV)

    light_channels = cv2.split(light_hsv)
    dark_channels = cv2.split(dark_hsv)

    plot_images_list([[light_bgr]+light_channels, [dark_bgr]+dark_channels])

    '''
    We can see that the main difference between light and dark image is in the Value channel
    Hue channel contains the information on the dominant wavelenght of the color so we should look to it to segment the images
    What happened to red? Hue is a angle of circle with values from 0-360 and red colors are in ranges [0, 60] and [300,360]
    '''
    