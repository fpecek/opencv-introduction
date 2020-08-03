'''
Histograms are graps that plot the distribution of intensities on the image
When working with histograms, we are always talking about intensity distribution per channel
If image is grayscale, only one histogram should be calculated
For every pixel value count the number of pixels on the image with that value
Plot a graph with pixel values on the X axis and number of pixels on the Y axis

If histogram is uniform, it means that number of pixels is same for every possible value.
Mainly present in gradients
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_image, plot_images_list

def plot_gradient_hist():

    image_src = cv2.imread("img/gradients.png", cv2.IMREAD_GRAYSCALE)

    images = [image_src]
    channels = [0]
    mask = None
    hist_size = [256]
    ranges = [0, 256]
    image_hist = cv2.calcHist(images, channels, mask, hist_size, ranges)

    plt.plot(image_hist); plt.show()

def plot_image_hist_gray():

    image_src = cv2.imread("img/lena.png", cv2.IMREAD_GRAYSCALE)

    images = [image_src]
    channels = [0]
    mask = None
    hist_size = [256]
    ranges = [0, 256]
    image_hist = cv2.calcHist(images, channels, mask, hist_size, ranges)

    plt.plot(image_hist); plt.show()

def compare_histograms():

    image_light = cv2.imread("img/lena.png", cv2.IMREAD_GRAYSCALE)
    image_dark = cv2.imread("img/lena_dark.png", cv2.IMREAD_GRAYSCALE)

    channels = [0]
    mask = None
    hist_size = [256]
    ranges = [0, 256]

    image_light_hist = cv2.calcHist([image_light], channels, mask, hist_size, ranges)
    image_dark_hist = cv2.calcHist([image_dark], channels, mask, hist_size, ranges)

    for image_hist, color in zip([image_light_hist, image_dark_hist], ["b", "r"]):
        plt.plot(image_hist, color=color)
        plt.xlim([0, 256])
    plt.show()

def plot_image_hist_color():
    image_src = cv2.imread("img/lena.png")

    image_channels = cv2.split(image_src)

    channels = [0]
    mask = None
    hist_size = [256]
    ranges = [0, 256]
    for image_channel, color in zip(image_channels, ["b", "g", "r"]):
        image_hist = cv2.calcHist([image_channel], channels, mask, hist_size, ranges)
        plt.plot(image_hist, color=color)
        plt.xlim([0, 256])

    plt.show()

def histogram_equalization():

    image_src = cv2.imread("img/lena_dark.png", cv2.IMREAD_GRAYSCALE)
    image_equ = cv2.equalizeHist(image_src)
    image_side_by_side = np.hstack((image_src, image_equ))

    plt.imshow(image_side_by_side, cmap='gray');plt.show()

    channels = [0]
    mask = None
    hist_size = [256]
    ranges = [0, 256]
    image_src_hist = cv2.calcHist([image_src], channels, mask, hist_size, ranges)
    image_equ_hist = cv2.calcHist([image_equ], channels, mask, hist_size, ranges)

    for image_hist, color in zip([image_src_hist, image_equ_hist], ["b", "r"]):
        plt.plot(image_hist, color=color)
        plt.xlim([0, 256])
    plt.show()

def color_histogram_equalisation_per_channel():

    image_src = cv2.imread("lena.png")

    image_channels = cv2.split(image_src)
    
    channels = [0]
    mask = None
    hist_size = [256]
    ranges = [0, 256]

    image_channels_equ = []
    for image_channel in image_channels:
        image_channel_equ = cv2.equalizeHist(image_channel)
        image_channels_equ.append(image_channel_equ)

    image_equ = cv2.merge(image_channels_equ)

    plot_image(image_equ)

    for image_channel, color in zip(image_channels, ["b", "g", "r"]):
        image_hist = cv2.calcHist([image_channel], channels, mask, hist_size, ranges)
        plt.plot(image_hist, color=color)
        plt.xlim([0, 256])

    plt.show()

def color_histogram_equalisation_per_channel_correct():

    image_src = cv2.imread("img/lena_dark.png")
    image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2YCrCb)

    image_channels = cv2.split(image_src)
    light_channel = image_channels[0]
    
    channels = [0]
    mask = None
    hist_size = [256]
    ranges = [0, 256]

    light_channel_equ = cv2.equalizeHist(light_channel)
    image_channels[0] = light_channel_equ

    image_equ = cv2.merge(image_channels)
    image_equ = cv2.cvtColor(image_equ, cv2.COLOR_YCrCb2BGR)

    plot_image(image_equ)
