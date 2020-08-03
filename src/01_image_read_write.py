'''
Reading images using OpenCV
OpenCV provides imread function for loading image files into numpy array
It uses standard libraries like libpng or libjpeg that implement format codec for coding and decoding of images
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_images():

    #read the lena.png file in the CWD using a flag to load the image as a color image (default).
    #cv2.IMREAD_COLOR is the default read mode
    image_bgr = cv2.imread("img/lena.png")
    cv2.imshow("Image Color", image_bgr)
    print(f"Image BGR shape: {image_bgr.shape}, image type: {type(image_bgr)}")

    #split the color image by channels
    image_b, image_g, image_r = np.dsplit(image_bgr, image_bgr.shape[-1])
    #image_b, image_g, image_r = cv2.split(image_bgr)
    cv2.imshow("Image Color B", image_b)
    cv2.imshow("Image Color G", image_g)
    cv2.imshow("Image Color R", image_r)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #changing colorspace from BGR to RGB for display in matplotlib
    #simple swap of channels in the 3-d dimension of array
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("RGB")
    ax2.set_title("BGR")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax2.imshow(image_bgr)
    plt.show()

    #changing colorspace by numpy indexing
    #notice the difference in flags
    #much faster, but with side-effects
    image_rgb_np = image_bgr[:,:,::-1]


    #other colorspace conversions
    #BGR to Grayscale: Y = 0.299*R + 0.587*G + 0.114*B
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image Gray", image_gray)
    cv2.waitKey()

    #reading image as grayscale
    image_gray2 = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Image Gray 2", image_gray)
    cv2.waitKey()

    #are these two gray images the same?
    image_gray3 = image_gray2 - image_gray
    cv2.imshow("Image Gray 3", image_gray3)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #write the color image as bitmap
    cv2.imwrite("lena.bmp", image_bgr)
