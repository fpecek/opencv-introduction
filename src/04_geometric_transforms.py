'''
We will look at the OpenCV functions for image geometric transformations: translation, scaling, rotation
Also, we will look at the matrices behind these transformations
These transformations are part of the affine transformations group
Common property of affine transforamtions is that parallel lines in the source image will be parallel after the transform
https://miro.medium.com/max/848/1*HUrMHLry6d8lxcB7qGjFTA.png
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_image, plot_images_list

identity_transform = np.float32([[1, 0, 0], [0, 1, 0]])

def image_resize():
    image_src = cv2.imread("img/logo_2.jpg")
    (image_src_h, image_src_w, _) = image_src.shape

    #for upscaling INTER_LINEAR and INTER_CUBIC interpolations are preffered
    #for downscaling INTER_AREA gives best quality result
    #INTER_NEAREST is fastes, but gives poorest results

    #resizing to double the size using cubic interpolation
    image_big = cv2.resize(image_src, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    #alternatively, you cn set the second parameter to a desires output image size and omit fx and fy
    image_big_w = image_src_w * 2
    image_big_h = image_src_h * 2
    image_big = cv2.resize(image_src, (image_big_w, image_big_h), interpolation=cv2.INTER_CUBIC)

def image_resize_ar(image_path, width, height):
    '''
    We will resize image to a fixed size of (100, 100) while preserving original AR
    '''

    image_src = cv2.imread(image_path)
    (image_src_h, image_src_w, _) = image_src.shape

    original_ar = image_src_w/image_src_h
    target_ar = width / height

    if original_ar > target_ar:
        #we need to resize image WRT width of the original image
        resize_f = width/image_src_w
        target_height = int(image_src_h*resize_f)
        padding = height - target_height
        padding = [padding//2, padding - padding//2, 0, 0]
    else:
        resize_f = height/image_src_h
        target_width = int(image_src_w*resize_f)
        padding = width - target_width
        padding = [0, 0, padding//2, padding - padding//2]

    interpolation = cv2.INTER_AREA if resize_f < 1 else cv2.INTER_LINEAR
    image_target = cv2.resize(image_src, None, fx=resize_f, fy=resize_f, interpolation=interpolation)
    
    #top, bottom, left, right
    image_target_padded = cv2.copyMakeBorder(image_target, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, value=0)
    plot_image(image_target_padded)


def resize_example_interpolation():
    image = np.linspace(1, 9, 9).reshape((3, 3)).astype(np.uint8)
    
    image_upscale_nn = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    image_upscale_lin = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    image_upscale_cube = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def image_translate(image_path):
    '''
    We will translate image from origin (0, 0) one quarted of height and width in the positive axes direction
    Shape of the resulting image is unchanged
    '''

    image_src = cv2.imread(image_path)
    (image_src_h, image_src_w, _) = image_src.shape

    translate_height = image_src_h/4
    translate_width = image_src_h/4

    translation_transform = identity_transform.copy()
    translation_transform[0, 2] = translate_width
    translation_transform[1, 2] = translate_height

    #transofrm the src image using the transformation matrix T and set the size of output image
    img_translation = cv2.warpAffine(image_src, translation_transform, (image_src_w, image_src_h))
    img_translation2 = cv2.warpAffine(image_src, translation_transform,  (image_src_w*2, image_src_h*2))

    plot_images_list([[image_src], [img_translation], [img_translation2]])


def image_rotate(image_path, rotate_degrees):
    image_src = cv2.imread(image_path)
    (image_src_h, image_src_w, _) = image_src.shape

    #helper function for creation of rotation matrix
    center = (image_src_w/2, image_src_h/2)
    angle = rotate_degrees
    scale = 1.0
    rotation_transform = cv2.getRotationMatrix2D(center, angle, scale)
    image_rotated = cv2.warpAffine(image_src, rotation_transform, (image_src_w, image_src_h))

    plot_image(image_rotated)

def image_rotate_keep(image_path, rotate_degrees):
    '''
    Rotate the image and keep entire content
    https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
    '''

    image_src = cv2.imread(image_path)
    (image_src_h, image_src_w, _) = image_src.shape

    #helper function for creation of rotation matrix
    center = (image_src_w/2, image_src_h/2)
    angle = rotate_degrees
    scale = 1.0
    rotation_transform = cv2.getRotationMatrix2D(center, angle, scale)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_transform[0, 0])
    abs_sin = abs(rotation_transform[0, 1])

    # find the new width and height bounds
    bound_w = int(image_src_h * abs_sin + image_src_w * abs_cos)
    bound_h = int(image_src_h * abs_cos + image_src_w * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_transform[0, 2] += bound_w/2 - image_src_w//2
    rotation_transform[1, 2] += bound_h/2 - image_src_h//2

    # rotate image with the new bounds and translated rotation matrix
    image_rotated = cv2.warpAffine(image_src, rotation_transform, (bound_w, bound_h))

    plot_image(image_rotated)
