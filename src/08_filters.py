import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import cv2
from utils import plot_image, plot_images_list

image_lena = cv2.imread('img/lena.jpg')
image = cv2.imread('img/havana.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel_eye = np.eye(3)


def normalize(image):
    image_norm = (image - image.min()) / (image.max() - image.min())
    return (255*image_norm).astype(np.uint8)


def test_convolution_modes(image, kernel):
    '''
    Convolution has different modes of operation which affects output sizes.
    OpenCV uses 'same' mode by default in cv2.filter2D.
    '''
    conv_full = scipy.signal.convolve(image, kernel, mode='full')
    conv_valid = scipy.signal.convolve(image, kernel, mode='valid')
    conv_same = scipy.signal.convolve(image, kernel, mode='same')

    print(f'full shape: {conv_full.shape}')
    print(f'valid shape: {conv_valid.shape}')
    print(f'same shape: {conv_same.shape}')


def test_image_padding(image, border=50):
    '''
    Convolutions with 'same' mode are implemented using image padding (adding values on image borders).
    The chosen padding method affects output results near the image border.
    OpenCV uses cv2.BORDER_REFLECT_101 by default in cv2.filter2D.

    https://docs.opencv.org/master/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    '''
    image_pad_constant = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, 0)
    image_pad_replicate = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REPLICATE)
    image_pad_reflect = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REFLECT)
    image_pad_reflect101 = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REFLECT_101) # default
    image_pad_wrap = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_WRAP)

    plot_images_list([[image, image_pad_constant, image_pad_replicate], [image_pad_reflect, image_pad_reflect101, image_pad_wrap]])


'''
Image filtering = calculating new pixel values from values in a local window around old pixel values.
'''
kernels = {
    # simple
    'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'shift_left': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),

    # smoothing, blurring - simple and common operation, usually used to reduce noise in images
    'average': np.ones((3, 3)) / 9, # box kernel
    'gaussian': np.matmul(cv2.getGaussianKernel(3, 1), cv2.getGaussianKernel(3, 1).T), # gaussian kernel, izgled jednadzbe i svojstva

    # "instagram""
    'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'sharpen2': np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - 1/9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),

    # image gradients
    'prewitt_x': np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    'prewitt_y': np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    'sobel_x': np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    'sobel_y': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    'laplace': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
    'laplace2': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
}

kernel_images_gray = {kernel_name: cv2.filter2D(image_gray, -1, kernel_value) for kernel_name, kernel_value in kernels.items()}

def gradient_magnitude(grad_x, grad_y):
    '''
    Calculate total magnitude of image gradients for every pixel in image, given gradients in x and y direction.
    '''
    return np.sqrt(grad_x.astype(np.float)**2 + grad_y.astype(np.float)**2)

image_DoG = cv2.subtract(cv2.GaussianBlur(image, (7,7), 1), cv2.GaussianBlur(image, (7,7), 5))

def webcam_filter(name='identity'):
    if name not in kernels:
        print(f'{name} is not a valid kernel name')
        return
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filtered = cv2.filter2D(frame, -1, kernels[name])
        cv2.imshow(name, frame_filtered)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_morphology_operations(image):
    ret, image_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    erosion = cv2.erode(image_threshold, kernel, iterations = 1)
    dilation = cv2.dilate(image_threshold, kernel, iterations = 1)
    opening = cv2.morphologyEx(image_threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image_threshold, cv2.MORPH_CLOSE, kernel)

    plot_images_list([[image, image_threshold], [erosion, dilation], [opening, closing]])


# canny edge detection
# https://docs.opencv.org/4.3.0/da/d22/tutorial_py_canny.html
# http://bigwww.epfl.ch/demo/ip/demos/edgeDetector/
#   smooth image
#   gradient direction and magnitude
#   NMS perpendicular to edge
#   2 thresholda za strong/weak/noedge
#   connect components
def webcam_canny():
    cv2.namedWindow('canny')
    cv2.createTrackbar('min', 'canny' , 50, 255, lambda x:None)
    cv2.createTrackbar('max', 'canny' , 200, 255, lambda x:None)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thr_min = cv2.getTrackbarPos('min', 'canny')
        thr_max = cv2.getTrackbarPos('max', 'canny')

        detected_edges = cv2.Canny(frame_gray, thr_min, thr_max)
        mask = detected_edges != 0
        result = frame * (mask[:,:,None].astype(frame.dtype))
        cv2.imshow('canny', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
