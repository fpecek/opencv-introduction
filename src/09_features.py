import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import plot_image, plot_images_list

WEBCAM_ID = 0

''' webcam template
def webcam(name='test'):
    cap = cv2.VideoCapture(WEBCAM_ID)
    cv2.namedWindow(name)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
'''

image = cv2.imread('img/havana.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
shelf1_original = cv2.imread('img/shelf1.jpg')
shelf2_original = cv2.imread('img/shelf2.jpg')
shelf1 = cv2.imread('img/shelf1.jpg', cv2.IMREAD_GRAYSCALE)
shelf2 = cv2.imread('img/shelf2.jpg', cv2.IMREAD_GRAYSCALE)


def test_harris_corner(image, blockSize=2, ksize=3, k=0.05):
    # harris corner
    # calculate Ix i Iy, calculate Ixx Ixy Iyy, calculate weighted sum, smallest eigenvalue heuristic, NMS
    # https://docs.opencv.org/master/d4/d7d/tutorial_harris_detector.html
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
    image_gray_float = image_gray.astype(np.float32)
    corners = cv2.cornerHarris(image_gray_float, blockSize, ksize, k)
    image_draw = np.copy(image)
    image_draw[corners > 0.01*corners.max()] = [0,0,255]
    return image_draw

def webcam_harris():
    cv2.namedWindow('harris')
    cv2.createTrackbar('blockSize', 'harris', 2, 9, lambda x:None)
    cv2.createTrackbar('ksize', 'harris', 3, 9, lambda x:None)
    cv2.createTrackbar('k', 'harris', 5, 100, lambda x:None)
    cv2.createTrackbar('threshold', 'harris', 10, 100, lambda x:None)

    cap = cv2.VideoCapture(WEBCAM_ID)
    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blockSize = max(1, cv2.getTrackbarPos('blockSize', 'harris'))
        ksize = int((cv2.getTrackbarPos('ksize', 'harris')/2))*2+1
        k = cv2.getTrackbarPos('k', 'harris') / 100
        threshold = cv2.getTrackbarPos('threshold', 'harris') / 100

        corners = cv2.cornerHarris(frame_gray, blockSize, ksize, k)
        frame_draw = np.copy(frame)
        idxs_y, idxs_x = np.where(corners > threshold*corners.max())
        for x,y in zip(idxs_x, idxs_y):
            cv2.circle(frame_draw, (x,y), 3, (0,255,0), 2)
        #frame_draw[corners > threshold*corners.max()] = [0,0,255]
        cv2.imshow('harris', frame_draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def normalize(image):
    image_norm = (image - image.min()) / (image.max() - image.min())
    return (255*image_norm).astype(np.uint8)

def webcam_laplacian():
    cv2.namedWindow('laplacian')
    cv2.createTrackbar('ksize', 'laplacian', 1, 9, lambda x:None)
    cv2.createTrackbar('scale', 'laplacian', 1, 30, lambda x:None)

    cap = cv2.VideoCapture(WEBCAM_ID)
    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ksize = max(1, cv2.getTrackbarPos('ksize', 'laplacian'))
        scale = int((cv2.getTrackbarPos('scale', 'laplacian')/2))*2+1

        laplacian = cv2.Laplacian(frame_gray, cv2.CV_32F, ksize, scale)
        cv2.imshow('laplacian', normalize(laplacian))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_feature(image, feature='SIFT'):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
    if feature == 'SIFT':
        # detector + descriptor
        # scale-invariant response map (scale-space DoG, multi-scale extrema), keypoint filtering (low-contrast, edge), orientation assignment
        # orientation histograms relative to main orientation (16x16 grid, 4x4 blocks with 8 orientation bins), vector of 16x8 values, normalization
        # https://docs.opencv.org/4.3.0/da/df5/tutorial_py_sift_intro.html
        # https://docs.opencv.org/master/d7/d60/classcv_1_1SIFT.html
        # http://weitz.de/sift/
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        kp = sift.detect(image_gray, None)
    elif feature == 'SURF':
        # https://docs.opencv.org/4.3.0/df/dd2/tutorial_py_surf_intro.html
        # https://docs.opencv.org/master/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=False, upright=False)
        surf.setHessianThreshold(5000)
        kp, desc = surf.detectAndCompute(image_gray, None)
    elif feature == 'FAST':
        # detector
        # https://docs.opencv.org/4.3.0/df/d0c/tutorial_py_fast.html
        # https://docs.opencv.org/master/df/d74/classcv_1_1FastFeatureDetector.html
        fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        kp = fast.detect(image_gray, None)
    elif feature == 'BRIEF':
        # https://docs.opencv.org/4.3.0/dc/d7d/tutorial_py_brief.html
        # https://docs.opencv.org/master/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html
        star = cv2.xfeatures2d.StarDetector_create(maxSize=45, responseThreshold=30, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=5)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False)
        kp = star.detect(image_gray, None)
        kp, desc = brief.compute(image_gray, kp)
    elif feature == 'ORB':
        # https://docs.opencv.org/4.3.0/d1/d89/tutorial_py_orb.html
        # https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html
        orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        kp, desc = orb.detectAndCompute(image_gray, None)

    image_draw = np.copy(image)
    image_draw = cv2.drawKeypoints(image, kp, image_draw, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_draw

def webcam_feature(name='SIFT'):
    if name not in ['SIFT', 'SURF', 'FAST', 'BRIEF', 'ORB']:
        print(f'{name} is not a valid feature name')
        return
    cap = cv2.VideoCapture(WEBCAM_ID)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_features = test_feature(frame, name)
        cv2.imshow(name, frame_features)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def webcam_SIFT():
    cv2.namedWindow('sift')

    cv2.createTrackbar('nfeatures', 'sift', 0, 1000, lambda x:None)
    cv2.createTrackbar('nOctaveLayers', 'sift', 3, 9, lambda x:None)
    cv2.createTrackbar('contrastThreshold', 'sift', 4, 100, lambda x:None)
    cv2.createTrackbar('edgeThreshold', 'sift', 10, 100, lambda x:None)
    cv2.createTrackbar('sigma', 'sift', 16, 50, lambda x:None)

    cap = cv2.VideoCapture(WEBCAM_ID)
    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        nfeatures = cv2.getTrackbarPos('nfeatures', 'sift')
        nOctaveLayers = max(1, cv2.getTrackbarPos('nOctaveLayers', 'sift'))
        contrastThreshold = cv2.getTrackbarPos('contrastThreshold', 'sift') / 100
        edgeThreshold = cv2.getTrackbarPos('edgeThreshold', 'sift')
        sigma = max(1, cv2.getTrackbarPos('sigma', 'sift')) / 10

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)
        kp = sift.detect(frame_gray, None)
        frame = cv2.drawKeypoints(frame, kp, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('sift', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def match_images(image1, image2, n=-1):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape)==3 else image1
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape)==3 else image2

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp1, desc1 = sift.detectAndCompute(image1_gray, None)
    kp2, desc2 = sift.detectAndCompute(image2_gray, None)

    #test_bruteforce_matching(image1, kp1, desc1, image2, kp2, desc2, n)
    test_knn_matching(image1, kp1, desc1, image2, kp2, desc2)

def test_bruteforce_matching(image1, kp1, desc1, image2, kp2, desc2, n=-1):
    # https://docs.opencv.org/4.3.0/dc/dc3/tutorial_py_matcher.html
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x:x.distance)
    if n > 0:
        matches = matches[:n]
    print(f'{len(matches)} matches')
    image_draw = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plot_image(image_draw)

def test_knn_matching(image1, kp1, desc1, image2, kp2, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance: # ratio test
            good.append(m)
    print(f'{len(good)} matches')
    image_draw = cv2.drawMatches(image1, kp1, image2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plot_image(image_draw)

def test_panorama(image1, image2):
    # https://docs.opencv.org/4.3.0/d1/de0/tutorial_py_feature_homography.html
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape)==3 else image1
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape)==3 else image2

    #image1_gray = cv2.GaussianBlur(image1_gray, (5,5), 1)
    #image2_gray = cv2.GaussianBlur(image2_gray, (5,5), 1)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp1, desc1 = sift.detectAndCompute(image1_gray, None)
    kp2, desc2 = sift.detectAndCompute(image2_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance: # ratio test
            good.append(m)
    print(len(good))

    image_draw = cv2.drawMatches(image1, kp1, image2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(image_draw[:,:,::-1]); plt.show()

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # https://stackoverflow.com/questions/54664721/stitch-two-images-using-homography-transform-transformed-image-cropped
    result = cv2.warpPerspective(image2, M, (image2.shape[1]+image1.shape[1], image2.shape[0]))
    result[0:image1.shape[0], 0:image1.shape[1]] = image1
    #result = cv2.warpPerspective(image1, M, (image1.shape[1]+image2.shape[1], image1.shape[0]))
    #result[0:image2.shape[0], 0:image2.shape[1]] = image2

    #plot_image(result)
    plt.imshow(result[:,:,::-1]); plt.show()


def webcam_panorama():
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    image1 = None
    cap = cv2.VideoCapture(WEBCAM_ID)
    cv2.namedWindow('panorama')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if image1 is None:
            image1 = frame
            kp1, desc1 = sift.detectAndCompute(frame, None)

        image2 = frame
        kp2, desc2 = sift.detectAndCompute(image2, None)

        try:
            matches = bf.knnMatch(desc1, desc2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance: # ratio test
                    good.append(m)
            result = cv2.drawMatches(image1, kp1, image2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            # M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            # result = cv2.warpPerspective(image2, M, (image2.shape[1]+image1.shape[1], image2.shape[0]))
            # result[0:image1.shape[0], 0:image1.shape[1]] = image1

            cv2.imshow('panorama', result)
        except:
            pass

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('w'):
            image1 = frame
            kp1, desc1 = sift.detectAndCompute(image1, None)

    cap.release()
    cv2.destroyAllWindows()