'''
OpenCV provides two clases for working with video: VideoCapture and VideoWriter
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

#unique interface for diverse number of video sources
usb_cam_address = "/dev/video0" #or just 0
file_address = "img/movie.mp4"
stream_url_address = "rtsp://admin:CAMpass001@172.16.50.201:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"


def simple_capture(address):
    """
    capture frames from a video device and display them
    """

    capture = cv2.VideoCapture(address)

    if not capture.isOpened():
        print(f"Cannot opet camera '{address}'")
        return

    while True:
        #capture next frame from the capture device
        ret, frame = capture.read()
        if not ret:
            break

        cv2.imshow("frame", frame)

        #enable stepping out from the loop by pressing the q key
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

def respect_the_fps_and_loop(file_address):
    '''
    We will play the video file in it's specified FPS and start from begging after last frame displayed
    '''

    capture = cv2.VideoCapture(file_address)

    if not capture.isOpened():
        print(f"Cannot opet camera '{file_address}'")
        return

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    inter_frame_ms = 1000/frame_rate
    num_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_index = -1
    timestamp = None

    while True:
        if frame_index == num_frames - 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = -1

        # capture next frame from the capture device
        ret, frame = capture.read()
        if not ret:
            break

        frame_index+=1

        if timestamp:
            elapsed_ms = time.time() - timestamp
            sleep_ms = inter_frame_ms - elapsed_ms
            if sleep_ms > 0:
                time.sleep(sleep_ms/1000)
        cv2.imshow("frame", frame)
        timestamp = time.time()

        # enable stepping out from the loop by pressing the q key
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


def write_video_as_grayscale(address):

    '''
    capture frames from a video device, convert each frame to grayscale and display them
    write greyscale video files to new video file
    we will use h264 video codec with mp4 container type to save the video file
    '''

    capture = cv2.VideoCapture(address)

    if not capture.isOpened():
        print(f"Cannot opet camera '{address}'")
        return

    # why use XVID codec with avi container?
    # h264 with mp4 container would be better solution
    # https://github.com/skvark/opencv-python/issues/100#issuecomment-394159998
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter('movie_grey.avi', fourcc, frame_rate, (frame_width, frame_height))

    while True:
        # capture next frame from the capture device
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

        cv2.imshow("frame", frame)

        # enable stepping out from the loop by pressing the q kframe_heighty
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()
