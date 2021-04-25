import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
from threading import Thread
import imutils
import cv2
import functools


# Code credit: adapted from https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


# Code for Magenta neural style transfer credit: adapted from
# https://towardsdatascience.com/fast-neural-style-transfer-in-5-minutes-with-tensorflow-hub-magenta-110b60431dcc


style_path = tf.keras.utils.get_file('Vincent_van_gogh%2C_la_camera_da_letto%2C_1889%2C_02.jpg',
                                     'https://upload.wikimedia.org/wikipedia/commons/8/8c/Vincent_van_gogh%2C_la_camera_da_letto%2C_1889%2C_02.jpg')


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Cache image file locally.
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


# Loading the image that will style the content image
style_image = load_image(style_path)

# Load Magenta's Arbitrary Image Stylization network from TensorFlow Hub

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


# Note: caching the style constant tensor can't hurt?
style_constant = tf.constant(style_image)

# creates a *threaded* video stream
vs = WebcamVideoStream(src=0).start()

# loop over webcam frames using the threaded stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 256 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=256)

    # Convert image to dtype, scaling (MinMax Normalization) its values if needed.
    frame = tf.image.convert_image_dtype(frame, tf.float32)

    # Adds a fourth dimension to the Tensor because
    # the model requires a 4-dimensional Tensor
    frame = frame[tf.newaxis, :]

    # Pass content and style images as arguments in TensorFlow Constant object format
    frame = hub_module(tf.constant(frame), style_constant)[0]

    # Back to numpy array for display
    frame = np.array(frame[0])

    # Display and wait for key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press q or esc to end
    if key == ord("q") or key == 27:
        break

# Close window and stop loop/stream
cv2.destroyAllWindows()
vs.stop()