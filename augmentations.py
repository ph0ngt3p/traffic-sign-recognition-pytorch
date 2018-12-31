import random
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte


def rand_similarity_trans(image, n):
    """
    apply random similarity transformation to the image, and return
    n transformed images
    """
    output_images = np.uint8(np.zeros((n, 32, 32, 1)))

    for i in range(n):
        angle = random.uniform(-15, 15)  # rotation

        s = random.uniform(0.9, 1.1)  # scale

        rows, cols = image.shape[0:2]
        image_center = (rows / 2.0, cols / 2.0)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
        M_rot = np.vstack([rot_mat, [0, 0, 1]])

        tx = random.uniform(-2, 2)  # translation along x axis
        ty = random.uniform(-2, 2)  # translation along y axis
        M_tran = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

        M = np.matrix(M_tran) * np.matrix(M_rot)

        M = np.float32(M[:2][:])  # similarity transform

        tmp = cv2.warpAffine(image, M, (cols, rows))
        tmp = tmp.reshape(tmp.shape + (1, ))
        output_images[i, :, :, :] = tmp
        # print(output_images[i, :, :, :].shape)

        # cv2.equalizeHist(image, image)

    return output_images


# def test():
#     img = cv2.imread('data/train/00000/00000_00029.ppm')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (32, 32))
#     img = rand_similarity_trans(img, 1)
#     print(img.dtype)
#     print(img[0, :, :, :].dtype)
#
#
# test()
