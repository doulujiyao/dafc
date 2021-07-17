""" Face landmarks utilities. """

import math
import numpy as np
import cv2
from fsgan.utils.bbox_utils import scale_bbox, crop_img


# Adapted from: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
def _gaussian(size=3, sigma=0.25, amplitude=1, normalize=False, width=None, height=None, sigma_horz=None,
              sigma_vert=None, mean_horz=0.5, mean_vert=0.5):
    """ Generate a guassian kernel.

    Args:
        size (int): The size of the kernel if the width or height are not specified
        sigma (float): Standard deviation of the kernel if sigma_horz or sigma_vert are not specified
        amplitude: The scale of the kernel
        normalize: If True, the kernel will be normalized such as values will sum to one
        width (int, optional): The width of the kernel
        height (int, optional): The height of the kernel
        sigma_horz (float, optional): Horizontal standard deviation of the kernel
        sigma_vert (float, optional): Vertical standard deviation of the kernel
        mean_horz (float): Horizontal mean of the kernel
        mean_vert (float): Vertical mean of the kernel

    Returns:
        np.array: The computed gaussian kernel
    """
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                    sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)

    return gauss


# Adapted from: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
def draw_gaussian(image, point, sigma,g):
    """ Draw gaussian circle at a point in an image.

    Args:
        image (np.array): An image of shape (H, W)
        point (np.array): The center point of the guassian circle
        sigma (float): Standard deviation of the gaussian kernel

    Returns:
        np.array: The image with the drawn gaussian.
    """
    # Check if the gaussian is inside
    point[0] = round(point[0], 2)
    point[1] = round(point[1], 2)

    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] = \
        image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1

    return image


# Adapted from: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/api.py
def generate_heatmaps(height, width, points, sigma=None):
    """ Generate heatmaps corresponding to a set of points.

    Args:
        height (int): Heatmap height
        width (int): Heatmap width
        points (np.array): An array of points of shape (N, 2)
        sigma (float, optional): Standard deviation of the gaussian kernel. If not specified it will be determined
            from the width of the heatmap

    Returns:
        np.array: The generated heatmaps.
    """
    sigma = max(1, int(np.round(width / 128.))) if sigma is None else sigma
    heatmaps = np.zeros((1, height, width), dtype=np.float32)
    size = 6 * sigma + 1
    g = _gaussian(size)
    for i in range(points.shape[0]):
        if points[i, 0] > 0:
            heatmaps[0] = draw_gaussian(
                heatmaps[0], points[i], sigma,g)

    return heatmaps


def heatmap2rgb(heatmap):
    """ Convert heatmap to an RGB image.

    Args:
        heatmap (np.array): The heatmap to convert

    Returns:
        np.array: RGB image representation of the heatmap
    """
    m = heatmap.mean(axis=0)
    rgb = np.stack((m, m, m), axis=-1)
    rgb *= 1.0 / rgb.max()

    return rgb


def hflip_face_landmarks(landmarks, width):
    """ Horizontal flip 68 points landmarks.

    Args:
        landmarks (np.array): Landmarks points of shape (68, 2)
        width (int): The width of the correspondign image

    Returns:
        np.array: Horizontally flipped landmarks.
    """
    landmarks = landmarks.copy()

    # Invert X coordinates
    for p in landmarks:
        p[0] = width - p[0]

    # Jaw
    right_jaw, left_jaw = list(range(0, 8)), list(range(16, 8, -1))
    landmarks[right_jaw + left_jaw] = landmarks[left_jaw + right_jaw]

    # Eyebrows
    right_brow, left_brow = list(range(17, 22)), list(range(26, 21, -1))
    landmarks[right_brow + left_brow] = landmarks[left_brow + right_brow]

    # Nose
    right_nostril, left_nostril = list(range(31, 33)), list(range(35, 33, -1))
    landmarks[right_nostril + left_nostril] = landmarks[left_nostril + right_nostril]

    # Eyes
    right_eye, left_eye = list(range(36, 42)), [45, 44, 43, 42, 47, 46]
    landmarks[right_eye + left_eye] = landmarks[left_eye + right_eye]

    # Mouth outer
    mouth_out_right, mouth_out_left = [48, 49, 50, 59, 58], [54, 53, 52, 55, 56]
    landmarks[mouth_out_right + mouth_out_left] = landmarks[mouth_out_left + mouth_out_right]

    # Mouth inner
    mouth_in_right, mouth_in_left = [60, 61, 67], [64, 63, 65]
    landmarks[mouth_in_right + mouth_in_left] = landmarks[mouth_in_left + mouth_in_right]

    return landmarks


def align_crop(img, landmarks, bbox, scale=2.0, square=True):
    """ Align and crop image and corresponding landmarks by bounding box.

    Args:
        img (np.array): An image of shape (H, W, 3) PIL
        landmarks (np.array): Face landmarks points of shape (68, 2)
        bbox (np.array): Bounding box in the format [left, top, width, height]
        scale (float): Multiply the bounding box by this scale
        square (bool): If True, make the shorter edges of the bounding box equal the length as the longer edges

    Returns:
        (np.array, np.array): A tuple of numpy arrays containing:
            - Aligned and cropped image (np.array)
            - Aligned and cropped landmarks (np.array)
    """
    # Rotate image for horizontal eyes
    right_eye_center = landmarks[36:42, :].mean(axis=0)
    left_eye_center = landmarks[42:48, :].mean(axis=0)
    # eye_center = np.round(np.mean(landmarks[:2], axis=0)).astype(int)
    eye_center = (right_eye_center + left_eye_center) / 2.0
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.)
    output = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    # Adjust landmarks
    new_landmarks = np.concatenate((landmarks, np.ones((68, 1))), axis=1)
    new_landmarks = new_landmarks.dot(M.transpose())

    # Scale bounding box
    bbox_scaled = scale_bbox(bbox, scale, square)

    # Crop image
    output, new_landmarks = crop_img(output, new_landmarks, bbox_scaled)

    return output, new_landmarks


def render_landmarks_lines_68pts(img, landmarks, thickness=1):
    """ Render 68 points landmarks with connected lines.

    Args:
        img (np.array): An image of shape (H, W, 3)
        landmarks (np.array): Face landmarks points of shape (68, 2)
        thickness (int): Line thickness [pixels]

    Returns:
        np.array: The rendered image.
    """
    assert landmarks.shape[0] == 68
    color = (255, 255, 255)
    for i in range(1, 17):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    for i in range(28, 31):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    for i in range(23, 27):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    for i in range(31, 36):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    for i in range(37, 42):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    for i in range(43, 48):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    for i in range(49, 60):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    for i in range(1, 17):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    for i in range(61, 68):
        cv2.line(img, landmarks[i], landmarks[i - 1], color, thickness)
    cv2.line(img, landmarks[60], landmarks[67], color, thickness)
