import cv2
import numpy as np

def align(src_image: np.ndarray, src_mask: np.ndarray, tgt_image: np.ndarray, tgt_mask: np.ndarray) -> np.ndarray:
    """Align the face in source image with the face in target image 
    Args:
        src_image: image that contains the face to be placed.
        src_mask: mask with respect to the source face (0 and 255).
        dst_image: image that contains the face to be replaced.
        dst_mask: mask with respect to the target face (0 and 255).
    Returns:
        aligned_src: aligned srouce image with the same size as dst_image
    """

    """Compute the size of source face"""
    ys, xs = np.nonzero(src_mask)
    src_center_y = (ys.min() + ys.max()) // 2
    src_center_x = (xs.min() + xs.max()) // 2
    src_height = ys.max() - ys.min() + 1
    src_width = xs.max() - xs.min() + 1

    """Compute the size of target face"""
    ys, xs = np.nonzero(tgt_mask)
    tgt_center_y = (ys.min() + ys.max()) // 2
    tgt_center_x = (xs.min() + xs.max()) // 2
    tgt_height = ys.max() - ys.min() + 1
    tgt_width = xs.max() - xs.min() + 1

    x_ratio = src_width / tgt_width
    y_ratio = src_height / tgt_height

    aligned_src = np.zeros(tgt_image.shape, dtype=np.uint8)

    for i in range(aligned_src.shape[0]):
        for j in range(aligned_src.shape[1]):
            ii = int(src_center_y + (i - tgt_center_y) * y_ratio)
            jj = int(src_center_x + (j - tgt_center_x) * x_ratio)
            if ii >= 0 and ii < src_image.shape[0] and jj >= 0 and jj < src_image.shape[1]:
                aligned_src[i][j] = src_image[ii][jj]
            else:
                aligned_src[i][j] = 0

    return aligned_src