import cv2
import numpy as np
from scipy.sparse import dok_matrix, csc_matrix

def combine(src_image: np.ndarray, tgt_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""Perform result_image = src_image * mask + tgt_image * (1 - mask)
	Args:
	    src_image: source image that are keep if mask = 1
	    tgt_image: target image that are keep if mask = 0
	    mask: the binary mask image
    Returns:
	    tgt_image: tgt_image with face replaced with the face in src_image
	"""
	for i in range(3):
		tgt_image[:, :, i] = src_image[:,:,i] * mask + tgt_image[:, :, i] * (1 - mask)
	return tgt_image

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

def get_laplacian(height: int, width: int, mask: np.ndarray) -> csc_matrix:
	"""Get laplacian matrix: the linear equation matrix A from AX = B
	Args:
		height: the height of tgt image
		width: the width of tgt image
		mask: the mask of ROI of tgt image
	Returns:
		A: the poisson equation linear function
	"""
	A = dok_matrix((height * width, height * width))
	for y in range(height):
		for x in range(width):
			row = y * width + x
			if mask[y, x] == 0:
				A[row, row] = 1
			else:
				A[row, row] = 4
				A[row, row + 1] = -1
				A[row, row - 1] = -1
				A[row, row - width] = -1
				A[row, row + width] = -1
	return A.tocsc()

def get_pyramid(image: np.ndarray, n_steps=4) -> tuple[np.ndarray, np.ndarray]:
	"""Compute Gaussian pyramid and Laplacian pyramid for image
	Args:
		image: the source image
		n_steps: number of steps to construct pyramids
	Returns:
		gaussian_pyramid: Gaussian pyramid with length n_steps + 1
		laplacian_pyramid: Laplacian pyramid with length n_steps
	"""
	gaussian_pyramid = [image.copy()]
	laplacian_pyramid = []
	for i in range(n_steps):
		image_current = gaussian_pyramid[-1]
		image_next = cv2.GaussianBlur(image_current, (5, 5), 0)
		laplacian_pyramid.append(image_current - image_next)

		col = int(np.ceil(image_next.shape[1] / 2))
		row = int(np.ceil(image_next.shape[0] / 2))
		image_next = cv2.resize(image_next, (col, row))
		gaussian_pyramid.append(image_next)
	
	return gaussian_pyramid, laplacian_pyramid
