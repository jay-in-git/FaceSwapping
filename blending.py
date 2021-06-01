import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

def get_laplacian(height: int, width: int, mask: np.ndarray) -> csc_matrix:
	"""Construct laplacian matrix to solve the linear equation
	Args:
		height: the height of tgt image
		width: the width of tgt image
		mask: the mask of ROI of tgt image
	Returns:
		A: the poisson equation linear function
	"""
	A = dok_matrix((height * width, height * width))
	for y in tqdm(range(height)):
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

def poisson_edit(src_image: np.ndarray, src_mask: np.ndarray, tgt_image: np.ndarray, tgt_mask: np.ndarray, method='Normal') -> np.ndarray:
	src_image = src_image / 255
	tgt_image = tgt_image / 255
	ys, xs = np.nonzero(tgt_mask)
	tgt_obj_point = (ys.min(), xs.min())
	tgt_obj_height = ys.max() - ys.min() + 1
	tgt_obj_width = xs.max() - xs.min() + 1

	"""Extract the src region and resize it to the tgt region"""
	src_lap = cv2.filter2D(src_image, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
	roi = src_lap[tgt_obj_point[0]:tgt_obj_point[0] + tgt_obj_height, tgt_obj_point[1]:tgt_obj_point[1] + tgt_obj_width, :]

	A = get_laplacian(tgt_image.shape[0], tgt_image.shape[1], tgt_mask)
	print('Construction done.')

	tgt_image[tgt_obj_point[0]:tgt_obj_point[0] + tgt_obj_height, tgt_obj_point[1]:tgt_obj_point[1] + tgt_obj_width, :] = roi
	tgt_mat = tgt_image.reshape((tgt_image.shape[0] * tgt_image.shape[1], tgt_image.shape[2]))
	B = csc_matrix(tgt_mat)

	X = spsolve(A, B).toarray() * 255
	X = X.reshape(tgt_image.shape)
	X[X > 255] = 255
	X[X < 0] = 0

	return X

def direct_blending(src_image: np.ndarray, tgt_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""Direct blending method: result = src_image * mask + tgt_image * (1 - mask)
	Args:
		src_image: image that contains the face to be pasted
		tgt_image: image that contains the face to be replaced
		mask: the binary mask image
	Returns:
		tgt_image: tgt_image with face replaced with the face in src_image
	"""
	for i in range(3):
		tgt_image[:, :, i] = src_image[:,:,i] * mask + tgt_image[:, :, i] * (1 - mask)
	return tgt_image
