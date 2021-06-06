import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from utility import align, combine, get_laplacian, get_pyramid

def poisson_edit(src_image: np.ndarray, tgt_image: np.ndarray, tgt_mask: np.ndarray, alpha=1, src_decay=255, tgt_decay=255) -> np.ndarray:
	"""Poisson image editing: result = tgt_image[tgt_mask != 0] union X achieved by poisson equation AX=B
	Args:
		src_image: image that contains the face to be pasted
		tgt_image: image that contains the face to be replaced
		tgt_mask: the binary mask image
		alpha: the ratio of mix gradient blending
	Returns:
		X: the solution to poisson equation
	"""
	src_image = src_image / src_decay
	tgt_image = tgt_image / tgt_decay

	src_lap = cv2.filter2D(src_image, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
	tgt_lap = cv2.filter2D(tgt_image, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
	src_lap = src_lap.reshape((tgt_image.shape[0] * tgt_image.shape[1], tgt_image.shape[2]))
	tgt_lap = tgt_lap.reshape((tgt_image.shape[0] * tgt_image.shape[1], tgt_image.shape[2]))

	A = get_laplacian(tgt_image.shape[0], tgt_image.shape[1], tgt_mask)
	print('Construction done.')

	tgt_mask_flatten = tgt_mask.flatten()
	tgt_array = tgt_image.reshape((tgt_image.shape[0] * tgt_image.shape[1], tgt_image.shape[2]))

	B_array = alpha * src_lap + (1 - alpha) * tgt_lap
	B_array[tgt_mask_flatten == 0] = tgt_array[tgt_mask_flatten == 0]
	B = csc_matrix(B_array)

	X = spsolve(A, B).toarray() * 255
	X = X.reshape(tgt_image.shape)
	X[X > 255] = 255
	X[X < 0] = 0
	return X

def direct_blending(src_image: np.ndarray, tgt_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Direct blending method: 
		result = src_image * mask + tgt_image * (1 - mask)
    Args:
	    src_image: image that contains the face to be pasted
	    tgt_image: image that contains the face to be replaced
	    mask: the binary mask image
    Returns:
	    tgt_image: tgt_image with face replaced with the face in src_image
    """
    return combine(src_image, tgt_image, mask)

def multi_resolution_blending(src_image: np.ndarray, tgt_image: np.ndarray, mask: np.ndarray, n_steps=6) -> np.ndarray:
	"""Multi-resolution blending method
	Args:
		src_image: image that contains the face to be pasted
		tgt_image: image that contains the face to be replaced
		mask: the binary mask image
		n_level: number of levels of the pyramid
	Returns:
		tgt_image: tgt_image with face replaced with the face in src_image
	"""
	mask = mask.astype(np.float64)
	src_image = src_image.astype(np.float64)
	tgt_image = tgt_image.astype(np.float64)

	print('Constructing pyramid...')
	src_gu, src_lap = get_pyramid(src_image, n_steps=n_steps)
	tgt_gu, tgt_lap = get_pyramid(tgt_image, n_steps=n_steps)
	mask_gu, _ = get_pyramid(mask, n_steps=n_steps)

	print('Reconstructing blend image')
	reconstruct = combine(src_gu[-1], tgt_gu[-1], mask_gu[-1])
	for i in range(n_steps - 1, -1, -1):
		row = int(src_lap[i].shape[0])
		col = int(src_lap[i].shape[1])
		reconstruct = cv2.resize(reconstruct, (col, row))
		laplacian = combine(src_lap[i], tgt_lap[i], mask_gu[i])
		reconstruct += laplacian
	return reconstruct.astype(np.uint8)

if __name__ == '__main__':
	from os import sys
	from utility import align
	src_image = cv2.imread(sys.argv[1])
	tgt_image = cv2.imread(sys.argv[2])
	src_mask = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
	tgt_mask = cv2.imread(sys.argv[4], cv2.IMREAD_GRAYSCALE)
	print('Aligning')
	src_image = align(src_image, src_mask, tgt_image, tgt_mask)
	print('done')
	result = poisson_edit(src_image, tgt_image, tgt_mask, alpha=0.7)
	cv2.imwrite('test.png', result)
