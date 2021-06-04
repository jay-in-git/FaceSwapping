import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from utility import align, combine, get_laplacian


def poisson_edit(src_image: np.ndarray, tgt_image: np.ndarray, tgt_mask: np.ndarray, alpha=1) -> np.ndarray:
	"""Poisson image editing: result = tgt_image[tgt_mask != 0] union X achieved by poisson equation AX=B
	Args:
		src_image: image that contains the face to be pasted
		tgt_image: image that contains the face to be replaced
		tgt_mask: the binary mask image
		alpha: the ratio of mix gradient blending
	Returns:
		X: the solution to poisson equation
	"""
	src_image = src_image / 255
	tgt_image = tgt_image / 255

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
    """Direct blending method: result = src_image * mask + tgt_image * (1 - mask)
    Args:
	    src_image: image that contains the face to be pasted
	    tgt_image: image that contains the face to be replaced
	    mask: the binary mask image
    Returns:
	    tgt_image: tgt_image with face replaced with the face in src_image
    """
    return combine(src_image, tgt_image, mask)

def multi_resolution_blending(src_image: np.ndarray, tgt_image: np.ndarray, mask: np.ndarray, n_level=5) -> np.ndarray:
	"""Direct blending method: 
		result = src_image * mask + tgt_image * (1 - mask)
	Args:
		src_image: image that contains the face to be pasted
		tgt_image: image that contains the face to be replaced
		mask: the binary mask image
		n_level: number of levels of the pyramid
	Returns:
		tgt_image: tgt_image with face replaced with the face in src_image
	"""
	mask = mask.astype(np.float32)

	print('Constructing gaussian pyramid...')
	src_gaussian_pyramid = [src_image.copy()]
	tgt_gaussian_pyramid = [tgt_image.copy()]
	mask_gaussian_pyramid = [mask.copy()]
	for i in range(n_level):
		src_image = cv2.pyrDown(src_image)
		tgt_image = cv2.pyrDown(tgt_image)
		mask = cv2.pyrDown(mask)
		src_gaussian_pyramid.append(src_image)
		tgt_gaussian_pyramid.append(tgt_image)
		mask_gaussian_pyramid.append(mask)

	print('Shapes of gaussian paramid:')
	print('\tsource:', [i.shape for i in src_gaussian_pyramid])
	print('\ttarget:', [i.shape for i in tgt_gaussian_pyramid])
	print('\tmask:', [i.shape for i in mask_gaussian_pyramid])

	print('Constructing Laplacian pyramid...')
	src_laplacian_pyramid = [src_gaussian_pyramid[n_level]]
	tgt_laplacian_pyramid = [tgt_gaussian_pyramid[n_level]]
	for i in range(n_level, 0 , -1):
		g = cv2.pyrUp(
			src_gaussian_pyramid[i], 
			dstsize=src_gaussian_pyramid[i-1].shape[1::-1],
		)
		src_laplacian_pyramid.append(
			cv2.subtract(src_gaussian_pyramid[i-1], g),
		)
		g = cv2.pyrUp(
			tgt_gaussian_pyramid[i], 
			dstsize=tgt_gaussian_pyramid[i-1].shape[1::-1],
		)
		tgt_laplacian_pyramid.append(
			cv2.subtract(tgt_gaussian_pyramid[i-1], g),
		)

	print('Shapes of laplacian paramid:')
	print('\tsource:', [i.shape for i in src_laplacian_pyramid])
	print('\ttarget:', [i.shape for i in tgt_laplacian_pyramid])
	
	# fig, axs = plt.subplots(2, n_level + 1)
	# for i in range(n_level+1):
	# 	axs[0][i].imshow(src_laplacian_pyramid[i])
	# for i in range(n_level+1):
	# 	axs[1][i].imshow(tgt_laplacian_pyramid[i])
	# plt.show()

	mask_gaussian_pyramid.reverse()
	reconstruct = combine(
		src_laplacian_pyramid[0], 
		tgt_laplacian_pyramid[0], 
		mask_gaussian_pyramid[0]
	).astype(np.uint8)
	for i in range(1, n_level+1):
		# plt.imshow(reconstruct[:,:,::-1])
		# plt.show()
		reconstruct = cv2.pyrUp(
			reconstruct, 
			dstsize=tgt_laplacian_pyramid[i].shape[1::-1],
		)
		laplacian = combine(
			src_laplacian_pyramid[i], 
			tgt_laplacian_pyramid[i], 
			mask_gaussian_pyramid[i],
		).astype(np.uint8)
		reconstruct = cv2.add(reconstruct, laplacian)

	return reconstruct

if __name__ == '__main__':
	from os import sys
	from utility import align
	src_image = cv2.imread(sys.argv[1])
	tgt_image = cv2.imread(sys.argv[2])
	src_mask = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
	tgt_mask = cv2.imread(sys.argv[4], cv2.IMREAD_GRAYSCALE)

	result = poisson_edit(align(src_image, src_mask, tgt_image, tgt_mask), tgt_image, tgt_mask, alpha=0.8)
	cv2.imwrite('test.png', result)