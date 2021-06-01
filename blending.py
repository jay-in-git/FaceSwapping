import cv2
import argparse
import numpy as np
from utility import align
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Src and Out images')
parser.add_argument('-s', type=str, dest="srcPath")
parser.add_argument('-t', type=str, dest="tgtPath")
parser.add_argument('-sm', type=str, dest="smPath")
parser.add_argument('-tm', type=str, dest="tmPath")
import time
def getTestCase(src, tgt):
    # left-up and right-down points
    return ((285, 437), (485, 663)), ((148, 640), (230, 743))

def getLaplacian(height, width, mask):
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

def poisson_edit(src_image: np.ndarray, src_mask: np.ndarray, tgt_image: np.ndarray, tgt_mask: np.ndarray, method='Normal'):
    src_image = src_image / 255
    tgt_image = tgt_image / 255
    ys, xs = np.nonzero(tgt_mask)
    tgt_obj_point = (ys.min(), xs.min())
    tgt_obj_height = ys.max() - ys.min() + 1
    tgt_obj_width = xs.max() - xs.min() + 1

    # tgt[tgt_mask != 0] = src[src_mask != 0]
    src_mask[src_mask != 0] = 1
    tgt_mask[tgt_mask != 0] = 1

    """Extract the src region and resize it to the tgt region"""
    src_lap = cv2.filter2D(src_image, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))

    A = getLaplacian(tgt_image.shape[0], tgt_image.shape[1], tgt_mask)
    print('Construction done.')

    tgt_mask_flatten = tgt_mask.flatten()
    tgt_array = tgt_image.reshape((tgt_image.shape[0] * tgt_image.shape[1], tgt_image.shape[2]))

    B_array = src_lap.reshape((tgt_image.shape[0] * tgt_image.shape[1], tgt_image.shape[2]))
    B_array[tgt_mask_flatten == 0] = tgt_array[tgt_mask_flatten == 0]
    B = csc_matrix(B_array)

    X = spsolve(A, B).toarray() * 255
    X = X.reshape(tgt_image.shape)
    X[X > 255] = 255
    X[X < 0] = 0

    return X

if __name__ == '__main__':
    argvs = parser.parse_args()
    src = cv2.imread(argvs.srcPath)
    tgt = cv2.imread(argvs.tgtPath)
    src_mask = cv2.imread(argvs.smPath, cv2.IMREAD_GRAYSCALE)
    tgt_mask = cv2.imread(argvs.tmPath, cv2.IMREAD_GRAYSCALE)
    result = poisson_edit(src, src_mask, tgt, tgt_mask, method='Normal')
    cv2.imwrite('merge.png', result)
