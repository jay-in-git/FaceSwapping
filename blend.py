import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csc_matrix, lil_matrix, block_diag, dia_matrix
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
    # Construct laplacian matrix
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
    # A.setdiag(4, 0)
    # A.setdiag(-1, -1)
    # A.setdiag(-1, 1)
    # A.setdiag(-1, -width)
    # A.setdiag(-1, width)
    # A = A.todok()
    # ys, xs = np.where(mask == 0)
    # start = time.time()
    # for y in tqdm(range(1, height - 1)):
    #     for x in range(1, width - 1):
    #         if mask[y, x] == 0:
    #             row = y * width + x
    #             A[row, row] = 1
    #             A[row, row + 1] = 0
    #             A[row, row - 1] = 0
    #             A[row, row + width] = 0
    #             A[row, row - width] = 0
    return A.tocsc()

def poisson_edit(source: np.ndarray, source_mask: np.ndarray, target: np.ndarray, target_mask: np.ndarray, method='Normal'):
    source = source / 255
    target = target / 255
    ys, xs = np.nonzero(source_mask)
    src_obj_point = (ys.min(), xs.min())
    src_obj_height = ys.max() - ys.min() + 1
    src_obj_width = xs.max() - xs.min() + 1

    ys, xs = np.nonzero(target_mask)
    tgt_obj_point = (ys.min(), xs.min())
    tgt_obj_height = ys.max() - ys.min() + 1
    tgt_obj_width = xs.max() - xs.min() + 1

    # target[target_mask != 0] = source[source_mask != 0]
    source_mask[source_mask != 0] = 1
    target_mask[target_mask != 0] = 1

    """Extract the source region and resize it to the target region"""
    source_lap = cv2.filter2D(source, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
    roi = source_lap[src_obj_point[0]:src_obj_point[0] + src_obj_height, src_obj_point[1]:src_obj_point[1] + src_obj_width, :]
    roi = cv2.resize(roi, (tgt_obj_width, tgt_obj_height))

    A = getLaplacian(target.shape[0], target.shape[1], target_mask)
    print('Construction done.')

    target[tgt_obj_point[0]:tgt_obj_point[0] + tgt_obj_height, tgt_obj_point[1]:tgt_obj_point[1] + tgt_obj_width, :] = roi
    target_mat = target.reshape((target.shape[0] * target.shape[1], target.shape[2]))
    B = csc_matrix(target_mat)

    X = spsolve(A, B).toarray()
    X = X.reshape(target.shape)
    X[X > 255] = 255
    X[X < 0] = 0

    return X * 255

if __name__ == '__main__':
    argvs = parser.parse_args()
    src = cv2.imread(argvs.srcPath)
    tgt = cv2.imread(argvs.tgtPath)
    src_mask = cv2.imread(argvs.smPath, cv2.IMREAD_GRAYSCALE)
    tgt_mask = cv2.imread(argvs.tmPath, cv2.IMREAD_GRAYSCALE)
    result = poisson_edit(src, src_mask, tgt, tgt_mask, method='Normal')
    cv2.imwrite('merge.png', result)
