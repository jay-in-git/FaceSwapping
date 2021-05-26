from operator import is_
import cv2
from os import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Src and Out images')
parser.add_argument('-i', type=str, dest="inPath")
parser.add_argument('-o', type=str, dest="outPath")

def getROIs(src, tgt):
    # left-up and right-down points
    return ((303, 437), (485, 663)), ((146, 644), (223, 746))

def getGradients(fig):
    kernel_x = np.array([0, -1, 1])
    kernel_y = np.array([[0], [-1], [1]])
    return cv2.filter2D(fig, -1, kernel_x), cv2.filter2D(fig, -1, kernel_y)

def getLaplacian(gx, gy):
    kernel_l = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    lap_x = cv2.filter2D(gx, -1, kernel_l)
    lap_y = cv2.filter2D(gy, -1, kernel_l)
    # lap_x[[0, lap_x.shape[0] - 1], :] = gx[[0, lap_x.shape[0] - 1], :]
    # lap_x[:, [0, lap_x.shape[1] - 1]] = gx[:, [0, lap_x.shape[1] - 1]]
    # lap_y[[0, lap_y.shape[0] - 1], :] = gy[[0, lap_y.shape[0] - 1], :]
    # lap_y[:, [0, lap_y.shape[1] - 1]] = gy[:, [0, lap_y.shape[1] - 1]]
    return lap_x + lap_y

def getA(fig):
    di = [0, -1, 0, 1]
    dj = [1, 0, -1, 0]
    A = dok_matrix((fig.shape[0] * fig.shape[1], fig.shape[0] * fig.shape[1]))
    print('Getting A...')
    for i in tqdm(range(fig.shape[0])):
        for j in range(fig.shape[1]):
            # if it's margin, A[row, row] should be 1
            row = i * fig.shape[0] + j
            is_margin = False
            for k in range(4):
                ni, nj = i + di[k], j + dj[k]
                if ni < 0 or ni >= fig.shape[0] or nj < 0 or nj >= fig.shape[1]:
                    is_margin = True
            if is_margin:
                A[row, row] = 1
                continue
            # if it's not, A[row, rrow] and the neighbor should be laplacian equation
            A[row, row] = -4
            for k in range(4):
                ni, nj = i + di[k], j + dj[k]
                col = ni * fig.shape[0] + nj
                A[row, col] = 1
    return A.tocsc()

if __name__ == '__main__':
    argvs = parser.parse_args()
    src = cv2.imread(argvs.inPath)
    tgt = cv2.imread(argvs.outPath)
    src_ROI, tgt_ROI = getROIs(src, tgt)
    ROI_pixels = src[src_ROI[0][0]:src_ROI[1][0] + 1, src_ROI[0][1]:src_ROI[1][1] + 1, :]
    ROI_pixels = cv2.resize(ROI_pixels, (tgt_ROI[1][1] - tgt_ROI[0][1] + 1, tgt_ROI[1][0] - tgt_ROI[0][0] + 1))
    
    ROI_gx, ROI_gy = getGradients(ROI_pixels)
    tgt_gx, tgt_gy = getGradients(tgt)
    tgt_gx[tgt_ROI[0][0]:tgt_ROI[1][0] + 1, tgt_ROI[0][1]:tgt_ROI[1][1] + 1] = ROI_gx
    tgt_gy[tgt_ROI[0][0]:tgt_ROI[1][0] + 1, tgt_ROI[0][1]:tgt_ROI[1][1] + 1] = ROI_gy
    # print(tgt_gx.shape)
    lap = getLaplacian(tgt_gx, tgt_gy)
    print(lap.shape)
    A = getA(lap)
    B = csc_matrix(lap.reshape((lap.shape[0] * lap.shape[1], lap.shape[2])))
    X = spsolve(A, B)
    print(X.toarray().shape)
    plt.imshow(X.toarray())
    plt.show()