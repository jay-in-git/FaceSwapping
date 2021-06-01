import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utility import *
import blending

methods = ['poisson', 'direct', 'gaussian']

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Src and Out images')
	parser.add_argument('src_image', type=str)
	parser.add_argument('tgt_image', type=str)
	parser.add_argument('src_mask', type=str)
	parser.add_argument('tgt_mask', type=str)
	parser.add_argument('-o', '--output', type=str, default='result.jpg')
	parser.add_argument('--method', type=str, default='poisson', choices=methods)
	
	argvs = parser.parse_args()
	src_image = cv2.imread(argvs.src_image)
	tgt_image = cv2.imread(argvs.tgt_image)
	src_mask = cv2.imread(argvs.src_mask, cv2.IMREAD_GRAYSCALE) // 255
	tgt_mask = cv2.imread(argvs.tgt_mask, cv2.IMREAD_GRAYSCALE) // 255

	src_image = align(src_image, src_mask, tgt_image, tgt_mask)
	print('Image aligned!')
	if argvs.method == methods[0]:
		result = blending.poisson_edit(src_image, src_mask, tgt_image, tgt_mask, method='Normal')
	elif argvs.method == methods[1]:
		result = blending.direct_blending(src_image, tgt_image, tgt_mask)
	else:
		pass

	cv2.imwrite(argvs.output, result)
