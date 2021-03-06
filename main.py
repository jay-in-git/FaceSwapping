import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utility import *
import blending
from mask import get_mask
methods = ['poisson', 'direct', 'multi']
options = ['face', 'head', 'eye', 'mouth', 'nose']

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Src and Out images')
	parser.add_argument('src_image', type=str, help="Image with the face to be pasted")
	parser.add_argument('tgt_image', type=str, help="Imgae with the face to be replaced")
	parser.add_argument('-o', '--output', type=str, default='result.jpg', help="Path to the result image")
	parser.add_argument('--method', type=str, default='poisson', choices=methods, help="Blending method")
	parser.add_argument('--option', type=str, default='face', choices=options, help="Specify specific part of face to be replaced")
	parser.add_argument('--mask_ratio', type=float, default=0.85, choices=[Range(0.7, 1.0)], help="Ratio between mask boundary and real face boundary. Range=[0.75, 1] (Bigger means that mask is closer to real face size)")
	
	argvs = parser.parse_args()
	src_image = cv2.imread(argvs.src_image)
	tgt_image = cv2.imread(argvs.tgt_image)
	src_mask = get_mask(src_image, argvs.option, argvs.mask_ratio)
	tgt_mask = get_mask(tgt_image, argvs.option, argvs.mask_ratio)

	print('Aligning image...')
	src_image = align(src_image, src_mask, tgt_image, tgt_mask)

	if argvs.method == methods[0]:
		result = blending.poisson_edit(src_image, tgt_image, tgt_mask)
	elif argvs.method == methods[1]:
		result = blending.direct_blending(src_image, tgt_image, tgt_mask)
	else:
		result = blending.multi_resolution_blending(src_image, tgt_image, tgt_mask)

	print(f'Result image stored to {argvs.output}')
	cv2.imwrite(argvs.output, result)
