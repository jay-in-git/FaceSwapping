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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Src and Out images')
	parser.add_argument('src_image', type=str)
	parser.add_argument('tgt_image', type=str)
	parser.add_argument('-o', '--output', type=str, default='result.jpg')
	parser.add_argument('--method', type=str, default='poisson', choices=methods)
	parser.add_argument('--option', type=str, default='face', choices=options)

	argvs = parser.parse_args()
	src_image = cv2.imread(argvs.src_image)
	tgt_image = cv2.imread(argvs.tgt_image)
	src_mask = get_mask(src_image, argvs.option)
	tgt_mask = get_mask(tgt_image, argvs.option)

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
