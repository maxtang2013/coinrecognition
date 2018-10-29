import numpy as np
import argparse
import cv2
import os
import os.path
import argparse

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

def bbox(img):
	"""
	Find the range where there is none-zero pixels.
	"""
	a = np.where(img != 0)
	bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
	return bbox

def extract_coin_from(filepath, outpath):
	"""
	Extract the coin inside the image, and save it to outpath.
	We assume that the image contains one coin with white background.
	"""
	# load the image, clone it for output, and then convert it to grayscale
	image = cv2.imread(filepath,  0)

	print ("processing file {0}".format(filepath))

	# to binary
	ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	# exclude pepper and salt noises
	thresh = cv2.medianBlur(thresh, 3)

	# cv2.imwrite("thresh.png",thresh)

	x1,x2,y1,y2 = bbox(thresh)

	center = ((x1+x2+1)//2, (y1+y2+1)//2)
	l = max((-x1+x2+1)//2, (-y1+y2+1)//2)
	img = image[center[0]-l:center[0]+l, center[1]-l:center[1]+l]

	cv2.imwrite(outpath, img)

def increaseContrast(gray):

	height,width = gray.shape
	arr = gray.reshape(height*width)

	# count the histogram of the original image
	# using numpy methods
	# iterating through image using gray[i, j] for every pair of (i,j) is extremely slow.
	hist = np.bincount(arr, minlength=256)
	
	hist = np.true_divide(hist, width*height)
	
	intensity = np.zeros(256)
	
	for i in range(0, 255):
		intensity[i+1] = hist[i + 1] * 255 + intensity[i]
	
	def mapFunc(x):
		return intensity[x]
	
	out = np.vectorize(mapFunc)(arr)
	
	return out.reshape((height,width)).astype(np.uint8)

def extract_coin_sub_background(filepath, outpath):
	origin = cv2.imread(filepath)
	im = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
	
	kenel = np.ones((15,15), np.uint8)
	bg = cv2.morphologyEx(im, cv2.MORPH_OPEN, kenel)

	fg = im - bg
	# cv2.imwrite(outpath.replace(".png", "_fg1.png"), fg)
	fg = cv2.GaussianBlur(fg, (3,3),0)
	# fg = increaseContrast(fg)

	# to binary
	ret, thresh = cv2.threshold(fg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	# exclude pepper and salt noises
	thresh = cv2.medianBlur(thresh, 3)
	thresh = 255 - thresh

	# cv2.imwrite(outpath.replace(".png", "_thresh.png"), thresh)

	x1,x2,y1,y2 = bbox(thresh)

	center = ((x1+x2+1)//2, (y1+y2+1)//2)
	l = max((-x1+x2+1)//2, (-y1+y2+1)//2)
	img = origin[center[0]-l:center[0]+l, center[1]-l:center[1]+l]

	cv2.imwrite(outpath, img)


def extract_coin_watershed(filepath, outpath):
	"""
	Using watershed algorithm to extract the coin inside the image, and save it to outpath.
	We assume that the image contains one coin with white background.
	"""
	# load the image, clone it for output, and then convert it to grayscale
	image = cv2.imread(filepath,  0)

	print ("processing file {0}".format(filepath))

	distance = ndi.distance_transform_edt(image)
	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
								labels=image)
	markers = ndi.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=image)

	np.savetxt("labels.txt", labels)
	print (labels.dtype)
	print (labels)

	# to binary
	# ret, thresh = cv2.threshold(labels,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	# exclude pepper and salt noises
	# thresh = cv2.medianBlur(thresh, 3)

	cv2.imwrite("labels.png",labels)
	return
	# cv2.imwrite("thresh.png",thresh)

	x1,x2,y1,y2 = bbox(thresh)

	center = ((x1+x2+1)//2, (y1+y2+1)//2)
	l = max((-x1+x2+1)//2, (-y1+y2+1)//2)
	img = image[center[0]-l:center[0]+l, center[1]-l:center[1]+l]

	cv2.imwrite(outpath, img)


def extract_all_coins_in(dir, out_dir):
	files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

	count = 1
	for f in files:
		if f.endswith('.jpg'):
			print ("Processing {0}".format(f))
			outpath = os.path.join(out_dir, "{0}.png".format(count))
			# extract_coin_from(f, outpath)
			# extract_coin_watershed(f, outpath)
			extract_coin_sub_background(f, outpath)
			count += 1


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", dest="dir", required = True, help = "Path to the directory that contains all the original photos")
ap.add_argument("-o", "--out-dir", dest="out_dir", required = True, help = "Path to the directory that contains all the original photos")
args = ap.parse_args()

extract_all_coins_in(args.dir, args.out_dir)