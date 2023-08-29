import os
import cv2
import dlib
import torch
import numpy as np

from PIL import Image
from imutils import face_utils
from torch.utils.data import Dataset


def findlandmark(img_path):
	cascade = '../face_align/shape_predictor_68_face_landmarks.dat'
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(cascade)

	image = cv2.imread(img_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		return shape


def add_noise(heatmap, channel, label, center_1, center_2, sigma, size, threshold=0.01):
	gauss_noise_1 = np.fromfunction(lambda y,x : ((x-center_1[0])**2 \
											+ (y-center_1[1])**2) / -(2.0*sigma*sigma),
											(size, size), dtype=int)
	gauss_noise_1 = np.exp(gauss_noise_1)
	gauss_noise_1[gauss_noise_1 < threshold] = 0
	gauss_noise_1[gauss_noise_1 > 1] = 1
	gauss_noise_2 = np.fromfunction(lambda y,x : ((x-center_2[0])**2 \
											+ (y-center_2[1])**2) / -(2.0*sigma*sigma),
											(size, size), dtype=int)
	if label[channel] == 1:
		heatmap[channel] += gauss_noise_1
	else:
		heatmap[channel] -= gauss_noise_1
	if center_2[0] == -1:
		return heatmap

	gauss_noise_2 = np.exp(gauss_noise_2)
	gauss_noise_2[gauss_noise_2 < threshold] = 0
	gauss_noise_2[gauss_noise_2 > 1] = 1
	if label[channel] == 1:
		heatmap[channel] += gauss_noise_2
	else:
		heatmap[channel] -= gauss_noise_2
	return heatmap


def au2heatmap(img_path, label, size, sigma, args):
	lmk_path = img_path.replace('aligned_images', 'aligned_landmarks')[:-4]+'.npy'
	if os.path.exists(lmk_path):
		lmk = np.load(lmk_path)
	else:
		print(img_path)
		lmk = findlandmark(img_path)

	if size == 128:
		lmk = lmk/2 # [68, 2]
	elif size == 64:
		lmk = lmk/4 # [68, 2]

	heatmap = np.zeros((args['num_labels'], size, size))
	lmk_eye_left = lmk[36:42]
	lmk_eye_right = lmk[42:48]
	eye_left = np.mean(lmk_eye_left, axis=0)
	eye_right = np.mean(lmk_eye_right, axis=0)
	lmk_eyebrow_left = lmk[17:22]
	lmk_eyebrow_right = lmk[22:27]
	eyebrow_left = np.mean(lmk_eyebrow_left, axis=0)
	eyebrow_right = np.mean(lmk_eyebrow_right, axis=0)
	IOD = np.linalg.norm(lmk[42] - lmk[39])

	if args['dataset'] == 'BP4D':
		# au1 lmk 21, 22
		heatmap = add_noise(heatmap, 0, label, lmk[21], lmk[22], sigma, size)

		# au2 lmk 17, 26
		heatmap = add_noise(heatmap, 1, label, lmk[17], lmk[26], sigma, size)

		# au4 brow center
		heatmap = add_noise(heatmap, 2, label, eyebrow_left, eyebrow_right, sigma, size)

		# au6 1 scale below eye bottom
		heatmap = add_noise(heatmap, 3, label, [eye_left[0], eye_left[1]+IOD], [eye_right[0], eye_right[1]+IOD], sigma, size)

		# au7 lmk 38, 43
		heatmap = add_noise(heatmap, 4, label, lmk[38], lmk[43], sigma, size)

		# au10 lmk 50, 52
		heatmap = add_noise(heatmap, 5, label, lmk[50], lmk[52], sigma, size)

		# au12 lmk 48, 54
		heatmap = add_noise(heatmap, 6, label, lmk[48], lmk[54], sigma, size)

		# au14 lmk 48, 54
		heatmap = add_noise(heatmap, 7, label, lmk[48], lmk[54], sigma, size)

		# au15 lmk 48, 54
		heatmap = add_noise(heatmap, 8, label, lmk[48], lmk[54], sigma, size)

		# au17 0.5 scale below lmk 56, 58 / 0.5 scale below lip center
		heatmap = add_noise(heatmap, 9, label, [lmk[56,0], lmk[56,1]+0.5*IOD], [lmk[58,0], lmk[58,1]+0.5*IOD], sigma, size)

		# au23 lmk 51, 57 / lip center
		heatmap = add_noise(heatmap, 10, label, lmk[51], lmk[57], sigma, size)

		# au24 lmk 51, 57 / lip center
		heatmap = add_noise(heatmap, 11, label, lmk[51], lmk[57], sigma, size)
	elif args['dataset'] == 'DISFA':
		# au1 lmk 21, 22
		heatmap = add_noise(heatmap, 0, label, lmk[21], lmk[22], sigma, size)

		# au2 lmk 17, 26
		heatmap = add_noise(heatmap, 1, label, lmk[17], lmk[26], sigma, size)

		# au4 brow center
		heatmap = add_noise(heatmap, 2, label, eyebrow_left, eyebrow_right, sigma, size)

		# au6 1 scale below eye bottom
		heatmap = add_noise(heatmap, 3, label, [eye_left[0], eye_left[1]+IOD], [eye_right[0], eye_right[1]+IOD], sigma, size)

		# au9 0.5 scale below lmk 39, 42 / lmk 39, 42 / lmk 21, 22
		heatmap = add_noise(heatmap, 4, label, lmk[39], lmk[42], sigma, size)

		# au12 lmk 48, 54
		heatmap = add_noise(heatmap, 5, label, lmk[48], lmk[54], sigma, size)

		# au25 lmk 51, 57
		heatmap = add_noise(heatmap, 6, label, lmk[51], lmk[57], sigma, size)

		# au26 0.5 scale below lmk 56, 58 / lmk 56, 58
		heatmap = add_noise(heatmap, 7, label, lmk[56], lmk[58], sigma, size)

	heatmap = np.clip(heatmap, -1., 1.)

	return heatmap

def heatmap2au(heatmap):
	avg = torch.mean(heatmap, dim=(2,3))
	label = (avg > 0).int()

	return label


class MyDataset(Dataset):
	def __init__(self, file_list, transform, type, args):
		self.file_list = file_list
		self.transform = transform
		self.args = args

		self.data_root = args['image_path']
		self.images = self.file_list['image_path']
		self.type = type
		if args['dataset'] == 'BP4D':
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
							self.file_list['au6'],
							self.file_list['au7'],
							self.file_list['au10'],
							self.file_list['au12'],
							self.file_list['au14'],
							self.file_list['au15'],
							self.file_list['au17'],
							self.file_list['au23'],
							self.file_list['au24'],
						]
		else:
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
							self.file_list['au6'],
							self.file_list['au9'],
							self.file_list['au12'],
							self.file_list['au25'],
							self.file_list['au26']
						]
		self.num_labels = len(self.labels)

	def __getitem__(self, index):
		# load image
		image_path = os.path.join(self.data_root, self.images[index])

		image = Image.open(image_path)
		image = self.transform(image)

		# load label
		label = []
		for i in range(self.num_labels):
			label.append(int(self.labels[i][index]))
		label = torch.FloatTensor(label)

		if self.type == 'train':
			heatmap = au2heatmap(image_path, label, self.args['dim'], self.args['sigma'], self.args) # [x,128,128]
			heatmap = torch.from_numpy(heatmap)

			return image, label, heatmap
		else:
			return image, label

	def collate_fn(self, data):
		if self.type == 'train':
			images, labels, heatmaps = zip(*data)

			images = torch.stack(images).cuda()
			labels = torch.stack(labels).cuda()
			heatmaps = torch.stack(heatmaps).float().cuda()

			return images, labels, heatmaps
		else:
			images, labels = zip(*data)

			images = torch.stack(images).cuda()
			labels = torch.stack(labels).cuda()

			return images, labels

	def __len__(self):
		return len(self.file_list)
