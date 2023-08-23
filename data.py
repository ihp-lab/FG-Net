import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from data_utils import au2heatmap


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
							self.file_list['au24']
						]
		elif args['dataset'] == 'GFT':
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
							self.file_list['au24']
						]
		elif args['dataset'] == 'Aff-Wild2':
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
							self.file_list['au6'],
							self.file_list['au7'],
							self.file_list['au10'],
							self.file_list['au12'],
							self.file_list['au15'],
							self.file_list['au23'],
							self.file_list['au24'],
							self.file_list['au25'],
							self.file_list['au26']
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

	def __getitem__(self, index):
		# load image
		image_path = os.path.join(self.data_root, self.images[index])

		image = Image.open(image_path)
		image = self.transform(image)

		# load label
		label = []
		for i in range(self.args['num_labels']):
			label.append(int(self.labels[i][index]))
		label = torch.FloatTensor(label)

		if self.type == 'train':
			heatmap = au2heatmap(image_path, label, self.args['dim'], self.args['sigma'], self.args)
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
