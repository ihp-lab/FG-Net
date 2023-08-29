import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
	def __init__(self, size, mode, align_corners=False):
		super(Interpolate, self).__init__()
		self.interp = nn.functional.interpolate
		self.size = size
		self.mode = mode
		self.align_corners = align_corners

	def forward(self, x):
		if self.align_corners:
			x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
		else:
			x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
		return x


class pyramid_interpreter(nn.Module):
	def __init__(self, points, dropout):
		super(pyramid_interpreter, self).__init__()
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
		self.layers = []
		self.layer0 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=512),
		)
		self.layer1 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(512, 512, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=512),
		)
		self.layer2 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(512, 512, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=512),
		)
		self.layer3 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(512, 512, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=512),
		)
		self.layer4 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(512, 256, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=256),
		)
		self.layer5 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=128),
		)
		self.layer6 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=64),
		)
		self.layer7 = nn.Sequential(
			nn.Conv2d(64, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=32),
		)
		self.layer8 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=32),
			nn.Conv2d(32, points, kernel_size=5, padding=2),
		)
		self.interpolate = Interpolate(128, 'bilinear')

		self.layers.append(self.layer0)
		self.layers.append(self.layer1)
		self.layers.append(self.layer2)
		self.layers.append(self.layer3)
		self.layers.append(self.layer4)
		self.layers.append(self.layer5)
		self.layers.append(self.layer6)
		self.layers.append(self.layer7)
		self.layers.append(self.layer8)

	def init_weights(self, init_type='normal', gain=0.02):
		def init_func(m):
			classname = m.__class__.__name__
			if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
				if init_type == 'normal':
					nn.init.normal_(m.weight.data, 0.0, gain)
				elif init_type == 'xavier':
					nn.init.xavier_normal_(m.weight.data, gain=gain)
				elif init_type == 'kaiming':
					nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
				elif init_type == 'orthogonal':
					nn.init.orthogonal_(m.weight.data, gain=gain)

				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.constant_(m.bias.data, 0.0)

			elif classname.find('BatchNorm2d') != -1:
				nn.init.normal_(m.weight.data, 1.0, gain)
				nn.init.constant_(m.bias.data, 0.0)

		self.apply(init_func)

	def forward(self, x):
		'''
		512, 4, 4
		512, 8, 8
		512, 16, 16
		512, 32, 32
		512, 64, 64
		256, 128, 128
		128, 256, 256
		64, 512, 512
		32, 1024, 1024
		'''
		for i in range(len(x)):
			if i == 0:
				hs = x[i]
			else:
				if x[i].shape[2] > 128:
					x[i] = self.interpolate(x[i])
				hs = self.layers[i-1](hs) + x[i]

		hs = self.layers[-1](hs)

		return hs
