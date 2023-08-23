import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' # see issue #152

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import json
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score

from data_utils import au2heatmap, heatmap2au
from model_utils import prepare_model, latent_to_image, load_psp_standalone
from models.interpreter import pyramid_interpreter, pixel_interpreter


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
		elif args['dataset'] == 'GFT':
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
							self.file_list['au6'],
							self.file_list['au10'],
							self.file_list['au12'],
							self.file_list['au14'],
							self.file_list['au15'],
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
			heatmap = au2heatmap(image_path, label, self.args['dim'][0], self.args['sigma'], self.args) # [x,128,128]
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


def prepare(args):
	g_all, upsamplers = prepare_model(args)

	pspencoder = load_psp_standalone(args['style_encoder_path'], 'cuda')

	transform = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
	])

	train_list = pd.read_csv(args['train_csv'])
	val_list = pd.read_csv(args['val_csv'])
	test_list = pd.read_csv(args['test_csv'])

	return train_list, val_list, test_list, g_all, upsamplers, pspencoder, transform


def val(interpreter, val_loader, pspencoder, g_all, upsamplers, args):
	interpreter.eval()
	pspencoder.eval()
	g_all.eval()

	with torch.no_grad():
		pred_list = []
		gt_list = []
		f1_list = []

		for i, (images, labels) in enumerate(tqdm(val_loader)):
			# get latent code
			latent_codes = pspencoder(images)
			latent_codes = g_all.style(latent_codes.reshape(latent_codes.shape[0]*latent_codes.shape[1], latent_codes.shape[2])).reshape(latent_codes.shape)
			# get stylegan features
			# features = latent_to_image(g_all, upsamplers, latent_codes, upsample=False, args=args)
			features = latent_to_image(g_all, upsamplers, latent_codes, upsample=True, args=args)

			heatmaps_pred = interpreter(features)
			heatmaps_pred = torch.clamp(heatmaps_pred, min=-1., max=1.)
			labels_pred = heatmap2au(heatmaps_pred)

			pred_list.append(labels_pred.detach().cpu())
			gt_list.append(labels.detach().cpu())

		pred_list = torch.cat(pred_list, dim=0).numpy()
		gt_list = torch.cat(gt_list, dim=0).numpy()

		for i in range(args['num_labels']):
			f1_list.append(100.0*f1_score(gt_list[:, i], pred_list[:, i]))

		return sum(f1_list)/len(f1_list), f1_list


def main(args):
	print('Prepare model')
	train_list, val_list, test_list, g_all, upsamplers, pspencoder, transform = prepare(args)

	print('Prepare data')
	train_data = MyDataset(train_list, transform, 'train', args)
	val_data = MyDataset(val_list, transform, 'val', args)
	val_loader = DataLoader(dataset=val_data, batch_size=args['batch_size'],
							shuffle=False, collate_fn=val_data.collate_fn)
	test_data = MyDataset(test_list, transform, 'val', args)
	test_loader = DataLoader(dataset=test_data, batch_size=args['batch_size'],
							shuffle=False, collate_fn=test_data.collate_fn)

	print('Start training')
	# interpreter = pyramid_interpreter(args['num_labels'], args['dropout']).cuda()
	interpreter = pixel_interpreter(args['num_labels'], args['dim'][2]).cuda()
	interpreter.init_weights()
	criterion = nn.MSELoss()
	optimizer = optim.AdamW(list(interpreter.parameters())
							+list(g_all.parameters())
							+list(pspencoder.parameters()),
							lr=args['learning_rate'], weight_decay=args['weight_decay'])

	total_loss = 0.
	total_sample = 0
	best_f1 = 0.
	best_f1_list = []

	if args['dataset'] == 'BP4D':
		aus = [1,2,4,6,7,10,12,14,15,17,23,24]
	elif args['dataset'] == 'GFT':
		aus = [1,2,4,6,10,12,14,15,23,24]
		weights = [2.47562240896833, 0.697006676039326, 2.43855426594634, 
			0.316453607031205, 0.195463980407708, 0.362326506270633,
			0.305554575001318, 3.06705439690589, 0.846217428243707,
			0.295121312829718, 0.358610921462698, 0.642013920893121]
	else:
		aus = [1,2,4,6,9,12,25,26]

	for epoch in range(args['num_epochs']):
		interpreter.train()
		g_all.train()
		pspencoder.train()
		train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, collate_fn=train_data.collate_fn)
		args['interval'] = min(args['interval'], len(train_loader))
		for i, (images, labels, heatmaps) in enumerate(tqdm(train_loader, total=args['interval'])):
			if i >= args['interval']:
				break
			batch_size = images.shape[0]
			# get latent code
			latent_codes = pspencoder(images)
			latent_codes = g_all.style(latent_codes.reshape(latent_codes.shape[0]*latent_codes.shape[1], latent_codes.shape[2])).reshape(latent_codes.shape)
			# get stylegan features
			# features = latent_to_image(g_all, upsamplers, latent_codes, upsample=False, args=args)
			features = latent_to_image(g_all, upsamplers, latent_codes, upsample=True, args=args)
			heatmaps_pred = interpreter(features)

			loss = 0.
			for i in range(len(aus)):
				loss += weights[i]*criterion(heatmaps_pred[:,i,:,:], heatmaps[:,i,:,:])
			loss /= len(aus)

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(list(interpreter.parameters())
											+list(g_all.parameters())
											+list(pspencoder.parameters()), 0.1)
			optimizer.step()

			total_loss += loss.item()*batch_size*args['dim'][0]*args['dim'][1]
			total_sample += batch_size*args['dim'][0]*args['dim'][1]

		avg_loss = total_loss / total_sample
		print('** Epoch {}/{} loss {:.6f} **'.format(epoch+1, args['num_epochs'], avg_loss))

		val_f1, val_f1_list = val(interpreter, val_loader, pspencoder, g_all, upsamplers, args)
		print('Val avg F1: {:.2f}'.format(val_f1))
		for i in range(args['num_labels']):
			print('AU {}: {:.2f}'.format(aus[i], val_f1_list[i]), end=' ')
		print('')

		if best_f1 < val_f1:
			best_f1 = val_f1
			best_f1_list = val_f1_list
			model_path = os.path.join(args['exp_dir'], 'model.pth')
			print('save to:', model_path)
			torch.save({'interpreter': interpreter.state_dict(),
						'g_all': g_all.state_dict(),
						'pspencoder': pspencoder.state_dict()}, model_path)

	# test_f1, test_f1_list = val(interpreter, test_loader, pspencoder, g_all, upsamplers, args)	
	print('Test avg F1: {:.2f}'.format(best_f1))
	for i in range(args['num_labels']):
		print('AU {}: {:.2f}'.format(aus[i], best_f1_list[i]), end=' ')
	print('')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--exp', type=str)
	args = parser.parse_args()
	opts = json.load(open(args.exp, 'r'))
	print('Opt', opts)

	os.makedirs(opts['exp_dir'], exist_ok=True)
	os.system('cp %s %s' % (args.exp, opts['exp_dir']))

	torch.manual_seed(opts['seed'])

	main(opts)
 