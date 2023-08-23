import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' # see issue #152

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import json
import argparse
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score

from data import MyDataset
from model_utils import latent_to_image, prepare_model, load_psp_standalone
from data_utils import heatmap2au
from models.interpreter import pyramid_interpreter


def prepare(args):
	g_all, upsamplers = prepare_model(args)

	pspencoder = load_psp_standalone(args['style_encoder_path'], 'cuda')

	transform = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
	])

	test_list = pd.read_csv(args['test_csv'])

	return test_list, g_all, upsamplers, pspencoder, transform


def val(interpreter, val_loader, pspencoder, g_all, upsamplers, args):
	interpreter.eval()
	pspencoder.eval()
	g_all.eval()

	with torch.no_grad():
		pred_list = []
		gt_list = []
		f1_list = []

		for (images, labels) in tqdm(val_loader):
			images = images.cuda()
			# get latent code
			latent_codes = pspencoder(images)
			latent_codes = g_all.style(latent_codes.reshape(latent_codes.shape[0]*latent_codes.shape[1], latent_codes.shape[2])).reshape(latent_codes.shape)

			features = latent_to_image(g_all, upsamplers, latent_codes, args=args)

			heatmaps_pred = interpreter(features)
			heatmaps_pred = torch.clamp(heatmaps_pred, min=-1., max=1.)
			labels_pred = heatmap2au(heatmaps_pred)

			if 'bp4d' in args['checkpoint_path'] and args['dataset'] == 'Aff-Wild2':
				indices_bp4d = torch.tensor([0, 1, 2, 3, 4, 5, 6, 8, 10, 11]).cuda()
				indices_aw2 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()

				labels = torch.index_select(labels, 1, indices_aw2)
				labels_pred = torch.index_select(labels_pred, 1, indices_bp4d)
			elif 'disfa' in args['checkpoint_path'] and args['dataset'] == 'Aff-Wild2':
				indices_disfa = torch.tensor([0, 1, 2, 3, 5, 6, 7]).cuda()
				indices_aw2 = torch.tensor([0, 1, 2, 3, 6, 10, 11]).cuda()

				labels = torch.index_select(labels, 1, indices_aw2)
				labels_pred = torch.index_select(labels_pred, 1, indices_disfa)
			elif 'disfa' in args['checkpoint_path'] and args['dataset'] == 'GFT':
				indices_disfa = torch.tensor([0, 1, 2, 3, 5]).cuda()
				indices_gft = torch.tensor([0, 1, 2, 3, 6]).cuda()

				labels = torch.index_select(labels, 1, indices_gft)
				labels_pred = torch.index_select(labels_pred, 1, indices_disfa)
			elif 'bp4d' in args['checkpoint_path'] and args['dataset'] == 'DISFA':
				indices_bp4d = torch.tensor([0, 1, 2, 3, 6]).cuda()
				indices_disfa = torch.tensor([0, 1, 2, 3, 5]).cuda()

				labels = torch.index_select(labels, 1, indices_disfa)
				labels_pred = torch.index_select(labels_pred, 1, indices_bp4d)
    
			elif 'disfa' in args['checkpoint_path'] and args['dataset'] == 'BP4D':
				indices_bp4d = torch.tensor([0, 1, 2, 3, 6]).cuda()
				indices_disfa = torch.tensor([0, 1, 2, 3, 5]).cuda()

				labels = torch.index_select(labels, 1, indices_bp4d)
				labels_pred = torch.index_select(labels_pred, 1, indices_disfa)
			else:
				import sys
				sys.exit()

			# indices_bp4d = torch.tensor([0, 1, 2, 3, 6]).cuda()
			# indices_disfa = torch.tensor([0, 1, 2, 3, 5]).cuda()
			# if args['dataset'] == 'BP4D':
			# 	labels = torch.index_select(labels, 1, indices_bp4d)
			# else:
			# 	labels = torch.index_select(labels, 1, indices_disfa)

			# if 'bp4d' in args['checkpoint_path']:
			# 	labels_pred = torch.index_select(labels_pred, 1, indices_bp4d)
			# 	# heatmaps_pred = torch.index_select(heatmaps_pred, 1, indices_bp4d)
			# else:
			# 	labels_pred = torch.index_select(labels_pred, 1, indices_disfa)
			# 	# heatmaps_pred = torch.index_select(heatmaps_pred, 1, indices_disfa)

			# print(heatmaps_pred.shape)
			# heatmaps_pred = heatmaps_pred.detach().cpu().numpy()

			# import matplotlib.pyplot as plt
			# for i in range(5):

			# 	plt.imshow(heatmaps_pred[0][i], cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
			# 	plt.axis('off')
			# 	plt.savefig('heatmap_pred_{}.png'.format(i), bbox_inches='tight', pad_inches=0)

			pred_list.append(labels_pred.detach().cpu())
			gt_list.append(labels.detach().cpu())

		pred_list = torch.cat(pred_list, dim=0).numpy()
		gt_list = torch.cat(gt_list, dim=0).numpy()

		for i in range(gt_list.shape[1]):
			f1_list.append(100.0*f1_score(gt_list[:, i], pred_list[:, i]))

		return sum(f1_list)/len(f1_list), f1_list

def main(args):
	print('Prepare model')
	val_list, g_all, upsamplers, pspencoder, transform = prepare(args)

	if 'bp4d' in args['checkpoint_path']:
		num_labels = 12
	else:
		num_labels = 8

	interpreter = pyramid_interpreter(num_labels, 0.1).cuda()

	checkpoint = torch.load(os.path.join(args['checkpoint_path']))
	interpreter.load_state_dict(checkpoint['interpreter'])
	g_all.load_state_dict(checkpoint['g_all'])
	pspencoder.load_state_dict(checkpoint['pspencoder'])

	print('Prepare data')
	val_data = MyDataset(val_list, transform, 'test', args)
	val_loader = DataLoader(dataset=val_data, batch_size=args['batch_size'],
							shuffle=False, collate_fn=val_data.collate_fn)

	print('Start evaluation')
	if 'bp4d' in args['checkpoint_path'] and args['dataset'] == 'Aff-Wild2':
		aus = [1,2,4,6,7,10,12,15,23,24]
	elif 'disfa' in args['checkpoint_path'] and args['dataset'] == 'Aff-Wild2':
		aus = [1,2,4,6,12,25,26]
	elif 'bp4d' in args['checkpoint_path'] and args['dataset'] == 'GFT':
		aus = [1,2,4,6,7,10,12,14,15,17,23,24]
	elif 'disfa' in args['checkpoint_path'] and args['dataset'] == 'GFT':
		aus = [1,2,4,6,12]
	else:
		aus = [1,2,4,6,12]
	val_f1, val_f1_list = val(interpreter, val_loader, pspencoder, g_all, upsamplers, args)
	print('Val avg F1: {:.2f}'.format(val_f1))
	for i in range(len(aus)):
		print('AU {}: {:.2f}'.format(aus[i], val_f1_list[i]), end=' ')
	print('')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--exp', type=str)
	args = parser.parse_args()
	opts = json.load(open(args.exp, 'r'))
	print('Opt', opts)

	main(opts)
