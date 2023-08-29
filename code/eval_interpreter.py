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

from data_utils import heatmap2au, MyDataset
from model_utils import latent_to_image, prepare_model, load_psp_standalone
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
			# get stylegan features
			features = latent_to_image(g_all, upsamplers, latent_codes, upsample=False, args=args)

			heatmaps_pred = interpreter(features)
			heatmaps_pred = torch.clamp(heatmaps_pred, min=-1., max=1.)
			labels_pred = heatmap2au(heatmaps_pred)

			if 'bp4d' in args['checkpoint_path'] and args['dataset'] == 'DISFA':
				indices_bp4d = torch.tensor([0, 1, 2, 3, 6]).cuda()
				indices_disfa = torch.tensor([0, 1, 2, 3, 5]).cuda()

				labels = torch.index_select(labels, 1, indices_disfa)
				labels_pred = torch.index_select(labels_pred, 1, indices_bp4d)
			elif 'disfa' in args['checkpoint_path'] and args['dataset'] == 'BP4D':
				indices_bp4d = torch.tensor([0, 1, 2, 3, 6]).cuda()
				indices_disfa = torch.tensor([0, 1, 2, 3, 5]).cuda()

				labels = torch.index_select(labels, 1, indices_bp4d)
				labels_pred = torch.index_select(labels_pred, 1, indices_disfa)

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
	num_labels = 32 # bug in model checkpoint

	interpreter = pyramid_interpreter(num_labels, 0.1).cuda()

	checkpoint = torch.load(args['checkpoint_path'])
	interpreter.load_state_dict(checkpoint['interpreter'])
	g_all.load_state_dict(checkpoint['g_all'])
	pspencoder.load_state_dict(checkpoint['pspencoder'])

	print('Prepare data')
	val_data = MyDataset(val_list, transform, 'test', args)
	val_loader = DataLoader(dataset=val_data, batch_size=args['batch_size'],
							shuffle=False, collate_fn=val_data.collate_fn)

	print('Start evaluation')
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
