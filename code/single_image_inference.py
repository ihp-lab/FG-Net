import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' # see issue #152

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import json
import argparse
import pandas as pd

from PIL import Image
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

	return g_all, upsamplers, pspencoder, transform


def inference(interpreter, images, pspencoder, g_all, upsamplers, args):
	interpreter.eval()
	pspencoder.eval()
	g_all.eval()

	with torch.no_grad():
		pred_list = []
		gt_list = []
		f1_list = []

		images = images.cuda()
		# get latent code
		latent_codes = pspencoder(images)
		latent_codes = g_all.style(latent_codes.reshape(latent_codes.shape[0]*latent_codes.shape[1], latent_codes.shape[2])).reshape(latent_codes.shape)
		# get stylegan features
		features = latent_to_image(g_all, upsamplers, latent_codes, upsample=False, args=args)

		heatmaps_pred = interpreter(features)
		heatmaps_pred = torch.clamp(heatmaps_pred, min=-1., max=1.)
		labels_pred = heatmap2au(heatmaps_pred)

		if args['dataset'] == 'DISFA':
			labels_pred = labels_pred[:, :8].detach().cpu()
		elif args['dataset'] == 'BP4D':
			labels_pred = labels_pred[:, :12].detach().cpu()

		return labels_pred


def main(args):
	print('Prepare model')
	g_all, upsamplers, pspencoder, transform = prepare(args)

	num_labels = 32 # bug in model checkpoint
	interpreter = pyramid_interpreter(num_labels, 0.1).cuda()

	checkpoint = torch.load(args['checkpoint_path'])
	interpreter.load_state_dict(checkpoint['interpreter'])
	g_all.load_state_dict(checkpoint['g_all'])
	pspencoder.load_state_dict(checkpoint['pspencoder'])

	print('Prepare data')
	image = Image.open('../test_image.jpg') # replace with your own image path
	image = transform(image)
	image = image.unsqueeze(0) # [1, 3, 256, 256]

	print('Start evaluation')
	if args['dataset'] == 'BP4D':
		aus = [1,2,4,6,7,10,12,14,15,17,23,24]
	elif args['dataset'] == 'DISFA':
		aus = [1,2,4,6,9,12,25,26]
	pred_aus = inference(interpreter, image, pspencoder, g_all, upsamplers, args)
	pred_aus = pred_aus.squeeze(0)

	for i in range(len(aus)):
		print('AU {}: {}'.format(aus[i], pred_aus[i]), end=' ')
	print('')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--exp', type=str)
	args = parser.parse_args()
	opts = json.load(open(args.exp, 'r'))
	print('Opt', opts)

	main(opts)
