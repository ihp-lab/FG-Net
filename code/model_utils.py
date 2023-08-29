import math
import argparse
from PIL import Image
from math import cos, pi

import torch
import torch.nn as nn

from models.encoders.psp_encoders import GradualStyleEncoder
from models.stylegan2_pytorch.stylegan2_pytorch import Generator as Stylegan2Generator


def load_psp_standalone(checkpoint_path, device='cuda'):
	ckpt = torch.load(checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	if 'output_size' not in opts:
		opts['output_size'] = 1024
	opts['n_styles'] = int(math.log(opts['output_size'], 2)) * 2 - 2
	opts = argparse.Namespace(**opts)
	psp = GradualStyleEncoder(50, 'ir_se', opts)
	psp_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
	psp.load_state_dict(psp_dict)
	psp = psp.to(device)
	latent_avg = ckpt['latent_avg'].to(device)

	def add_latent_avg(model, inputs, outputs):
		return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)

	psp.register_forward_hook(add_latent_avg)
	return psp


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


downsample = Interpolate(256, 'bilinear')
def latent_to_image(g_all, upsamplers, style_latents, upsample=True, args=None):
	images_recon, affine_layers = g_all.g_synthesis(style_latents, noise=None)
	images_recon = downsample(images_recon)

	feature_layer_list = [1,3,5,7,9,11,13,15,17]

	affine_layers_upsamples = []
	if not upsample:
		for i in feature_layer_list:
			affine_layers_upsamples.append(affine_layers[i])
	else:
		for i in feature_layer_list:
			affine_layers_upsamples.append(upsamplers[i](affine_layers[i]))
		affine_layers_upsamples = torch.cat(affine_layers_upsamples, dim=1)

	return affine_layers_upsamples


def prepare_model(args):
	res = 1024
	out_res = args['dim']

	g_all = Stylegan2Generator(res, 512, 8, channel_multiplier=2, randomize_noise=False).cuda()
	checkpoint = torch.load(args['stylegan_checkpoint'])
	g_all.load_state_dict(checkpoint['g_ema'], strict=True)
	g_all.make_mean_latent(5000)

	mode = 'bilinear'

	bi_upsamplers = [
						nn.Upsample(scale_factor=out_res / 4, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 4, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 8, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 8, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 16, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 16, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 32, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 32, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 64, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 64, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 128, mode=mode, align_corners=True),
						nn.Upsample(scale_factor=out_res / 128, mode=mode, align_corners=True),
						Interpolate(out_res, mode),
						Interpolate(out_res, mode),
						Interpolate(out_res, mode),
						Interpolate(out_res, mode),
						Interpolate(out_res, mode),
						Interpolate(out_res, mode)
					]

	return g_all, bi_upsamplers
