import logging
import sys

import numpy as np
import PIL.Image
import torch

import Bialecki_Viehmann_StyleGAN as bvsgen


def latent_to_w(latents, psi, cpuonly):
    device = 'cuda:0' if not cpuonly and torch.cuda.is_available() else 'cpu'
    latents = torch.tensor([latents], device=device)
    with torch.no_grad():
        return bvsgen.x2w(latents, psi=psi).tolist()


def w_to_image(w, downsample, cpuonly, seed=27182818):
    device = 'cuda:0' if not cpuonly and torch.cuda.is_available() else 'cpu'
    w = torch.tensor(w, device=device)
    torch.manual_seed(seed)
    with torch.no_grad():
        images = bvsgen.w2img(w, downsample=downsample).cpu()
    return [PIL.Image.fromarray(np.uint8(images[i,:,:,:].permute(1, 2, 0).detach().numpy()*255), 'RGB') for i in range(images.size()[0])]


def latent_to_image(latents, seed=27182818, psi=0.7, cpuonly=False, downsample=1):
    device = 'cuda:0' if not cpuonly and torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    latents = torch.tensor(latents, device=device)
    with torch.no_grad():
        images = bvsgen.generate(latents, psi, downsample).cpu()

    return [PIL.Image.fromarray(np.uint8(images[i,:,:,:].permute(1, 2, 0).detach().numpy()*255), 'RGB') for i in range(images.size()[0])]
