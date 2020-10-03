from __future__ import print_function
import os
import random
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import random

from functools import partial
from torch.autograd import grad as torch_grad
from PIL import Image

import math
from math import floor, log2
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset

import train_test as tt
import networks as nw 

# Set random seed for reproducibility
manualSeed = 1
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def data_sel(option):
    if option == 'KDEF':
        # Root directory for KDEF dataset
        dataroot = "/home/kyungsung/CatGAN/KDEF"
    elif option == 'MMI':
        # Root directory for MMI_sel dataset
        dataroot = "/home/kyungsung/CatGAN/MMI_selected"
    else:
      print('No dataset selected')
    return dataroot




# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Size of z latent vector (i.e. size of generator input)
nz = 128

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 9e-5

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

K = 5

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


# We can use an image folder dataset the way we have it setup.
# Create the dataset
def datagen(dataroot):
	dataset = dset.ImageFolder(root=dataroot)

	transform1 = transform=transforms.Compose([
	              #  transforms.Resize((224, 224)),
	                transforms.Resize((60,48)),
	                transforms.ToTensor(),
	                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	            ])
	transform2 = transform=transforms.Compose([
	                # transforms.Resize((224, 224)),
	                transforms.Resize((60,48)),
	                transforms.ToTensor(),
	                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	            ])
	# Create the dataloader
	test_len = int(len(dataset)*0.3)
	train_len = len(dataset) - test_len

	train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])

	train_set = MyDataset(train_set, transform1)
	test_set = MyDataset(test_set, transform2)

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
	                                         shuffle=True, num_workers=workers)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_len,
	                                         shuffle=True, num_workers=workers)

	return train_loader, test_loader

def pltfig(G_losses, D_losses, checkpoints_dir, lr = None):
	plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses,label="G")
	plt.plot(D_losses,label="D")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	if lr == None:
		plt.savefig(os.path.join(checkpoints_dir, 'train_graph.png'))
	else:
		plt.savefig(os.path.join(checkpoints_dir, '{}_train_graph.png'.format(lr)))


def giffig(img_list, checkpoints_dir, lr = None):
	writergif = animation.PillowWriter(fps=3) 

	fig = plt.figure(figsize=(8,8))
	plt.axis("off")
	ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
	ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
	if lr == None:
		ani.save(os.path.join(checkpoints_dir, 'gen_result.gif'), writer = writergif)
	else:
		ani.save(os.path.join(checkpoints_dir, '{}_gen_result.gif'.format(lr)), writer = writergif)


root = "/home/kyungsung/CatGAN/Result/"

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
# fixed_noise = torch.rand(batch_size, nz, device=device)#.half()
#.half()
# fixed_noise = torch.rand(batch_size, nz, 1, 1, device=device)


networks = ['ResNet 50', 'AlexNet', 'VGG 16', 'Original', 'GoogleNet']
nets = ['resnet', 'alex', 'vgg', 'catgan', 'google']
activations = ['LeakyReLU', 'ClippedReLU', 'ReLU', ]
actis = ['leaky', 'clipped', 'relu']
optimizers = ['SGDM', 'RMSProp', 'Adam']
optis = ['sgdm', 'rmsprop', 'adam']
gens = ['vae', 'style']
gen_n = ['VAE','StyleGAN']


dataroot = data_sel('MMI')
data = 'MMI'
train_loader, test_loader = datagen(dataroot)
for s in range(len(gen_n)):
	gen = gens[s]
	for j in range(len(networks)):
		n = nets[j]
		for i in range(len(activations)):
			af = actis[i]
			for k in range(len(optimizers)):
				opt = optis[k]

				checkpoints_dir = os.path.join(root, data, gen_n[s], networks[j], activations[i], optimizers[k])
				if os.path.isdir(checkpoints_dir) == True:
					pass
				else:
					os.makedirs(os.path.join(checkpoints_dir, 'images'))
				  # Lists to keep track of progress

				iters = 0
				
				# print(type(G_net).__name__)
				with open(os.path.join(checkpoints_dir, 'results.txt'), 'w') as f:
					print('learning with', gen_n[s], 'and' , networks[j], '-', activations[i], '-', optimizers[k], file=f)
				print('learning with', gen, 'and' , n, '-', af, '-', opt)
				if gen == gens[0]:
					fixed_noise = torch.rand(batch_size, nz, device=device)
					if opt == 'sgdm':
						lr = 1e-5
						img_list = []
						G_losses = []
						D_losses = []
						G_net = nw.gen_sel(gen, nz, device, af)
						D_net = nw.disc_sel(n, K, device, af)
						G_optimizer = nw.opti_sel(G_net, option = opt, lr=lr)
						D_optimizer = nw.opti_sel(D_net, option = opt, lr=lr)
						img_list, G_losses, D_losses = tt.orig(G_net, train_loader, G_optimizer, img_list, G_losses, D_losses, fixed_noise, nz, num_epochs, checkpoints_dir, device, K)
						img_list, G_losses, D_losses = tt.VAEGAN(G_net, D_net, G_optimizer, D_optimizer, img_list, G_losses, D_losses, train_loader, fixed_noise, nz, num_epochs, checkpoints_dir, device, K)
						giffig(img_list, checkpoints_dir, lr)					
						pltfig(G_losses, D_losses, checkpoints_dir, lr)
						tt.test(D_net, test_loader, device, checkpoints_dir, lr)
						plt.close('all')
						del G_net, D_net, G_optimizer, D_optimizer
					else:
						for l in range(2):
							if l == 0:
								lr = 1e-5
								img_list = []
								G_losses = []
								D_losses = []
								with open(os.path.join(checkpoints_dir, 'accuracy.txt'), 'w') as f:
									print('Learning rate = {}'.format(lr), file=f)
								G_net = nw.gen_sel(gen, nz, device, af)
								D_net = nw.disc_sel(n, K, device, af)
								G_optimizer = nw.opti_sel(G_net, option = opt, lr=lr)
								D_optimizer = nw.opti_sel(D_net, option = opt, lr=lr)
								img_list, G_losses, D_losses = tt.orig(G_net, train_loader, G_optimizer, img_list, G_losses, D_losses, fixed_noise, nz, num_epochs, checkpoints_dir, device, K)
								img_list, G_losses, D_losses = tt.VAEGAN(G_net, D_net, G_optimizer, D_optimizer, img_list, G_losses, D_losses, train_loader, fixed_noise, nz, num_epochs, checkpoints_dir, device, K)
								giffig(img_list, checkpoints_dir, lr)			
								pltfig(G_losses, D_losses, checkpoints_dir, lr)
								tt.test(D_net, test_loader, device, checkpoints_dir, lr)
								plt.close('all')
								del G_net, D_net, G_optimizer, D_optimizer

							elif l == 1:
								lr = 1e-4
								img_list = []
								G_losses = []
								D_losses = []
								with open(os.path.join(checkpoints_dir, 'accuracy.txt'), 'a+') as f:
									print('Learning rate = {}'.format(lr), file=f)
								G_net = nw.gen_sel(gen, nz, device, af)
								D_net = nw.disc_sel(n, K, device, af)
								G_optimizer = nw.opti_sel(G_net, option = opt, lr=lr)
								D_optimizer = nw.opti_sel(D_net, option = opt, lr=lr)
								img_list, G_losses, D_losses = tt.orig(G_net, train_loader, G_optimizer, img_list, G_losses, D_losses, fixed_noise, nz, num_epochs, checkpoints_dir, device, K)
								img_list, G_losses, D_losses = tt.VAEGAN(G_net, D_net, G_optimizer, D_optimizer, img_list, G_losses, D_losses, train_loader, fixed_noise, nz, num_epochs, checkpoints_dir, device, K)
								giffig(img_list, checkpoints_dir, lr)					
								pltfig(G_losses, D_losses, checkpoints_dir, lr)
								tt.test(D_net, test_loader, device, checkpoints_dir, lr)
								plt.close('all')
								del G_net, D_net, G_optimizer, D_optimizer

				elif gen == gens[1]:
					fixed_noise = torch.rand(batch_size, nz, nz, 1, device=device)
					lr = 1e-4
					img_list = []
					G_losses = []
					D_losses = []
					G_net = nw.gen_sel(gen, nz, device, af)
					D_net = nw.disc_sel(n, K, device, af)
					G_optimizer = nw.opti_sel(G_net, option = opt, lr=lr)
					D_optimizer = nw.opti_sel(D_net, option = opt, lr=lr)
					GE = nw.gen_sel(gen, nz, device, af)
					for p in GE.parameters():
					    p.requires_grad = False
					GenEMA = tt.GenEMA()
					img_list, G_losses, D_losses = tt.SGAN(G_net, D_net, G_optimizer, D_optimizer, img_list, G_losses, D_losses, train_loader, fixed_noise, nz, num_epochs, checkpoints_dir, device, GenEMA, GE, K)
					giffig(img_list, checkpoints_dir)					
					pltfig(G_losses, D_losses, checkpoints_dir)
					tt.test(D_net, test_loader, device, checkpoints_dir)
					plt.close('all')
					del G_net, D_net, G_optimizer, D_optimizer

# for layer in G_net.modules():
#   if isinstance(layer, nn.BatchNorm2d):
#     layer.float() 

# for layer in D_net.modules():
#   if isinstance(layer, nn.BatchNorm2d):
#     layer.float()   
# Print the model

# Setup Adam optimizers for both G and D


# orig()
# VAEGAN()
# GAN()
