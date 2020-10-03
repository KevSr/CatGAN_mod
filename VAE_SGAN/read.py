from __future__ import print_function
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import random

import csv
import math
from math import floor, log2

import train_test as tt
import networks as nw 


root = "/home/kyungsung/CatGAN/Result/"

networks = ['ResNet 50', 'AlexNet', 'VGG 16', 'Original', 'GoogleNet']
nets = ['resnet', 'alex', 'vgg', 'catgan', 'google']
activations = ['LeakyReLU', 'ClippedReLU', 'ReLU', ]
actis = ['leaky', 'clipped', 'relu']
optimizers = ['SGDM', 'RMSProp', 'Adam']
optis = ['sgdm', 'rmsprop', 'adam']
gens = ['vae', 'style']
gen_n = ['VAE','StyleGAN']
header = ['Dataset', 'Generator Type', 'Discriminator Type', 'Activations', 'Optimizers', 'Learning Rate', 'accuracy']

with open('result.csv', 'w') as fp:
  writer = csv.writer(fp, delimiter=',')
  writer.writerow(header)
for a in range(2):
	if a == 0:
		data = 'KDEF'
	elif a == 1:
		data = 'MMI'
	for s in range(len(gen_n)):
		gen = gens[s]
		for j in range(len(networks)):
			n = nets[j]
			for i in range(len(activations)):
				af = actis[i]
				for k in range(len(optimizers)):
					opt = optis[k]

					checkpoints_dir = os.path.join(root, data, gen_n[s], networks[j], activations[i], optimizers[k])
					if gen == 'vae':
						if opt =='sgdm':
							with open(os.path.join(checkpoints_dir, 'accuracy.txt'), 'r') as f:
								print("Result of", data, "-", gen_n[s], "-", networks[j], "-", activations[i], "-", optimizers[k])
								a = 0
								lr = 1e-5
								while a != '':
									a1 = a
									a = f.readline()
								acc = a1

								print(acc)

								with open('result.csv', 'a') as fp:
								    writer = csv.writer(fp, delimiter=',')
								    writer.writerow([data, gen_n[s], networks[j], activations[i], optimizers[k], lr, acc])

									
						else:
							with open(os.path.join(checkpoints_dir, 'accuracy.txt'), 'r') as f:
								print("Result of", data, "-", gen_n[s], "-", networks[j], "-", activations[i], "-", optimizers[k])
								lr1 = f.readline()
								acc1 = f.readline()
								lr2 = f.readline()
								acc2 = f.readline()
								lr_1 = ''
								lr_2 = ''
								for r in range(len(lr1)):
									if r >= 16:
										lr_1 += lr1[r]
								for z in range(len(lr2)):
									if z >= 16:
										lr_2 += lr2[z]

								print(lr_1)
								print(lr_2)
								print(lr1, acc1, lr2, acc2)

								with open('result.csv', 'a') as fp:
								    writer = csv.writer(fp, delimiter=',')
								    writer.writerow([data, gen_n[s], networks[j], activations[i], optimizers[k], lr_1, acc1])
								    writer.writerow([data, gen_n[s], networks[j], activations[i], optimizers[k], lr_2, acc2])
					else:
						with open(os.path.join(checkpoints_dir, 'accuracy.txt'), 'r') as f:
							print("Result of", data, "-", gen_n[s], "-", networks[j], "-", activations[i], "-", optimizers[k])
							a = 0
							lr = 1e-4
							while a != '':
								a1 = a
								a = f.readline()
							acc = a1

							print(acc)

							with open('result.csv', 'a') as fp:
							    writer = csv.writer(fp, delimiter=',')
							    writer.writerow([data, gen_n[s], networks[j], activations[i], optimizers[k], lr, acc])