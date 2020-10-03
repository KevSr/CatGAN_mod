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

from functools import partial
from torch.autograd import grad as torch_grad
from PIL import Image

import math
from math import floor, log2
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset

import networks as net

class MarginalHLoss(nn.Module):
  def __init__(self):
    super(MarginalHLoss, self).__init__()
  def forward(self, x):# NxK
    x = x.mean(axis=0)
    x = -torch.sum(x*torch.log(x+1e-6))
    return x

class JointHLoss(nn.Module):
  def __init__(self):
    super(JointHLoss, self).__init__()
  def forward(self, x, batch_size):
    x = -x*torch.log(x+1e-6)
    x = (1.0/batch_size) * torch.sum(x)
    return x
    #marginalized entropy

class CrossHLoss(nn.Module):
  def __init__(self):
    super(CrossHLoss, self).__init__()
  def forward(self, x, y, batch_size = None):
    x = -torch.sum(x*torch.log(y+1e-6))
    # x = (1.0/batch_size) * x
    return x
    #marginalized entropy

class KLLoss(nn.Module):
  def __init__(self):
    super(KLLoss, self).__init__()
  def forward(self, x, y, batch_size):
    x = 1 + y - x**2 - y.exp()
    x = -0.5 * torch.sum(x)
    x = (1.0/batch_size) * x
    return x
    #marginalized entropy


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new       

class PL_length(nn.Module):
  def __init__(self):
    super(PL_length, self).__init__()

  def forward(self, styles, images, device):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).to(device=device) / math.sqrt(num_pixels)
    outputs = torch.sum(images * pl_noise)
    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape).to(device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    return (pl_grads ** 2).sum().mean().sqrt()  

class PathPenalty(nn.Module):
  def __init__(self):
    super(PathPenalty, self).__init__()
  def forward(self, pl_lengths, plmean): 
    if not is_empty(plmean):
        pl_loss = ((pl_lengths - plmean) ** 2).mean()
    else:
      pl_loss = None
    return pl_loss

class GenEMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.ema_updater = EMA(0.995)
    def EMA(self, GE, G_net):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(GE, G_net)


def generate_and_save_images(epoch, model, device, nz, checkpoints_dir):
    img_list=[]
    # noise = torch.rand(4, nz, 1, 1).to(device=device)
    # noise = torch.rand(4, 3, 60, 48).to(device=device)
    name = type(model).__name__
    if name == 'VAE':
        noise = torch.randn(4, nz).to(device=device)
        predictions = model.decode(noise).float().detach().cpu()
    else:
        get_latents_fn = mixed_list if random.random() < 0.9 else noise_list
        style = get_latents_fn(4, int(log2(60) - 1), 512, device)
        noise = torch.rand(4, nz, nz, 1).to(device=device)
        _, predictions = model(style, noise)
        predictions = predictions.float().detach().cpu()

    img_list.append(vutils.make_grid(predictions, padding=2, normalize=True))

    fig = plt.figure(figsize=(8,8))

    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)))] for i in img_list]
    name = type(model).__name__
    plt.savefig(os.path.join(checkpoints_dir, 'images', '{}_epoch_{:04d}.png'.format(name, epoch)))
    plt.close()




# D_net = torchvision.models.vgg16(num_classes=6).to(device)
# D_optimizer = opti_sel(D_net, option = 'adam')

jointH = JointHLoss()
marginalH = MarginalHLoss()
crossH = CrossHLoss()
KL = KLLoss()
crossPY = nn.CrossEntropyLoss()
bcnloss = nn.BCELoss()
sigmoid = nn.Sigmoid()
pathlength = PathPenalty()
pll = PL_length()
def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).to(device=device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)
    
def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return t is None

def GAN(G_net, D_net, G_optimizer, D_optimizer, img_list, G_losses, D_losses, train_loader, fixed_noise, nz, num_epochs, checkpoints_dir, device, K):
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data, labels) in enumerate(train_loader) :

            b_size=data.shape[0] 

            data = data.to(device=device).half()
            label = F.one_hot(labels, num_classes=K).to(device=device)
            #Train D

            D_net.zero_grad()
            y_real = D_net(data)
            
            cross_entropy = crossH(label, y_real)
            joint_entropy_real = jointH(y_real, b_size)#minimize uncertainty
            print(label[0], y_real[0], cross_entropy.item()) 
            marginal_entropy_real = marginalH(y_real)#maximize uncertainty

            z = torch.rand(b_size, nz, nz).to(device=device).half() #uniform distribution sampling
            # z = torch.rand(b_size,nz,1,1).to(device=device)#uniform distribution sampling
            fake_images = G_net(z)


            y_fake = D_net(fake_images.detach())
            joint_entropy_fake = jointH(y_fake, b_size)#maximize uncertainty

            loss_D = - (- cross_entropy  + marginal_entropy_real+ joint_entropy_fake -joint_entropy_real)
           
            loss_D.backward(retain_graph=True)
            D_optimizer.step()

            #Train G
            del y_fake, data
            G_net.zero_grad()
            y_fake = D_net(fake_images)
            marginal_entropy_fake = marginalH(y_fake)#maximize uncertainty
            joint_entropy_fake = jointH(y_fake, b_size)#maximize uncertainty
  
            loss_G = joint_entropy_fake - marginal_entropy_fake

            loss_G.backward(retain_graph=True)
            G_optimizer.step()


            if (i+1)%20 == 0 or i ==0:
                with open(os.path.join(checkpoints_dir, 'results.txt'), 'a+') as f:
                    print('Epoch [{}/{}], Step [{}/{}] \n D_loss: {:.4f}, H_x[p(y|D)] : {:.4f}, E[H[p(y|x,D)]] : {:.4f}, \n \
                        E[H[p(y|G(z),D)]]:{:.4f}\n G_loss: {:.4f}, H_G[p(y|D)] : {:.4f}, E[H[p(y|G(z),D)]]: {:.4f}, \n \
                        E[CE[y,p(y|x,D)]: {:.4f}'#, KL: {:.4f}' 
                        .format(epoch, num_epochs, i+1, len(train_loader), loss_D.item(), marginal_entropy_real.item(), joint_entropy_real.item()\
                            ,joint_entropy_fake.item(),loss_G.item(),marginal_entropy_fake.item(), joint_entropy_fake.item(), cross_entropy.item()), file = f)#\
                              # ,reparamatize_loss.item()))
                    print(y_real[0], y_fake[0], label[0], file = f)#, loss_D.item(), loss_G.item())
                print('Epoch [{}/{}], Step [{}/{}] \n D_loss: {:.4f}, H_x[p(y|D)] : {:.4f}, E[H[p(y|x,D)]] : {:.4f}, \n \
                    E[H[p(y|G(z),D)]]:{:.4f}\n G_loss: {:.4f}, H_G[p(y|D)] : {:.4f}, E[H[p(y|G(z),D)]]: {:.4f}, \n \
                    E[CE[y,p(y|x,D)]: {:.4f}'#, KL: {:.4f}' 
                    .format(epoch, num_epochs, i+1, len(train_loader), loss_D.item(), marginal_entropy_real.item(), joint_entropy_real.item()\
                        ,joint_entropy_fake.item(),loss_G.item(),marginal_entropy_fake.item(), joint_entropy_fake.item(), cross_entropy.item()))#\
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())
                with torch.no_grad():

                    fake = G_net(fixed_noise).float().detach().cpu()
                    img_list.append(vutils.make_grid(torch.reshape(fake,(b_size,3,60,48))[:64], padding=2, normalize=True))
                    del fake

            del y_real, y_fake, cross_entropy, joint_entropy_fake, joint_entropy_real, marginal_entropy_real
            del label, marginal_entropy_fake, fake_images, loss_G, loss_D
        # generate samples every 2 epochs for surveillance
            

        if epoch % 1 == 0:
            generate_and_save_images(epoch, G_net, device, nz, checkpoints_dir)


        # do checkpointing every 20 epochs
        # if epoch == (num_epochs - 1):
        #     torch.save(G_net.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join(checkpoints_dir), 40))
        #     torch.save(D_net.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join(checkpoints_dir), 40))



def VAEGAN(G_net, D_net, G_optimizer, D_optimizer, img_list, G_losses, D_losses, train_loader, fixed_noise, nz, num_epochs, checkpoints_dir, device, K):
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # print(torch.cuda.memory_summary())
        # For each batch in the dataloader
        for i, (data, labels) in enumerate(train_loader) :

            b_size=data.shape[0] 

            data = data.to(device=device)#.half()
            label = F.one_hot(labels, num_classes=K).to(device=device).float()
            #Train D

            D_net.zero_grad()
            y_real = D_net(data)
            
            cross_entropy = crossH(label, y_real)
            joint_entropy_real = jointH(y_real, b_size)#minimize uncertainty
            marginal_entropy_real = marginalH(y_real)#maximize uncertainty

            # z = torch.rand(b_size, 3, 60, 48).to(device=device)#.half() #uniform distribution sampling
            z = torch.randn(b_size,nz).to(device=device)#uniform distribution sampling
            fake_images = G_net.decode(z)

            y_fake = D_net(fake_images.detach())
            joint_entropy_fake = jointH(y_fake, b_size)#maximize uncertainty

            loss_D = - (- cross_entropy  + marginal_entropy_real+ joint_entropy_fake -joint_entropy_real)
           
            loss_D.backward(retain_graph=True)
            D_optimizer.step()

            #Train G
            del y_fake, data
            G_net.zero_grad()
            y_fake = D_net(fake_images)

            marginal_entropy_fake = marginalH(y_fake)#maximize uncertainty

            joint_entropy_fake = jointH(y_fake, b_size)#maximize uncertainty

            # reparamatize_loss = KL(m, logvar)
            loss_G = joint_entropy_fake - marginal_entropy_fake #+ reparamatize_loss

            loss_G.backward(retain_graph=True)
            G_optimizer.step()


            if (i+1)%20 == 0 or i ==0:
                with open(os.path.join(checkpoints_dir, 'results.txt'), 'a+') as f:
                    print('Epoch [{}/{}], Step [{}/{}] \n D_loss: {:.4f}, H_x[p(y|D)] : {:.4f}, E[H[p(y|x,D)]] : {:.4f}, \n \
                        E[H[p(y|G(z),D)]]:{:.4f}\n G_loss: {:.4f}, H_G[p(y|D)] : {:.4f}, E[H[p(y|G(z),D)]]: {:.4f}, \n \
                        E[CE[y,p(y|x,D)]: {:.4f}'#, KL: {:.4f}' 
                        .format(epoch, num_epochs, i+1, len(train_loader), loss_D.item(), marginal_entropy_real.item(), joint_entropy_real.item()\
                            ,joint_entropy_fake.item(),loss_G.item(),marginal_entropy_fake.item(), joint_entropy_fake.item(), cross_entropy.item()), file = f)#\
                              # ,reparamatize_loss.item()))
                    print(y_real[0], y_fake[0], label[0], file = f)#, loss_D.item(), loss_G.item())
                print('Epoch [{}/{}], Step [{}/{}] \n D_loss: {:.4f}, H_x[p(y|D)] : {:.4f}, E[H[p(y|x,D)]] : {:.4f}, \n \
                    E[H[p(y|G(z),D)]]:{:.4f}\n G_loss: {:.4f}, H_G[p(y|D)] : {:.4f}, E[H[p(y|G(z),D)]]: {:.4f}, \n \
                    E[CE[y,p(y|x,D)]: {:.4f}'#, KL: {:.4f}' 
                    .format(epoch, num_epochs, i+1, len(train_loader), loss_D.item(), marginal_entropy_real.item(), joint_entropy_real.item()\
                        ,joint_entropy_fake.item(),loss_G.item(),marginal_entropy_fake.item(), joint_entropy_fake.item(), cross_entropy.item()))#\
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())

                with torch.no_grad():
                    fake = G_net.decode(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(torch.reshape(fake,(b_size,3,60,48))[:64], padding=2, normalize=True))
                  
            # del y_real, y_fake, cross_entropy, joint_entropy_fake, joint_entropy_real, marginal_entropy_real
            # del label, marginal_entropy_fake, fake_images, loss_G, loss_D
        # generate samples every 2 epochs for surveillance


        if epoch % 5  == 0:
            generate_and_save_images(epoch, G_net, device, nz, checkpoints_dir)

    return img_list, G_losses, D_losses
        # do checkpointing every 20 epochs
        # if epoch == (num_epochs - 1):
        #     torch.save(G_net.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join(checkpoints_dir), 40))
        #     torch.save(D_net.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join(checkpoints_dir), 40))

def orig(G_net, train_loader, G_optimizer, img_list, G_losses, D_losses, fixed_noise, nz, num_epochs, checkpoints_dir, device, K):

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data, labels) in enumerate(train_loader) :

            b_size=data.shape[0] 

            data = data.to(device=device)
            #label = labels.to(device=device)
            label = F.one_hot(labels, num_classes=K).to(device=device)
            #Train D

            G_net.zero_grad()

            mu, logvar, output = G_net(data, b_size)

            reparam = KL(mu, logvar, b_size)

            target = sigmoid(data)
            BCE = bcnloss(output, target)

            loss_G = BCE + reparam 

            loss_G.backward(retain_graph=True)
            
            G_optimizer.step()      
            # D_net.zero_grad()
            # y_real = D_net(data)

            # cross_entropy = crossH(label, y_real)

            # loss_D = cross_entropy 
            # loss_D.backward(retain_graph=True)
            # D_optimizer.step()
            

            if (i+1)%20 == 0 or i ==0:
                with open(os.path.join(checkpoints_dir, 'results.txt'), 'a+') as f:
                    print('Epoch [{}/{}], Step [{}/{}] \n D_loss: {:.4f}' 
                        .format(epoch, num_epochs, i+1, len(train_loader), loss_G.item()), file = f)
                    print(BCE.item(), reparam.item(), file = f)
                # G_losses.append(loss_G.item())
                G_losses.append(loss_G.item())
                D_losses.append(0)
                print('Epoch [{}/{}], Step [{}/{}] \n D_loss: {:.4f}' 
                    .format(epoch, num_epochs, i+1, len(train_loader), loss_G.item()))
                with torch.no_grad():
                    fake = G_net.decode(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(torch.reshape(fake,(b_size,3,60,48))[:64], padding=2, normalize=True))
        if epoch % 5 == 0:
            generate_and_save_images(epoch, G_net, device, nz, checkpoints_dir)


    return img_list, G_losses, D_losses
        # do checkpointing every 20 epochs
        # if epoch == (num_epochs - 1):
        #     torch.save(G_net.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join(checkpoints_dir), 40))
        #     torch.save(D_net.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join(checkpoints_dir), 40))

def SGAN(G_net, D_net, G_optimizer, D_optimizer, img_list, G_losses, D_losses, train_loader, fixed_noise, nz, num_epochs, checkpoints_dir, device, GenEMA, GE_net, K):
    print("Starting Training Loop...")
    plmean = None
    pl_length_ma = EMA(0.99)
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data, labels) in enumerate(train_loader) :

            b_size=data.shape[0]      
            data = data.to(device=device)
            label = F.one_hot(labels, num_classes= K).float().to(device=device)
            #Train D

            D_net.zero_grad()
            y_real = D_net(data).float()
            
            cross_entropy = crossH(label, y_real)
            joint_entropy_real = jointH(y_real, b_size)#minimize uncertainty
            marginal_entropy_real = marginalH(y_real)#maximize uncertainty


            get_latents_fn = mixed_list if random.random()  < 0.9 else noise_list
            style = get_latents_fn(b_size, int(log2(60) - 1), 512, device)

            z = torch.rand(b_size, nz, nz, 1).to(device=device)
            styles, fake_images = G_net(style, z)

            y_fake = D_net(fake_images.detach())
            joint_entropy_fake = jointH(y_fake, b_size)#maximize uncertainty

            loss_D = - (- cross_entropy  + marginal_entropy_real+ joint_entropy_fake -joint_entropy_real)
           
            loss_D.backward(retain_graph=True)
            D_optimizer.step()

            #Train G
            del y_fake, data
            G_net.zero_grad()
            y_fake = D_net(fake_images)
            marginal_entropy_fake = marginalH(y_fake)#maximize uncertainty
            joint_entropy_fake = jointH(y_fake, b_size)#maximize uncertainty
            pl_lengths = pll(styles, fake_images, device)
            avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())
            path_length = pathlength(pl_lengths, plmean)
            
            if type(path_length) == type(None):
              loss_G = joint_entropy_fake - marginal_entropy_fake
            else:
              loss_G = joint_entropy_fake - marginal_entropy_fake + path_length

            loss_G.backward(retain_graph=True)
            G_optimizer.step()

            if not np.isnan(avg_pl_length):
                plmean = pl_length_ma.update_average(plmean, avg_pl_length)

            if epoch %10 == 0 and i > 5:
              GenEMA.EMA(GE_net, G_net)


            if (i+1)%20 == 0 or i ==0:
                with open(os.path.join(checkpoints_dir, 'results.txt'), 'a+') as f:
                    print('Epoch [{}/{}], Step [{}/{}] \n D_loss: {:.4f}, H_x[p(y|D)] : {:.4f}, E[H[p(y|x,D)]] : {:.4f}, \n \
                            E[H[p(y|G(z),D)]]:{:.4f}\n G_loss: {:.4f}, H_G[p(y|D)] : {:.4f}, E[H[p(y|G(z),D)]]: {:.4f}, \n \
                            E[CE[y,p(y|x,D)]: {:.4f}'#, KL: {:.4f}' 
                        .format(epoch, num_epochs, i+1, len(train_loader), loss_D.item(), marginal_entropy_real.item(), joint_entropy_real.item()\
                            ,joint_entropy_fake.item(),loss_G.item(),marginal_entropy_fake.item(), joint_entropy_fake.item(), cross_entropy.item()), file = f)#\
                    if type(path_length) == type(None):
                        pass
                    else:
                        print('Path Length: {:.4f}'.format(path_length.item()), file = f)
                print('Epoch [{}/{}], Step [{}/{}] \n D_loss: {:.4f}, H_x[p(y|D)] : {:.4f}, E[H[p(y|x,D)]] : {:.4f}, \
                        E[H[p(y|G(z),D)]]:{:.4f}\n G_loss: {:.4f}, H_G[p(y|D)] : {:.4f}, E[H[p(y|G(z),D)]]: {:.4f}, E[CE[y,p(y|x,D)]: {:.4f}' 
                    .format(epoch, num_epochs, i+1, len(train_loader), loss_D.item(), marginal_entropy_real.item(), joint_entropy_real.item()\
                        ,joint_entropy_fake.item(),loss_G.item(),marginal_entropy_fake.item(),joint_entropy_fake.item(), cross_entropy.item()))
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())
                if type(path_length) == type(None):
                    pass
                else:
                    print('Path Length: {:.4f}'.format(path_length.item()))
                print(y_real[0], y_fake[0], label[0])#, loss_D.item(), loss_G.item())
                with torch.no_grad():
                    _, fake = G_net(style, fixed_noise)
                    fake = fake.detach().cpu()
                    img_list.append(vutils.make_grid(torch.reshape(fake,(b_size,3,60,48))[:64], padding=2, normalize=True))
            del y_real, y_fake, cross_entropy, joint_entropy_fake, joint_entropy_real, marginal_entropy_real
            del label, marginal_entropy_fake, fake_images, loss_G, loss_D, pl_lengths, avg_pl_length, path_length
        # generate samples every 2 epochs for surveillance


        if epoch % 1 == 0:
            generate_and_save_images(epoch, G_net, device, nz, checkpoints_dir)

    return img_list, G_losses, D_losses
        # do checkpointing every 20 epochs
        # if epoch == (num_epochs - 1):
        #     torch.save(G_net.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join(checkpoints_dir), 40))
        #     torch.save(D_net.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join(checkpoints_dir), 40))



def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i].cpu().detach(), true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    predicted_label = np.argmax(predictions_array).numpy()

    if predicted_label == true_label.numpy():
        color = 'blue'
    else:
        color = 'red'

    plt.imshow(np.transpose(img,(1,2,0)))

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*predictions_array.max().numpy(),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i].cpu().detach(), true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def result(predictions, test_la, A, class_names, checkpoints_dir, lr = None):
    num_rows = 8
    num_cols = 10
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_la, A, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_la)
    if lr == None:
        plt.savefig(os.path.join(checkpoints_dir, 'test_result.png'))
    else:
        plt.savefig(os.path.join(checkpoints_dir, '{}_test_result.png'.format(lr)))
    plt.close()

def accuracy(predictions, test_la, checkpoints_dir):
    correctness = 0
    for i in range(predictions.shape[0]):
        if np.argmax(predictions[i].cpu().detach()) == test_la[i]:
            correctness += 1
        else:
            correctness += 0


    with open(os.path.join(checkpoints_dir, 'accuracy.txt'), 'a+') as f:
        print(correctness/predictions.shape[0], file=f)  # Python 3.x


def test(D_net, test_loader, device, checkpoints_dir, lr = None):
  test = next(iter(test_loader))
  test_im, test_la = test
  del test
  with torch.no_grad():
      # predictions = D_net(test_im.to(device).half()).float()
      predictions = D_net(test_im.to(device)).float()

  A = test_im.cpu()
  A -= A.min()
  A /= A.max()

  class_names = ['Angry',
  'Frown',
  'Sad',
  'Smile',
  'Surprise']

  accuracy(predictions, test_la, checkpoints_dir)
  result(predictions, test_la, A, class_names, checkpoints_dir, lr)








