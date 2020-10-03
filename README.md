# Modification on CatGAN

**Ongoing Project**

In this project, I applied various modifications on Categorical Generative Adversarial Network (CatGAN) [[1]](#1), to investigate the network, and to improve it.

For the input dataset, two Facial Expression Recognition (FER) datasets are used.
* MMI [[2]](#2) 
* KDEF [[3]](#3) 

All codes are made with python, and neural networks with *pytorch*.


## Motivation
As input datasets has different distributions, the performance of networks on each dataset is different. This is also applied with CatGAN.  
Therefore by changing Discriminator and/or Generator of the CatGAN, it might be possible to obtain a better performing network for some datsets.

## Discriminator Options
Discriminator was the first to consider, because there are various state-of-the-art classification networks available. Among these, I chose following networks.
1. AlexNet [[4]](#4)
2. VGG (VGG - 16) [[5]](#5)
3. ResNet (ResNet - 50) [[6]](#6)
4. GoogLeNet [[7]](#7)

## Hyperparameter Options
Due to the nature of the hyperparameters, they do effect the performance and convergence rate of the networks. In this project, I tried to give options to two hyperparameters;
* Activation Function
* Optimisation Algorithm

Following Hyperparameters are chosen.  
For activation function,
1. ReLU [[8]](#8)
2. Leaky ReLU [[9]](#9)
3. Clipped ReLU [[10]](#10)

For optimiser,
1. SGDM [[11]](#11)
2. RMSProp [[12]](#12)
3. Adam [[13]](#13)

These hyperparameters are chosen due to their popularity, and did not consider their performances.

As result, it has been discovered that two combinations of network resulted with good performances
* Original CatGAN
* VGG-16 as Discriminator

---

## Generator Options

After testing with previous settings, it has been discoverd that the generator does not create images with features visible in the input dataset (faces). 

![Sample generated image with CatGAN generator](https://github.com/KevSr/CatGAN_mod/blob/master/catgan_results/mmi_leaky_rms.png)

Therefore, other generator models are used to check if this is due to the problem of CatGAN's generator.

Following models are used
1. Variational Autoencoder (VAE) [[14]](#14)
2. Generator of StyleGAN 2 [[15]](#15)

Generated Image with VAE
![Sample generated image with VAE](https://github.com/KevSr/CatGAN_mod/blob/master/catgan_results/VAE_epoch_0045.png)

Generated Image with StyleGAN 2
![Sample generated image with StyleGAN2](https://github.com/KevSr/CatGAN_mod/blob/master/catgan_results/StyleGAN_epoch_0049.png)

---

## File description
* CatGAN.ipynb  
All codes required to run the experiments.
* rosenbrock.py  
Code for visualisation of rosenbrock's function with two optimisers.
* Performance Comparison of CatGAN using different Facial Expression Recognition Datasets.pptx  
A ppt file used for a conference in South Korea
* VAE_SGAN  
Folder including codes needed to run experiments on different workstation. (Conversion from .ipynb to .py)
    * CatGAN.py  
    Main code to run experiments
    * networks.py  
    Code for creating networks
    * train_test.py  
    Code for training and testing the generated models
    * read.py  
    Code to rearrange results in a .csv file
* catgan_results
Folder including results from experiments.  
Full results (including images) from StyleGAN and VAE are not included due to the size limit.
    * table_of_results.xlsx  
    This file contains all relevant results of the experiment.
    

## References
<a id="1">[1]</a>  J. T. Springenberg, “Unsupervised and Semi-Supervised Learning with Categorical Generative Adversarial Networks,” *ICLR* 2016.
<a id="2">[2]</a> M. Pantic, M. Valstar, R. Rademaker, L. Maat, "Web-based database for facial expression analysis," *Proc. 13th ACM Int'l Conf. Multimedia (Multimedia '05)*, pp. 317-321, 2005. 
<a id="3">[3]</a> M. G. Calvo, D. Lundqvist, “Facial expressions of emotion (KDEF): Identification under different display-duration conditions,” *Behav Res*, Vol. 40, no. 1, pp. 109–115, 2008.
<a id="4">[4]</a> A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” *Adv. Neural Inf. Process*. Syst. 25, pp. 1106–1114, 2012.
<a id="5">[5]</a> K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” *Proc. 3rd Int. Conf. Learn. Representat.*, 2015.
<a id="6">[6]</a> K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, pp. 770–778, 2016.
<a id="7">[7]</a>  C. Szegedy et al., “Going deeper with convolutions,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 1–9, 2015.
<a id="8">[8]</a> V. Nair and G. E. Hinton. “Rectified Linear Units Improve Restricted Boltzmann Machines,” *ICML,* pp. 807–814, 2010.
<a id="9">[9]</a> A. L. Maas, A. Y. Hannun, A. Y. Ng, “Rectifier nonlinearities improve neural network acoustic models,” *ICML*, Vol. 30, 2013.
<a id="10">[10]</a> A. Hannun et al., “Deep Speech: Scaling up end-to-end speech recognition,” <a href = "http://arxiv.org/abs/1412.5567">*http://arxiv.org/abs/1412.5567*</a>, 2014
<a id="11">[11]</a> N. Qian, “On the momentum term in gradient descent learning algorithms,” *Neural Networks*, Vol. 12, no. 1, pp. 145-151, 1999.
<a id="12">[12]</a> T. Tieleman, G. Hinton, “Lecture 6.5—RmsProp: Divide the gradient by a running average of its recent magnitude,” *COURSERA: Neural Networks for Machine Learning*, 2012.
<a id="13">[13]</a>  D. Kingma, J. Ba, “Adam: A method for stochastic optimization,” *ICLR* 2015.
<a id="14">[14]</a> D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes", <a href = "https://arxiv.org/abs/1312.6114">*https://arxiv.org/abs/1312.6114*</a>, 2014
<a id="15">[15]</a> T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen and T. Aila,
"Analyzing and Improving the Image Quality of StyleGAN", <a href = "https://arxiv.org/abs/1912.04958">*https://arxiv.org/abs/1912.04958*</a> , 2020,




