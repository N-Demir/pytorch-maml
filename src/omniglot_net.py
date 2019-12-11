import numpy as np
import random
import math
from collections import OrderedDict

import torch
import torch.nn as nn

from layers import *


class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes, latent_dim, im_shape):
        super(ConditionalGenerator, self).__init__()
        self.img_shape = im_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Define the network following this architecture:
        # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
        self.operations = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(num_classes + latent_dim, 128)),
                ('leakyrelu1', nn.LeakyReLU(0.2, inplace=True)),
                ('linear2', nn.Linear(128, 256)),
                ('1dbn2', nn.BatchNorm1d(256, 0.8)),
                ('leakyrelu2', nn.LeakyReLU(0.2, inplace=True)),
                ('linear3', nn.Linear(256, 512)),
                ('1dbn3', nn.BatchNorm1d(512, 0.8)),
                ('leakyrelu3', nn.LeakyReLU(0.2, inplace=True)),
                ('linear4', nn.Linear(512, 1024)),
                ('1dbn4', nn.BatchNorm1d(1024, 0.8)),
                ('leakyrelu4', nn.LeakyReLU(0.2, inplace=True)),
                ('linear5', nn.Linear(1024, int(np.prod(self.img_shape)))),
                ('1dbn5', nn.BatchNorm1d(int(np.prod(self.img_shape)), 0.8)),
                ('leakyrelu5', nn.LeakyReLU(0.2, inplace=True)),
                # ('tanh', nn.Tanh()), # TODO: Should this be something else?
        ]))

        # Initialize weights
        self._init_weights()

    def forward(self, label, noise, weights=None):
        out = torch.cat((label.float(), noise), -1)
        if weights == None:
            out = self.operations(out)
        else:
            out = linear(out, weights['operations.linear1.weight'], weights['operations.linear1.bias'])
            out = leakyrelu(out, negative_slope=0.2, inplace=True)
            out = linear(out, weights['operations.linear2.weight'], weights['operations.linear2.bias'])
            out = batchnorm(out, weights['operations.1dbn2.weight'], weights['operations.1dbn2.bias'], momentum=0.8)
            out = leakyrelu(out, negative_slope=0.2, inplace=True)
            out = linear(out, weights['operations.linear3.weight'], weights['operations.linear3.bias'])
            out = batchnorm(out, weights['operations.1dbn3.weight'], weights['operations.1dbn3.bias'], momentum=0.8)
            out = leakyrelu(out, negative_slope=0.2, inplace=True)
            out = linear(out, weights['operations.linear4.weight'], weights['operations.linear4.bias'])
            out = batchnorm(out, weights['operations.1dbn4.weight'], weights['operations.1dbn4.bias'], momentum=0.8)
            out = leakyrelu(out, negative_slope=0.2, inplace=True)
            out = linear(out, weights['operations.linear5.weight'], weights['operations.linear5.bias'])
            out = batchnorm(out, weights['operations.1dbn5.weight'], weights['operations.1dbn5.bias'], momentum=0.8)
            out = leakyrelu(out, negative_slope=0.2, inplace=True)
            # out = nn.Tanh()(out)

        out = out.view(out.shape[0], *self.img_shape)
        return out

    def net_forward(self, label, noise, weights=None):
        return self.forward(label, noise, weights)
    
    def _init_weights(self):
        ''' Set weights to Gaussian, biases to zero '''
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        print('generator init weights')
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                #m.bias.data.zero_() + 1
                m.bias.data = torch.ones(m.bias.data.size())
    
    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        # TODO: breaks if nets are not identical
        # TODO: won't copy buffers, e.g. for batch norm
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.BatchNorm1d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()




class OmniglotNet(nn.Module):
    '''
    The base model for few-shot learning on Omniglot
    '''

    def __init__(self, num_classes, loss_fn, num_in_channels=3):
        super(OmniglotNet, self).__init__()
        # Define the network
        self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(num_in_channels, 64, 3)),
                ('bn1', nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(2,2)),
                ('conv2', nn.Conv2d(64,64,3)),
                ('bn2', nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(2,2)),
                ('conv3', nn.Conv2d(64,64,3)),
                ('bn3', nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu3', nn.ReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(2,2))
        ]))
        self.add_module('fc', nn.Linear(64, num_classes + 1))
        
        # Define loss function
        self.loss_fn = loss_fn

        # Initialize weights
        self._init_weights()

    def forward(self, x, weights=None):
        ''' Define what happens to data in the net '''
        if weights == None:
            x = self.features(x)
            x = x.view(x.size(0), 64)
            x = self.fc(x)
        else:
            x = conv2d(x, weights['features.conv1.weight'], weights['features.conv1.bias'])
            x = batchnorm(x, weight = weights['features.bn1.weight'], bias = weights['features.bn1.bias'], momentum=1)
            x = relu(x)
            x = maxpool(x, kernel_size=2, stride=2) 
            x = conv2d(x, weights['features.conv2.weight'], weights['features.conv2.bias'])
            x = batchnorm(x, weight = weights['features.bn2.weight'], bias = weights['features.bn2.bias'], momentum=1)
            x = relu(x)
            x = maxpool(x, kernel_size=2, stride=2) 
            x = conv2d(x, weights['features.conv3.weight'], weights['features.conv3.bias'])
            x = batchnorm(x, weight = weights['features.bn3.weight'], bias = weights['features.bn3.bias'], momentum=1)
            x = relu(x)
            x = maxpool(x, kernel_size=2, stride=2) 
            x = x.view(x.size(0), 64)
            x = linear(x, weights['fc.weight'], weights['fc.bias'])
        return x

    def net_forward(self, x, weights=None):
        return self.forward(x, weights)
    
    def _init_weights(self):
        ''' Set weights to Gaussian, biases to zero '''
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        print('init weights')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                #m.bias.data.zero_() + 1
                m.bias.data = torch.ones(m.bias.data.size())
    
    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        # TODO: breaks if nets are not identical
        # TODO: won't copy buffers, e.g. for batch norm
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
