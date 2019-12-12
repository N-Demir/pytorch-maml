import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

from omniglot_net import OmniglotNet
from layers import *
from score import *
from data_loading import *

class InnerLoop(OmniglotNet):
    '''
    This module performs the inner loop of MAML
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''

    def __init__(self, num_classes, loss_fn, num_updates, step_size, batch_size, meta_batch_size, num_in_channels=3):
        super(InnerLoop, self).__init__(num_classes, loss_fn, num_in_channels)
        # Number of updates to be taken
        self.num_updates = num_updates

        # Step size for the updates
        self.step_size = step_size

        # PER CLASS Batch size for the updates
        self.batch_size = batch_size

        # for loss normalization 
        self.meta_batch_size = meta_batch_size
    

    def net_forward(self, x, weights=None):
        return super(InnerLoop, self).forward(x, weights)

    # def forward_pass(self, in_, target, weights=None):
    #     ''' Run data through net, return loss and output '''
    #     input_var = torch.autograd.Variable(in_).cuda(async=True)
    #     target_var = torch.autograd.Variable(target).cuda(async=True)
    #     # Run the batch through the net, compute loss
    #     out = self.net_forward(input_var, weights)
    #     loss = self.loss_fn(out, target_var)
    #     return loss, out
    
    def forward(self, task, generator):
        train_loader = get_data_loader(task, self.batch_size)
        val_loader = get_data_loader(task, self.batch_size, split='val')
        ##### Test net before training, should be random accuracy ####
        tr_pre_loss, tr_pre_acc = evaluate(self, train_loader)
        val_pre_loss, val_pre_acc = evaluate(self, val_loader)
        fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
        gen_fast_weights = OrderedDict((name, param) for (name, param) in generator.named_parameters())
        for i in range(self.num_updates):
            print('inner step', i)
            in_, target = train_loader.__iter__().next()
            if i==0:
                #TODO: If we want to separate discriminator and generator updates do it here
                net_loss, _ = forward_pass(self, in_, target)
                net_grads = torch.autograd.grad(net_loss, self.parameters(), create_graph=True)
            if i >= 3:
                net_loss, gen_loss, _, fake_out = forward_pass(self, in_, target, generator=generator, net_weights=fast_weights, gen_weights=gen_fast_weights)
                net_grads = torch.autograd.grad(net_loss, fast_weights.values(), create_graph=True)
                gen_grads = torch.autograd.grad(gen_loss, gen_fast_weights.values(), create_graph=True)
                gen_fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(gen_fast_weights.items(), gen_grads))
            else:
                net_loss, _ = forward_pass(self, in_, target, net_weights=fast_weights)
                net_grads = torch.autograd.grad(net_loss, fast_weights.values(), create_graph=True)
            fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(fast_weights.items(), net_grads))
        
        ##### Test net after training, should be better than random ####
        tr_post_loss, tr_post_acc = evaluate(self, train_loader, fast_weights)
        val_post_loss, val_post_acc = evaluate(self, val_loader, fast_weights) 
        print('\n Train Inner step Loss', tr_pre_loss, tr_post_loss)
        print('Train Inner step Acc', tr_pre_acc, tr_post_acc)
        print('\n Val Inner step Loss', val_pre_loss, val_post_loss)
        print('Val Inner step Acc', val_pre_acc, val_post_acc)
        
        # Compute the meta gradient and return it
        in_, target = val_loader.__iter__().next()
        net_loss, _ = forward_pass(self, in_, target, net_weights=fast_weights, outer_update=True) 
        net_loss = net_loss / self.meta_batch_size # normalize loss
        net_grads = torch.autograd.grad(net_loss, self.parameters(), retain_graph=True)
        gen_grads = torch.autograd.grad(net_loss, generator.parameters())

        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), net_grads)}
        gen_meta_grads = {name:g for ((name, _), g) in zip(generator.named_parameters(), gen_grads)}
        metrics = (tr_post_loss, tr_post_acc, val_post_loss, val_post_acc)
        return metrics, meta_grads, gen_meta_grads, fake_out

