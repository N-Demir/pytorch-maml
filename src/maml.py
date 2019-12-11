import click
import os, sys
import numpy as np
import random
from setproctitle import setproctitle
import inspect
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import SGD, Adam
from torch.nn.modules.loss import CrossEntropyLoss

from task import OmniglotTask, MNISTTask
from dataset import Omniglot, MNIST
from inner_loop import InnerLoop
from omniglot_net import OmniglotNet, ConditionalGenerator
from score import *
from data_loading import *
from tensorboardX import SummaryWriter
import datetime


class MetaLearner(object):
    def __init__(self,
                dataset,
                num_classes,
                num_inst,
                meta_batch_size, 
                meta_step_size, 
                inner_batch_size, 
                inner_step_size,
                num_updates, 
                num_inner_updates,
                loss_fn):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_inst = num_inst
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_batch_size = inner_batch_size
        self.inner_step_size = inner_step_size
        self.num_updates = num_updates
        self.num_inner_updates = num_inner_updates
        self.loss_fn = loss_fn
        self.generator_noise_dim = 10
        
        # Make the nets
        #TODO: don't actually need two nets
        num_input_channels = 1 if self.dataset == 'mnist' else 3
        self.net = OmniglotNet(num_classes, self.loss_fn, num_input_channels)
        self.net.cuda()
        self.fast_net = InnerLoop(num_classes, self.loss_fn, self.num_inner_updates, self.inner_step_size, self.inner_batch_size, self.meta_batch_size, num_input_channels)
        self.fast_net.cuda()

        self.gen = ConditionalGenerator(num_classes, self.generator_noise_dim, (1, 28, 28))
        self.gen.cuda()
        self.fast_gen = ConditionalGenerator(num_classes, self.generator_noise_dim, (1, 28, 28))
        self.fast_gen.cuda()

        self.opt = Adam(self.net.parameters(), lr=meta_step_size)
        self.gen_opt = Adam(self.gen.parameters(), lr=meta_step_size)
            
    def get_task(self, root, n_cl, n_inst, split='train'):
        if 'mnist' in root:
            return MNISTTask(root, n_cl, n_inst, split)
        elif 'omniglot' in root:
            return OmniglotTask(root, n_cl, n_inst, split)
        else:
            print('Unknown dataset')
            raise(Exception)

    def meta_update(self, task, ls, gen_grads):
        # TODO: Use gen_grads to also create a meta update for the generator
        print('\n Meta update \n')
        loader = get_data_loader(task, self.inner_batch_size, split='val')
        in_, target = loader.__iter__().next()

        # We use a dummy forward / backward pass to get the correct grads into self.net
        loss, out = forward_pass(self.net, in_, target)
        # Unpack the list of grad dicts
        gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []
        for (k,v) in self.net.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt.zero_grad()
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()


        # Repeat the above process but for the generator (LOL)
        _, gen_loss, _, gen_out = forward_pass(self.net, in_, target, generator=self.gen)
        gradients = {k: sum(d[k] for d in gen_grads) for k in gen_grads[0].keys()}
        hooks = []
        for (k,v) in self.generator.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()
        for h in hooks:
            h.remove()

    def test(self):
        # TODO: Make test also incorporate the generator
        num_in_channels = 1 if self.dataset == 'mnist' else 3
        test_net = OmniglotNet(self.num_classes, self.loss_fn, num_in_channels)
        mtr_loss, mtr_acc, mval_loss, mval_acc = 0.0, 0.0, 0.0, 0.0
        # Select ten tasks randomly from the test set to evaluate on
        for _ in range(10):
            # Make a test net with same parameters as our current net
            test_net.copy_weights(self.net)
            test_net.cuda()
            test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
            task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst, split='test')
            # Train on the train examples, using the same number of updates as in training
            train_loader = get_data_loader(task, self.inner_batch_size, split='train')
            for i in range(self.num_inner_updates):
                in_, target = train_loader.__iter__().next()
                loss, _  = forward_pass(test_net, in_, target)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            # Evaluate the trained model on train and val examples
            tloss, tacc = evaluate(test_net, train_loader)
            val_loader = get_data_loader(task, self.inner_batch_size, split='val')
            vloss, vacc = evaluate(test_net, val_loader)
            mtr_loss += tloss
            mtr_acc += tacc
            mval_loss += vloss
            mval_acc += vacc

        mtr_loss = mtr_loss / 10
        mtr_acc = mtr_acc / 10
        mval_loss = mval_loss / 10
        mval_acc = mval_acc / 10

        print('-------------------------')
        print('Meta train:', mtr_loss, mtr_acc)
        print('Meta val:', mval_loss, mval_acc)
        print('-------------------------')
        del test_net
        return mtr_loss, mtr_acc, mval_loss, mval_acc

    # def _train(self, exp):
    #     ''' debugging function: learn two tasks '''
    #     task1 = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst)
    #     task2 = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst)
    #     for it in range(self.num_updates):
    #         grads = []
    #         for task in [task1, task2]:
    #             # Make sure fast net always starts with base weights
    #             self.fast_net.copy_weights(self.net)
    #             _, g = self.fast_net.forward(task)
    #             grads.append(g)
    #         self.meta_update(task, grads)
            
    def train(self, exp):
        # For logging
        writer = SummaryWriter('../output/{}/'.format(exp + "_" + str(datetime.datetime.now())))

        tr_loss, tr_acc, val_loss, val_acc = [], [], [], []
        mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []
        for it in range(self.num_updates):
            # Evaluate on test tasks
            mt_loss, mt_acc, mv_loss, mv_acc = self.test() #TODO

            writer.add_scalar('meta_train_loss', mt_loss, it)
            writer.add_scalar('meta_train_acc', mt_acc, it)
            writer.add_scalar('meta_val_loss', mv_loss, it)
            writer.add_scalar('meta_val_acc', mv_acc, it)

            mtr_loss.append(mt_loss)
            mtr_acc.append(mt_acc)
            mval_loss.append(mv_loss)
            mval_acc.append(mv_acc)
            # Collect a meta batch update
            grads = []
            gen_grads = []
            tloss, tacc, vloss, vacc = 0.0, 0.0, 0.0, 0.0
            for i in range(self.meta_batch_size):
                task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst)
                self.fast_net.copy_weights(self.net)
                self.fast_gen.copy_weights(self.gen)
                metrics, g, g_gen = self.fast_net.forward(task, self.fast_gen)
                (trl, tra, vall, vala) = metrics
                grads.append(g)
                gen_grads.append(g_gen)
                tloss += trl
                tacc += tra
                vloss += vall
                vacc += vala

            # Perform the meta update
            print('Meta update', it)
            self.meta_update(task, grads, gen_grads) #TODO

            # Save a model snapshot every now and then
            if it % 500 == 0:
                torch.save(self.net.state_dict(), '../output/{}/train_iter_{}.pth'.format(exp, it))
                torch.save(self.gen.state_dict(), '../output/{}/gen_train_iter_{}.pth'.format(exp, it))

            # Save stuff
            tr_loss.append(tloss / self.meta_batch_size)
            tr_acc.append(tacc / self.meta_batch_size)
            val_loss.append(vloss / self.meta_batch_size)
            val_acc.append(vacc / self.meta_batch_size)

            np.save('../output/{}/tr_loss.npy'.format(exp), np.array(tr_loss))
            np.save('../output/{}/tr_acc.npy'.format(exp), np.array(tr_acc))
            np.save('../output/{}/val_loss.npy'.format(exp), np.array(val_loss))
            np.save('../output/{}/val_acc.npy'.format(exp), np.array(val_acc))

            np.save('../output/{}/meta_tr_loss.npy'.format(exp), np.array(mtr_loss))
            np.save('../output/{}/meta_tr_acc.npy'.format(exp), np.array(mtr_acc))
            np.save('../output/{}/meta_val_loss.npy'.format(exp), np.array(mval_loss))
            np.save('../output/{}/meta_val_acc.npy'.format(exp), np.array(mval_acc))

        writer.close()

@click.command()
@click.argument('exp')
@click.option('--dataset', type=str)
@click.option('--num_cls', type=int)
@click.option('--num_inst', type=int)
@click.option('--batch', type=int)
@click.option('--m_batch', type=int)
@click.option('--num_updates', type=int)
@click.option('--num_inner_updates', type=int)
@click.option('--lr',type=str)
@click.option('--meta_lr', type=str)
@click.option('--gpu', default=0)
def main(exp, dataset, num_cls, num_inst, batch, m_batch, num_updates, num_inner_updates, lr, meta_lr, gpu):
    random.seed(1337)
    np.random.seed(1337)
    setproctitle(exp)
    # Print all the args for logging purposes
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for arg in args:
        print(arg, values[arg])

    # make output dir
    output = '../output/{}'.format(exp)
    try:
        os.makedirs(output)
    except:
        pass
    # Set the gpu
    print('Setting GPU to', str(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    loss_fn = CrossEntropyLoss() 
    learner = MetaLearner(dataset, num_cls, num_inst, m_batch, float(meta_lr), batch, float(lr), num_updates, num_inner_updates, loss_fn)
    learner.train(exp)

if __name__ == '__main__':
    main()

