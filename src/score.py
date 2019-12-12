import numpy as np

import torch
from torch.autograd import Variable

'''
Helper methods for evaluating a classification network
'''

def count_correct(pred, target):
    ''' count number of correct classification predictions in a batch '''
    pairs = [int(x==y) for (x, y) in zip(pred, target)]
    return sum(pairs)

def forward_pass(net, in_, target, net_weights=None, generator=None, gen_weights=None, outer_update=False):
    ''' forward in_ through the net, return loss and output '''
    # print(net)
    # print(in_)
    # print(target)

    real_input_var = Variable(in_).cuda(async=True)
    target_var = Variable(target).cuda(async=True)

    # Real Loss
    out = net.net_forward(real_input_var, net_weights)
    # real_loss = net.loss_fn(out, target_var) if not outer_update else net.loss_fn(out[:, :-1], target_var)
    real_loss = net.loss_fn(out, target_var)

    # Fake Loss
    if generator is not None:
        noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (in_.shape[0], generator.latent_dim)))).cuda(async=True)
        # one_hot_targets = torch.nn.functional.one_hot(target_var, generator.num_classes)
        fake_input_var = generator.forward(noise, real_input_var, gen_weights)
        fake_target_var = Variable(torch.ones(target.shape, dtype=torch.int64) * generator.num_classes).cuda(async=True)

        fake_out = net.net_forward(fake_input_var, net_weights)
        fake_loss = net.loss_fn(fake_out, fake_target_var)

        net_loss = (fake_loss + real_loss * 5) / 6.0

        # Get generator loss
        gen_loss = (-fake_loss + real_loss * 5) / 6.0

        if outer_update:
            return real_loss, gen_loss, out, fake_input_var
        return net_loss, gen_loss, out, fake_input_var

    else:
        return real_loss, out

def evaluate(net, loader, weights=None):
    ''' evaluate the net on the data in the loader '''
    num_correct = 0
    loss = 0
    for i, (in_, target) in enumerate(loader):
        batch_size = in_.numpy().shape[0]
        l, out = forward_pass(net, in_, target, weights)
        loss += l.cpu().detach()
        num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
    return float(loss) / len(loader), float(num_correct) / (len(loader)*batch_size)
