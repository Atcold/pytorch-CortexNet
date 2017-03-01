import argparse
import os
import time
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim

from sys import exit, argv
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torchvision import transforms as trn
from datetime import timedelta

from data.VideoFolder import VideoFolder, BatchSampler, VideoCollate

parser = argparse.ArgumentParser(description='PyTorch MatchNet generative model training script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_ = parser.add_argument  # define add_argument shortcut
_('--data', type=str, default='./data/processed-data', help='location of the video data')
_('--model', type=str, default='model_01', help='type of auto-encoder')
_('--size', type=int, default=(3, 6, 12), nargs='*', help='number and size of hidden layers', metavar='S')
_('--spatial-size', type=int, default=(256, 256), nargs=2, help='frame cropping size', metavar=('H', 'W'))
_('--lr', type=float, default=0.1, help='initial learning rate')
_('--momentum', type=float, default=0.9, metavar='M', help='momentum')
_('--weight-decay', type=float, default=1e-4, metavar='W', help='weight decay')
_('--lambda', type=float, default=0.1, help='CE stabiliser multiplier', dest='lambda_', metavar='λ')
_('--epochs', type=int, default=6, help='upper epoch limit')
_('--batch-size', type=int, default=20, metavar='B', help='batch size')
_('--big-t', type=int, default=20, help='sequence length', metavar='T')
_('--seed', type=int, default=0, help='random seed')
_('--log-interval', type=int, default=200, metavar='N', help='report interval')
_('--save', type=str, default='model.pth.tar', help='path to save the final model')
_('--cuda', action='store_true', help='use CUDA')
args = parser.parse_args()
args.size = tuple(args.size)  # cast to tuple

# Print current options
print('CLI arguments:', ' '.join(argv[1:]))

# Print current commit
if path.isdir('.git'):  # if we are in a repo
    with os.popen('git rev-parse HEAD') as pipe:  # get the HEAD's hash
        print('Current commit hash:', pipe.read(), end='')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


def main():
    # Load data
    print('Define image pre-processing')
    # normalise? do we care?
    t = trn.Compose((trn.ToPILImage(), trn.CenterCrop(args.spatial_size), trn.ToTensor()))

    print('Define train data loader')
    train_data_name = 'train_data.tar'
    if os.access(train_data_name, os.R_OK):
        train_data = torch.load(train_data_name)
    else:
        train_path = path.join(args.data, 'train')
        train_data = VideoFolder(root=train_path, transform=t, video_index=True)
        torch.save(train_data, train_data_name)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size * args.big_t,
        shuffle=False,
        sampler=BatchSampler(data_source=train_data, batch_size=args.batch_size),
        num_workers=1,
        collate_fn=VideoCollate(batch_size=args.batch_size),
        pin_memory=True
    )

    print('Define validation data loader')
    val_data_name = 'val_data.tar'
    if os.access(val_data_name, os.R_OK):
        val_data = torch.load(val_data_name)
    else:
        val_path = path.join(args.data, 'val')
        val_data = VideoFolder(root=val_path, transform=t, video_index=True)
        torch.save(val_data, val_data_name)

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=BatchSampler(data_source=val_data, batch_size=args.batch_size),
        num_workers=1,
        collate_fn=VideoCollate(batch_size=args.batch_size),
        pin_memory=True
    )

    # Build the model
    if args.model == 'model_01':
        from model.Model01 import Model01 as Model
    else:
        print('\n{:#^80}\n'.format(' Please select a valid model '))
        exit()

    print('Define model')
    nb_train_videos = len(train_data.videos)
    model = Model(args.size + (nb_train_videos,), args.spatial_size)

    print('Create a MSE and balanced NLL criterions')
    mse = nn.MSELoss()

    # balance classes based on frames per video; default balancing weight is 1.0f
    w = torch.Tensor(train_data.frames_per_video)
    w.div_(w.mean()).pow_(-1)
    nll = nn.CrossEntropyLoss(w)

    if args.cuda:
        model.cuda()
        mse.cuda()
        nll.cuda()

    print('Instantiate a SGD optimiser')
    optimiser = optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Loop over epochs
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        train(train_loader, model, (mse, nll), optimiser, epoch)
        print(80 * '-', '| end of epoch {:3d} |'.format(epoch + 1), sep='\n', end=' ')
        val_loss = validate(val_loader, model, (mse, nll))
        elapsed_time = str(timedelta(seconds=int(time.time() - epoch_start_time)))  # HH:MM:SS time format
        print('time: {} | MSE {:5.2f} | CE {:5.2f}'.format(elapsed_time, val_loss['mse'], val_loss['ce']))
        print(80 * '-')

    if args.save != '':
        torch.save(model, args.save)


def train(train_loader, model, loss_fun, optimiser, epoch):
    print('Training epoch', epoch + 1)
    model.train()  # set model in train mode
    total_loss = {'mse': 0, 'ce': 0}
    mse, nll = loss_fun

    def compute_loss(x_, next_x, y_, state_):
        (x_hat_, state_), (_, idx_) = model(V(x_), state_)
        mse_loss_ = mse(x_hat_, V(next_x))
        ce_loss_ = nll(idx_, V(y_))
        total_loss['mse'] += mse_loss_.data[0]
        total_loss['ce'] += ce_loss_.data[0]
        return ce_loss_, mse_loss_, state_

    data_time = 0
    batch_time = 0
    end_time = time.time()
    if not hasattr(train, 'state'): train.state = None  # init state attribute
    state = train.state
    from_past = None
    for batch_nb, (x, y) in enumerate(train_loader):
        data_time += time.time() - end_time
        if args.cuda:
            x = x.cuda(async=True)
            y = y.cuda(async=True)
        state = repackage_state(state)
        loss = 0
        # BTT loop
        if from_past:
            ce_loss, mse_loss, state = compute_loss(from_past[0], x[0], from_past[1], state)
            loss += mse_loss + ce_loss * args.lambda_
        for t in range(0, min(args.big_t, x.size(0)) - 1):  # first batch we go only T - 1 steps forward / backward
            ce_loss, mse_loss, state = compute_loss(x[t], x[t + 1], y[t], state)
            loss += mse_loss + ce_loss * args.lambda_

        # compute gradient and do SGD step
        model.zero_grad()
        loss.backward()
        optimiser.step()

        # save last column for future
        from_past = x[-1], y[-1]

        # measure batch time
        batch_time += time.time() - end_time
        end_time = time.time()  # for computing data_time

        if (batch_nb + 1) % args.log_interval == 0:
            cur_mse_loss = total_loss['mse'] / args.log_interval
            cur_ce_loss = total_loss['ce'] / args.log_interval
            avg_batch_time = batch_time * 1e3 / args.log_interval
            avg_data_time = data_time * 1e3 / args.log_interval
            print('| epoch {:3d} | {:4d}/{:4d} batches | lr {:02.2f} |'
                  ' ms/batch {:7.2f} | ms/data {:7.2f} | MSE {:5.2f} | CE {:5.2f}'.
                  format(epoch + 1, batch_nb + 1, len(train_loader), args.lr,
                         avg_batch_time, avg_data_time, cur_mse_loss, cur_ce_loss))
            for k in total_loss: total_loss[k] = 0  # zero the losses
            batch_time = 0
            data_time = 0
    train.state = state  # preserve state across epochs


def validate(val_loader, model, loss_fun):
    model.eval()  # set model in evaluation mode
    total_loss = {'mse': 0, 'ce': 0}
    mse, nll = loss_fun
    batches = iter(val_loader)

    (x, y) = next(batches)
    if args.cuda:
        x = x.cuda(async=True)
        y = y.cuda(async=True)
    if not hasattr(validate, 'state'): validate.state = None  # init state attribute
    state = validate.state
    for (next_x, next_y) in batches:
        if args.cuda:
            next_x = next_x.cuda(async=True)
            next_y = next_y.cuda(async=True)
        (x_hat, state), (_, idx) = model(V(x[0], volatile=True), state)  # do not compute graph (volatile)
        mse_loss = mse(x_hat, V(next_x[0]))
        ce_loss = nll(idx, V(y[0])) * args.lambda_
        total_loss['mse'] += mse_loss.data[0]
        total_loss['ce'] += ce_loss.data[0]
        x, y = next_x, next_y
    validate.state = state  # preserve state across epochs

    for k in total_loss: total_loss[k] /= (len(val_loader) - 1)  # average out
    return total_loss


def repackage_state(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """
    if not h:
        return None
    elif type(h) == V:
        return V(h.data)
    else:
        return list(repackage_state(v) for v in h)


if __name__ == '__main__':
    main()


__author__ = "Alfredo Canziani"
__credits__ = ["Alfredo Canziani"]
__maintainer__ = "Alfredo Canziani"
__email__ = "alfredo.canziani@gmail.com"
__status__ = "Development"  # "Prototype", "Development", or "Production"
__date__ = "Feb 17"
