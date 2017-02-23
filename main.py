import argparse
import time
import math
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim

from sys import exit
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torchvision import transforms as trn

from data.VideoFolder import VideoFolder, BatchSampler, VideoCollate

parser = argparse.ArgumentParser(description='PyTorch MatchNet generative model training script')
_ = parser.add_argument  # define add_argument shortcut
_('--data',         type=str,   default='./data/tmp_data_set', help='location of the video data')
_('--model',        type=str,   default='model_01',            help='type of auto-encoder')
_('--size',         type=int,   default=(3, 6, 12), nargs='*', help='number and size of hidden layers', metavar='S')
_('--spatial-size', type=int,   default=(256, 256), nargs=2,   help='frame cropping size', metavar=('H', 'W'))
_('--nb-videos',    type=int,   default=10,                    help='number of training videos')
_('--lr',           type=float, default=0.1,                   help='initial learning rate')
_('--momentum',     type=float, default=0.9, metavar='M',      help='momentum')
_('--weight-decay', type=float, default=1e-4, metavar='W',     help='weight decay (default: 1e-4)')
_('--clip',         type=float, default=0.5,                   help='gradient clipping')
_('--epochs',       type=int,   default=6,                     help='upper epoch limit')
_('--batch-size',   type=int,   default=20,    metavar='B',    help='batch size')
_('--big-t',        type=int,   default=20,                    help='sequence length')
_('--seed',         type=int,   default=0,                     help='random seed')
_('--log-interval', type=int,   default=200,   metavar='N',    help='report interval')
_('--save',         type=str,   default='model.pth.tar',       help='path to save the final model')
_('--cuda',                     action='store_true',           help='use CUDA')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


def main():

    # Build the model
    if args.model == 'model_01':
        from model.Model01 import Model01 as Model
    else:
        print('\n{:#^80}\n'.format(' Please select a valid model '))
        exit()

    print('Define model')
    model = Model(args.size + (args.nb_videos,), args.spatial_size)

    if args.cuda:
        model.cuda()

    print('Create a MSE and NLL criterions')
    mse = nn.MSELoss()
    nll = nn.CrossEntropyLoss()

    print('Instantiate a SGD optimiser')
    optimiser = optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Load data
    print('Define image pre-processing')
    # normalise? do we care?
    t = trn.Compose((trn.ToPILImage(), trn.CenterCrop(args.spatial_size), trn.ToTensor()))

    print('Define train data loader')
    train_path = path.join(args.data, 'train')
    train_data = VideoFolder(root=train_path, transform=t)
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
    val_path = path.join(args.data, 'val')
    val_data = VideoFolder(root=val_path, transform=t)
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size * args.big_t,
        shuffle=False,
        sampler=BatchSampler(data_source=val_data, batch_size=args.batch_size),
        num_workers=1,
        collate_fn=VideoCollate(batch_size=args.batch_size),
        pin_memory=True
    )

    # Loop over epochs
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        train(train_loader, model, (mse, nll), optimiser, epoch)
        # val_loss = validate(val_loader, model, mse)
        val_loss = 0

        print(80 * '-', '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.
              format(epoch + 1, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)), 80 * '-', sep='\n')

    if args.save != '':
        torch.save(model, args.save)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if not h: return None
    elif type(h) == V: return V(h.data)
    else: return list(repackage_hidden(v) for v in h)


def train(train_loader, model, loss_fun, optimiser, epoch):
    print('Training epoch', epoch + 1)
    total_loss = 0
    mse, nll = loss_fun
    start_time = time.time()
    state = None
    for batch_nb, (x, y) in enumerate(train_loader):
        print('Training batch', batch_nb + 1)
        state = repackage_hidden(state)
        loss = 0
        # BTT loop
        for t in range(0, args.big_t - 1):
            # TODO: deal with sample x[T + 1]
            (x_hat, state), (emb, idx) = model(V(x[t]), state)
            loss += mse(x_hat, V(x[t + 1])) + nll(idx, V(y[t]))

        # compute gradient and do SGD step
        model.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.data

        if batch_nb % args.log_interval == 0 and batch_nb:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.
                  format(epoch, batch_nb + 1, len(train_loader) // args.bptt, 0,
                         elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == '__main__':
    main()
