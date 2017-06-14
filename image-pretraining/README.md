# Image pre-training

Find the original code at [PyTorch ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet).  
This adaptation trains the discriminative branch of CortexNet for TempoNet.

## Training

To train the discriminative branch of CortexNet, run `main.py` with the path to an image data set:

```bash
python main.py <image data path> | tee train.log
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs.

## Usage

```
usage: main.py [-h] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
               [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--size [S [S ...]]]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --size [S [S ...]]    number and size of hidden layers
```
