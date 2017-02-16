import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable as V
from math import ceil


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2
KERNEL_STRIDE = 2
OUTPUT_ADJUST = KERNEL_SIZE - 2 * PADDING


class Model01(nn.Module):
    """
    Generate a constructor for model_01 type of network
    """

    def __init__(self, network_size: tuple, input_spatial_size: tuple) -> None:
        """
        Initialise Model01 constructor

        :param network_size: (n, h1, h2, ..., emb_size, nb_videos)
        :type network_size: tuple
        :param input_spatial_size: (height, width)
        :type input_spatial_size: tuple
        """
        super().__init__()
        self.hidden_layers = len(network_size) - 2

        print('Hidden layers:', self.hidden_layers)
        print('Net sizing:', network_size)
        print('Input spatial size:', input_spatial_size)

        # main auto-encoder blocks
        self.activation_size = [input_spatial_size]
        for layer in range(0, self.hidden_layers):
            # print some annotation when building model
            print('Layer', str(layer + 1), 40 * '-')
            print('Bottom size:', self.activation_size[-1])
            self.activation_size.append(tuple(ceil(s / 2) for s in self.activation_size[layer]))
            print('Top size:', self.activation_size[-1])

            # init D (discriminative) blocks
            setattr(self, 'D_' + str(layer + 1), nn.Conv2d(
                in_channels=network_size[layer], out_channels=network_size[layer + 1],
                kernel_size=KERNEL_SIZE, stride=KERNEL_STRIDE, padding=PADDING
            ))
            setattr(self, 'BN_D_' + str(layer + 1), nn.BatchNorm2d(network_size[layer + 1]))

            # init G (generative) blocks
            setattr(self, 'G_' + str(layer + 1), nn.ConvTranspose2d(
                in_channels=network_size[layer + 1], out_channels=network_size[layer],
                kernel_size=KERNEL_SIZE, stride=KERNEL_STRIDE, padding=PADDING
            ))
            setattr(self, 'BN_G_' + str(layer + 1), nn.BatchNorm2d(network_size[layer]))
        print(48 * '-')

        # init auxiliary classifier
        self.average = nn.AvgPool2d(self.activation_size[-1])
        self.stabiliser = nn.Linear(network_size[-2], network_size[-1])

    def forward(self, x, state):
        activation_sizes = [x.size()]  # start from the input
        residuals = list()
        for layer in range(0, self.hidden_layers):  # connect discriminative blocks
            x = getattr(self, 'D_' + str(layer + 1))(x)
            residuals.append(x)
            if layer < self.hidden_layers - 1 and state: x += state[layer]
            x = f.relu(x)
            x = getattr(self, 'BN_D_' + str(layer + 1))(x)
            activation_sizes.append(x.size())  # cache output size for later retrieval
        state = state or [None] * (self.hidden_layers - 1)
        for layer in reversed(range(0, self.hidden_layers)):  # connect generative blocks
            x = getattr(self, 'G_' + str(layer + 1))(x, activation_sizes[layer])
            if layer:
                x += residuals[layer - 1]
                state[layer - 1] = x

            x = f.relu(x)
            x = getattr(self, 'BN_G_' + str(layer + 1))(x)
        x_mean = self.average(residuals[-1])
        video_index = self.stabiliser(x_mean.view(x_mean.size(0), -1))

        return (x, state), (x_mean, video_index)


def _test_model_01():
    T = 2
    x = torch.rand(T + 1, 1, 3, 4 * 2**3 + 3, 6 * 2**3 + 5)
    K = 10
    y = torch.LongTensor(T, 1).random_(K)
    model_01 = Model01(network_size=(3, 6, 12, 18, K),
                       input_spatial_size=x[0].size()[2:])

    state = None
    (x_hat, state), (emb, idx) = model_01(V(x[0]), state)

    print('Input size:', tuple(x.size()))
    print('Output size:', tuple(x_hat.data.size()))
    print('Video index size:', tuple(idx.size()))
    for i, s in enumerate(state):
        print('State', i + 1, 'has size:', tuple(s.size()))
    print('Embedding has size:', emb.data.numel())

    mse = nn.MSELoss()
    nll = nn.CrossEntropyLoss()
    x_next = V(x[1])
    y_var = V(y[0])
    loss_t1 = mse(x_hat, x_next) + nll(idx, y_var)

    from visualise import show_graph
    show_graph(loss_t1)

    # run one more time
    (x_hat, _), (_, idx) = model_01(V(x[1]), state)

    x_next = V(x[2])
    y_var = V(y[1])
    loss_t2 = mse(x_hat, x_next) + nll(idx, y_var)
    loss_tot = loss_t2 + loss_t1

    show_graph(loss_tot)

if __name__ == '__main__':
    _test_model_01()
