import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.ConvLSTMCell import ConvLSTMCell


class GenerativeCell(nn.Module):
    """
    Single generative layer
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory = ConvLSTMCell(input_size['error']+input_size['state'], hidden_size)

    def forward(self, error, topdown_state, state):
        return self.memory(torch.cat((error, F.upsample_nearest(topdown_state, scale_factor=2)), 1), state)


def test_layer2():
    print('Define model for layer 2')
    generator = GenerativeCell(input_size={'error': 2*16, 'state': 32}, hidden_size=16)

    print('Define error and top down state')
    input_error = Variable(torch.randn(1, 2*16, 4, 6))
    topdown_state = Variable(torch.randn(1, 32, 2, 3))

    print('Input error has size', list(input_error.data.size()))
    print('Top down state has size', list(topdown_state.data.size()))

    print('Forward error and top down state to the model')
    state = None
    state = generator(input_error, topdown_state, state)

    # print output size
    print('Layer 2 state has size', list(state[0].data.size()))

    return state[0]  # the element 1 is the cell state


def test_layer1(topdown_state):
    print('Define model for layer 1')
    generator = GenerativeCell(input_size={'error': 2*3, 'state': 16}, hidden_size=3)

    print('Define error and top down state')
    input_error = Variable(torch.randn(1, 2*3, 8, 12))

    print('Input error has size', list(input_error.data.size()))
    print('Top down state has size', list(topdown_state.data.size()))

    print('Forward error and top down state to the model')
    state = None
    state = generator(input_error, topdown_state, state)

    # print output size
    print('Layer 1 state has size', list(state[0].data.size()))


def test_layers():
    state = test_layer2()
    test_layer1(topdown_state=state)


if __name__ == '__main__':
    test_layers()
