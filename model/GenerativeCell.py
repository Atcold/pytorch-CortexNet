import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.ConvLSTMCell import ConvLSTMCell


class GenerativeCell(nn.Module):
    """
    Single generative layer
    """

    def __init__(self, input_size, hidden_size, error_init_size=None):
        """
        Create a generative cell (bottom_up, r_state) -> error

        :param input_size: {'error': error_size, 'up_state': r_state_size}, r_state_size can be 0
        :param hidden_size: int, shooting dimensionality
        :param error_init_size: tuple, full size of initial (null) error
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.error_init_size = error_init_size
        self.memory = ConvLSTMCell(input_size['error']+input_size['up_state'], hidden_size)

    def forward(self, error, top_down_state, state):
        if error is None:  # we just started
            error = Variable(torch.zeros(self.error_init_size))
        model_input = error
        if top_down_state is not None:
            model_input = torch.cat((error, F.upsample_nearest(top_down_state, scale_factor=2)), 1)
        return self.memory(model_input, state)


def _test_layer2():
    print('Define model for layer 2')
    generator = GenerativeCell(input_size={'error': 2*16, 'up_state': 0}, hidden_size=16)

    print('Define error and top down state')
    input_error = Variable(torch.randn(1, 2*16, 4, 6))
    topdown_state = None

    print('Input error has size', list(input_error.data.size()))
    print('Top down state is None')

    print('Forward error and top down state to the model')
    state = None
    state = generator(input_error, topdown_state, state)

    # print output size
    print('Layer 2 state has size', list(state[0].data.size()))

    return state[0]  # the element 1 is the cell state


def _test_layer1(topdown_state):
    print('Define model for layer 1')
    generator = GenerativeCell(input_size={'error': 2*3, 'up_state': 16}, hidden_size=3)

    print('Define error and top down state')
    input_error = Variable(torch.randn(1, 2*3, 8, 12))

    print('Input error has size', list(input_error.data.size()))
    print('Top down state has size', list(topdown_state.data.size()))

    print('Forward error and top down state to the model')
    state = None
    state = generator(input_error, topdown_state, state)

    # print output size
    print('Layer 1 state has size', list(state[0].data.size()))


def _test_layers():
    state = _test_layer2()
    _test_layer1(topdown_state=state)


if __name__ == '__main__':
    _test_layers()
