import torch
from torch import nn
from torch.autograd import Variable

from model.DiscriminativeCell import DiscriminativeCell
from model.GenerativeCell import GenerativeCell


# Define some constants
LAYER_SIZE = [3] + list(2 ** p for p in range(4, 7))


class BuildOneLayerModel(nn.Module):
    """
    Build a one layer Prednet model
    """

    def __init__(self, error_size_list):
        super().__init__()
        self.discriminator = DiscriminativeCell(
            input_size={'input': LAYER_SIZE[0], 'state': LAYER_SIZE[0]},
            hidden_size=LAYER_SIZE[0],
            first=True
        )
        self.generator = GenerativeCell(
            input_size={'error': 2 * LAYER_SIZE[0], 'up_state': 0},
            hidden_size=LAYER_SIZE[0],
            error_init_size=error_size_list[0]
        )

    def forward(self, bottom_up_input, prev_error, state):
        state = self.generator(prev_error, None, state)
        error = self.discriminator(bottom_up_input, state[0])
        return error, state


class BuildTwoLayerModel(nn.Module):
    """
    Build a two layer Prednet model
    """

    def __init__(self, error_size_list):
        super().__init__()
        self.discriminator_1 = DiscriminativeCell(
            input_size={'input': LAYER_SIZE[0], 'state': LAYER_SIZE[0]},
            hidden_size=LAYER_SIZE[0],
            first=True
        )
        self.discriminator_2 = DiscriminativeCell(
            input_size={'input': 2 * LAYER_SIZE[0], 'state': LAYER_SIZE[1]},
            hidden_size=LAYER_SIZE[1]
        )
        self.generator_1 = GenerativeCell(
            input_size={'error': 2 * LAYER_SIZE[0], 'up_state': LAYER_SIZE[1]},
            hidden_size=LAYER_SIZE[0],
            error_init_size=error_size_list[0]
        )
        self.generator_2 = GenerativeCell(
            input_size={'error': 2 * LAYER_SIZE[1], 'up_state': 0},
            hidden_size=LAYER_SIZE[1],
            error_init_size=error_size_list[1]
        )

    def forward(self, bottom_up_input, error, state):
        state[1] = self.generator_2(error[1], None, state[1])
        state[0] = self.generator_1(error[0], state[1][0], state[0])
        error[0] = self.discriminator_1(bottom_up_input, state[0][0])
        error[1] = self.discriminator_2(error[0], state[1][0])
        return error, state


def test_one_layer_model():
    print('Create the input image')
    input_image = Variable(torch.rand(1, 3, 8, 12))

    print('Input has size', list(input_image.data.size()))

    error_init_size = (1, 6, 8, 12)
    print('The error initialisation size is', error_init_size)

    print('Define a 1 layer Prednet')
    model = BuildOneLayerModel([error_init_size])

    print('Forward input and state to the model')
    state = None
    error = None
    error, state = model(input_image, prev_error=error, state=state)

    print('The error has size', list(error.data.size()))
    print('The state has size', list(state[0].data.size()))


def test_two_layer_model():
    print('Create the input image')
    input_image = Variable(torch.rand(1, 3, 8, 12))

    print('Input has size', list(input_image.data.size()))

    error_init_size_list = [(1, 6, 8, 12), (1, 32, 4, 6)]
    print('The error initialisation sizes are', *error_init_size_list)

    print('Define a 2 layer Prednet')
    model = BuildTwoLayerModel(error_init_size_list)

    print('Forward input and state to the model')
    state = [None] * 2
    error = [None] * 2
    error, state = model(input_image, error=error, state=state)

    for layer in range(0, 2):
        print('Layer', layer + 1, 'error has size', list(error[layer].data.size()))
        print('Layer', layer + 1, 'state has size', list(state[layer][0].data.size()))


def main():
    test_one_layer_model()
    test_two_layer_model()


if __name__ == '__main__':
    main()
