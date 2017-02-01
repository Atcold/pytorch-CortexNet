import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.DiscriminativeCell import DiscriminativeCell
from model.GenerativeCell import GenerativeCell


# Define some constants
LAYER_SIZE = [3] + list(2**p for p in range(4, 7))


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
            input_size={'error': 2*LAYER_SIZE[0], 'up_state': 0},
            hidden_size=LAYER_SIZE[0],
            error_init_size=error_size_list[0]
        )

    def forward(self, bottom_up_input, prev_error, state):
        state = self.generator(prev_error, None, state)
        error = self.discriminator(bottom_up_input, state[0])
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


def main():
    test_one_layer_model()


if __name__ == '__main__':
    main()
