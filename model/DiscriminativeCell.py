import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2
POOL = 2


class DiscriminativeCell(nn.Module):
    """
    Single discriminative layer
    """

    def __init__(self, input_size, hidden_size, first=False):
        super().__init__()
        self.intput_size = input_size
        self.hidden_size = hidden_size
        self.first = first
        if not first:
            self.from_bottom = nn.Conv2d(input_size['bottom'], hidden_size, KERNEL_SIZE, padding=PADDING)
        self.from_state = nn.Conv2d(input_size['top'], hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, bottom_up, state):
        input_projection = self.first and bottom_up or F.relu(F.max_pool2d(self.from_bottom(bottom_up), POOL, POOL))
        state_projection = F.relu(self.from_state(state))
        error = F.relu(torch.cat((input_projection - state_projection, state_projection - input_projection), 1))
        return error


def main():
    print('Define model')
    discriminator = DiscriminativeCell(input_size={'bottom': 3, 'top': 3}, hidden_size=3, first=True)

    print('Define some inputs')
    input_image = Variable(torch.Tensor(1, 3, 8, 12))
    system_state = Variable(torch.Tensor(1, 3, 8, 12))

    print('Forward inputs to the model')
    e = discriminator(input_image, system_state)

    # print output size
    print('The error has size', list(e.data.size()))



if __name__ == '__main__':
    main()
