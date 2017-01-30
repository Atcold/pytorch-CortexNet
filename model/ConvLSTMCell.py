import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input, prev_state):

        # get batch and spatial sizes
        batch_size = input.data.size()[0]
        spatial_size = input.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunks across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return hidden, cell


def main():
    """
    Run some basic tests on the API
    """

    # defines batch_size, channels, height, width
    b, c, h, w = 1, 3, 4, 8
    d = 5           # defines hidden state size
    lr = 1e-1       # defines learning rate
    T = 6           # define sequence length
    max_epoch = 20  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    print('Instantiate model')
    model = ConvLSTMCell(c, d)

    print('Create input and target Variables')
    x = Variable(torch.rand(T, b, c, h, w))
    y = Variable(torch.randn(T, b, d, h, w))

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        state = None
        loss = 0
        for t in range(0, T):
            state = model(x[t], state)
            loss += loss_fn(state[0], y[t])

        print('Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data[0]))

        # print('Zero grad parameters')
        model.zero_grad()

        # print('Compute new grad parameters')
        loss.backward()

        # print('Step against the gradient')
        for p in model.parameters():
            p.data.sub_(p.grad.data * lr)

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))


if __name__ == '__main__':
    main()
