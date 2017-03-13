from torch import nn


class RG(nn.Module):
    """Recurrent Generative Module"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """ Initialise RG Module (parameters as nn.ConvTranspose2d)"""
        super().__init__()
        self.from_input = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.from_state = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, x, state):
        """
        Calling signature

        :param x: (input, output_size)
        :type x: tuple
        :param state: previous output
        :type state: torch.Tensor
        :return: current state
        :rtype: torch.Tensor
        """
        x = self.from_input(*x)  # the very first x is a tuple (input, expected_output_size)
        if state: x += self.from_state(state)
        return x
