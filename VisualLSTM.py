import torch
from torch.nn import LSTM

# Assumptions:
# 1) activ func will always be tanh in non-custom pytorch
# 2)

# Data members we need to present (and have)
# 1) Data on this cell: input, input_size, hidden_size, bias, weight_ih, weight_hh, bias: bias_ih, bias_hh


class VisualLSTM(LSTM):

    def __init__(self, input_size, hidden_size):
        self.input = None
        self.current_hidden_state = None
        self.current_memory_state = None
        self.forward_output = None

        super(LSTM, self).__init__('LSTM', input_size, hidden_size)

    def forward(self, input, hx=None):
        print("Forward: " + str(input))
        self.input = input
        self.forward_output = (super(VisualLSTM, self).forward(input, hx))
        return self.forward_output

    def __call__(self, *input, **kwargs):
        self.current_hidden_state, self.current_memory_state = super(VisualLSTM, self).__call__(*input, **kwargs)
        return self.current_hidden_state, self.current_memory_state

    def print(self):
        print("input size: " + str(self.input_size))
        print("hidden size: " + str(self.hidden_size))
        print("bias ih: " + str(self.bias_ih))
        print("bias hh: " + str(self.bias_hh))
        print("weight ih: " + str(self.weight_ih))
        print("weight hh: " + str(self.weight_hh))
        print("input DATA: " + str(self.input))
        print("Forward output: " + str(self.forward_output))
        print("current hidden state: " + str(self.current_hidden_state))
        print("current memory state: " + str(self.current_memory_state))





