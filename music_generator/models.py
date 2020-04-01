# PyTorch
import torch
import torch.nn as nn

# Custom
from utils import get_device


class LSTMSimple(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMSimple, self).__init__()
        self.inputs_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Model layout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)
        self.state = None

        self.training_losses = []
        self.validation_losses = []

    def init_state(self):
        computing_device = get_device()
        hidden = torch.zeros(1, 1, self.hidden_size).to(computing_device)
        cell = torch.zeros(1, 1, self.hidden_size).to(computing_device)
        self.state = (hidden, cell)

    def forward(self, sequence):
        self.state[0].detach_()
        self.state[1].detach_()
        lstm_out, self.state = self.lstm(sequence, self.state)
        lstm_out = self.drop(lstm_out)
        return self.fc(lstm_out)


class VanillaRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.inputs_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Model layout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)
        self.state = None

        self.training_losses = []
        self.validation_losses = []

    def init_state(self):
        self.state = torch.zeros(1, 1, self.hidden_size).to(get_device())

    def forward(self, sequence):
        self.state.detach_()
        out, self.state = self.rnn(sequence, self.state)
        out = self.drop(out)
        return self.fc(out)
