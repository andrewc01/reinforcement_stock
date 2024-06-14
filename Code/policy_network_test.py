import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, output_size):
        super(PolicyNetwork, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)

        # Dense layer
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        # Dense layer
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        # Softmax layer
        out = self.softmax(out)

        return out
    

    