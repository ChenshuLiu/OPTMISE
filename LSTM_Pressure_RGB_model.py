import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # Define LSTM layer with input size and hidden size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define fully connected layer
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, output_size)
    
    def forward(self, x):
        # Forward pass through LSTM layer
        out, _ = self.lstm(x)
        # Forward pass through fully connected layer
        out = self.fc1(out)  # Use only the output of the last time step
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out[:, -1, :])
        return out