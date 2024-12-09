import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False), # Input shape : (batch_size, 1024) -> Output shape : (batch_size, 1024 // 4) --> (batch_size, 256)
            nn.ReLU(inplace=True), # Activation function
            nn.Linear(c_in // reduction, c_in, bias=False), # Input shape : (batch_size, 256) -> Output shape : (batch_size, 1024)
            nn.ReLU(inplace=True)  # Activation function
        )

    def forward(self, x):
        return self.fc(x) # Return the output of the fully connected layer

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, drop_probs):
        super().__init__()
        map_layers = [
            nn.Linear(input_dim, output_dim), # Input shape : (batch_size, 768) -> Output shape : (batch_size, 1024)
            nn.Dropout(p=drop_probs[0]) # Dropout layer
        ]
        
        for _ in range(1, num_layers): # Loop through the number of layers
            map_layers.extend([
                nn.ReLU(), # Activation function
                nn.Linear(output_dim, output_dim), # Input shape : (batch_size, 1024) -> Output shape : (batch_size, 1024)
                nn.Dropout(p=drop_probs[0]) # Dropout layer
            ])
        
        self.proj = nn.Sequential(*map_layers)

    def forward(self, x):
        return self.proj(x)