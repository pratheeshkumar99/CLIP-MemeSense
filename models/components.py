import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, drop_probs):
        super().__init__()
        map_layers = [
            nn.Linear(input_dim, output_dim),
            nn.Dropout(p=drop_probs[0])
        ]
        
        for _ in range(1, num_layers):
            map_layers.extend([
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.Dropout(p=drop_probs[0])
            ])
        
        self.proj = nn.Sequential(*map_layers)

    def forward(self, x):
        return self.proj(x)