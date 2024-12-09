import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=30, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype)) # Semantic weight matrix
        self.scale = scale # Scale factor for the cosine classifier
        self.reset_parameters() # Initialize the weight matrix  

    def reset_parameters(self):
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)  # Initialize the weight matrix

    def forward(self, x): 
        x = F.normalize(x, dim=-1) # Normalize the input features : shape (batch_size, 1024) --> (batch_size, 1024)
        weight = F.normalize(self.weight, dim=-1) # Normalize the weight matrix : shape (2, 1024) --> (2, 1024)
        return F.linear(x, weight) * self.scale # computing the cosine similarity between the Image and text fused features and the semantic weight matrix with the idea that prompt based weight which are accurate based on the fused would be higher than the other prompt based weight and applied scaling to have a better separation between the classes

    def apply_weight(self, weight):
        self.weight.data = weight.clone() # Apply the weight matrix to the model, where weight is nothing but the prompt based on the label
        #which are semantically embedded in the CLIP model and the weight matrix is the semantic weight matrix