import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from .components import Adapter, LinearProjection
from .classifier import CosineClassifier

class MemeCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Load and freeze CLIP
        self.clip_model, _ = clip.load(cfg.clip_variant, device=cfg.device, jit=False)
        self.clip_model.float()
        for param in self.clip_model.parameters(): # Freeze the CLIP model
            param.requires_grad = False
        
        # Initialize components
        self.img_adapter = Adapter(cfg.map_dim, 4).to(self.clip_model.dtype) # This is nothing but a fully connected layer that serves as a bottleneck layer which acts as feature image feature extractor
        self.text_adapter = Adapter(cfg.map_dim, 4).to(self.clip_model.dtype) # This is nothing but a fully connected layer that serves as a bottleneck layer which acts as a feature text feature extractor
         
        self.image_map = LinearProjection(
            cfg.unmapped_dim,  # This is the dimension of the input features --> 768
            cfg.map_dim, # This is the dimension of the output features --> 1024
            cfg.num_mapping_layers,  # This is the number of layers in the projection head --> 1
            cfg.drop_probs # This is the dropout probability --> [0.1, 0.4, 0.2]
        )
        self.text_map = LinearProjection(
            cfg.unmapped_dim, 
            cfg.map_dim,
            cfg.num_mapping_layers, 
            cfg.drop_probs
        )
        
        # Pre-output layers
        self.pre_output = self._build_pre_output_layers()
        
        # Classifier
        self.classifier = CosineClassifier(
            feat_dim=cfg.map_dim, # This is the dimension of the input features --> 1024
            num_classes=cfg.num_classes, # This is the number of classes --> 2
            scale=cfg.scale, # This is the scale factor for the cosine classifier --> 30
            dtype=self.clip_model.dtype # This is the data type of the model --> torch.float32
        )
        
        self.init_head_text_feat() # Inintialize the weights of the classifier

    def _build_pre_output_layers(self):
        layers = [nn.Dropout(p=self.cfg.drop_probs[1])] # Dropout layer
        if self.cfg.num_pre_output_layers >= 1:
            layers.extend([
                nn.Linear(self.cfg.map_dim, self.cfg.map_dim), # Input shape : (batch_size, 1024) -> Output shape : (batch_size, 1024)
                nn.ReLU(), # Activation function
                nn.Dropout(p=self.cfg.drop_probs[2]) # Dropout layer
            ])
        return nn.Sequential(*layers)

    def init_head_text_feat(self):
        template = "a photo of a {}." # Template for the prompt
        prompts = [template.format(c.replace("_", " ")) for c in self.cfg.class_names] # Get the prompts based on the class names : for a task of hate detection, the prompts will be ['a photo of a Benign Meme.', 'a photo of a Harmful Meme.']
        prompts = clip.tokenize(prompts, context_length=77, truncate=True).to(self.cfg.device) # Tokenize the prompts
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(prompts) # Extract the text features from the CLIP model for the prompts : shape (no. of classes for the task, 768); if the task is hate detection, shape will be (2, 768)
            text_features = F.normalize(text_features, dim=-1) # Normalize the text features ; shape (no. of classes for the task, 768)
            text_features = text_features @ self.clip_model.visual.proj.t() # Multiply the text features with the projection matrix of the CLIP model  : shape (no. of classes for the task, 768) -> (no. of classes for the task, 1024)
            text_features = F.normalize(text_features, dim=-1) # Normalize the text features ; shape (no. of classes for the task, 1024)
            self.classifier.apply_weight(text_features) # Apply the weights to the classifier

    def forward(self, image_features, text_features):
        # Project features
        image_projection = self.image_map(image_features) # Basically a Fully connected layer that maps the CLIP extacted images features to a higher dimension : shape (batch_size, 768) -> (batch_size, 1024)
        text_projection = self.text_map(text_features) # Basically a Fully connected layer that maps the CLIP extacted text features to a higher dimension : shape (batch_size, 768) -> (batch_size, 1024)
        
        # Apply adapters
        image_adapted = self.img_adapter(image_projection) # It serves a bottleneck layer which acts as a feature image feature extractor : shape (batch_size, 1024) -> (batch_size, 1024)
        text_adapted = self.text_adapter(text_projection) # It serves a bottleneck layer which acts as a feature text feature extractor : shape (batch_size, 1024) -> (batch_size, 1024)
        
        # Residual connections
        image_features = self.cfg.ratio * image_adapted + (1 - self.cfg.ratio) * image_projection # This is the residual connection that feature extracted from the image at adapter layer and the projected image features by weight ratio of alpha. This is done to prevent the model from overfitting : shape (batch_size, 1024)
        text_features = self.cfg.ratio * text_adapted + (1 - self.cfg.ratio) * text_projection # This is the residual connection that feature extracted from the text at adapter layer and the projected text features by weight ratio of alpha. This is done to prevent the model from overfitting : shape (batch_size, 1024)
         
        # Normalize
        image_features = F.normalize(image_features, dim=-1) # Normalize the image features : shape (batch_size, 1024)
        text_features = F.normalize(text_features, dim=-1) # Normalize the text features : shape (batch_size, 1024)
        
        # Fuse features
        fused_features = torch.mul(image_features, text_features) # Multiply the image and text features : shape (batch_size, 1024)
        
        # Pre-output and classification
        features = self.pre_output(fused_features) # Pass the fused features through the pre-output layers which is a fully connected layer followed by a ReLU activation function and a dropout layer (feat extraction after fusion): shape (batch_size, 1024) -> (batch_size, 1024)
        logits = self.classifier(features).squeeze(dim=1) # Pass the features through the classifier which is a cosine classifier : shape (batch_size, 1024) -> (batch_size, 1)
        
        return logits
