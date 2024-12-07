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
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Initialize components
        self.img_adapter = Adapter(cfg.map_dim, 4).to(self.clip_model.dtype)
        self.text_adapter = Adapter(cfg.map_dim, 4).to(self.clip_model.dtype)
        
        self.image_map = LinearProjection(
            cfg.unmapped_dim, 
            cfg.map_dim,
            cfg.num_mapping_layers, 
            cfg.drop_probs
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
            feat_dim=cfg.map_dim,
            num_classes=cfg.num_classes,
            scale=cfg.scale,
            dtype=self.clip_model.dtype
        )
        
        self.init_head_text_feat()

    def _build_pre_output_layers(self):
        layers = [nn.Dropout(p=self.cfg.drop_probs[1])]
        if self.cfg.num_pre_output_layers >= 1:
            layers.extend([
                nn.Linear(self.cfg.map_dim, self.cfg.map_dim),
                nn.ReLU(),
                nn.Dropout(p=self.cfg.drop_probs[2])
            ])
        return nn.Sequential(*layers)

    def init_head_text_feat(self):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in self.cfg.class_names]
        prompts = clip.tokenize(prompts, context_length=77, truncate=True).to(self.cfg.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
            text_features = text_features @ self.clip_model.visual.proj.t()
            text_features = F.normalize(text_features, dim=-1)
            self.classifier.apply_weight(text_features)

    def forward(self, image_features, text_features):
        # Project features
        image_projection = self.image_map(image_features)
        text_projection = self.text_map(text_features)
        
        # Apply adapters
        image_adapted = self.img_adapter(image_projection)
        text_adapted = self.text_adapter(text_projection)
        
        # Residual connections
        image_features = self.cfg.ratio * image_adapted + (1 - self.cfg.ratio) * image_projection
        text_features = self.cfg.ratio * text_adapted + (1 - self.cfg.ratio) * text_projection
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Fuse features
        fused_features = torch.mul(image_features, text_features)
        
        # Pre-output and classification
        features = self.pre_output(fused_features)
        logits = self.classifier(features).squeeze(dim=1)
        
        return logits
