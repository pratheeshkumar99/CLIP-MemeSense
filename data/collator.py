import torch
import clip

class MemeCollator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_model, self.clip_preprocess = clip.load(
            cfg.clip_variant, 
            device=cfg.device, 
            jit=False
        )
        self.clip_model.float().eval()

    def __call__(self, batch):
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = [item['idx_meme'] for item in batch]
        
        image_embed_list = []
        text_embed_list = []
        
        for item in batch:
            pixel_values = self.clip_preprocess(item['image']).unsqueeze(0)
            text = clip.tokenize(item['text'], context_length=77, truncate=True)
            
            image_features, text_features = self.compute_clip_features(
                pixel_values.to(self.cfg.device),
                text.to(self.cfg.device)
            )
            
            image_embed_list.append(image_features.cpu().detach())
            text_embed_list.append(text_features.cpu().detach())
        
        return {
            'image_features': torch.cat(image_embed_list, dim=0),
            'text_features': torch.cat(text_embed_list, dim=0),
            'labels': labels,
            'idx_memes': idx_memes
        }

    def compute_clip_features(self, img_input, text_input):
        with torch.no_grad():
            image_features = self.clip_model.visual(img_input)
            text_features = self.encode_text(text_input)
        return image_features, text_features

    def encode_text(self, text_input):
        x = self.clip_model.token_embedding(text_input)
        x = x + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        return x[torch.arange(x.shape[0]), text_input.argmax(dim=-1)]