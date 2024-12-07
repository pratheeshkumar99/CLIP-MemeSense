import torch
from tqdm import tqdm
from utils.metrics import MemeMetrics, MetricTracker

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.cfg = cfg
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metrics = MemeMetrics(cfg.num_classes, cfg.device)
        self.tracker = MetricTracker()
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move to device
            image_features = batch['image_features'].to(self.cfg.device)
            text_features = batch['text_features'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(image_features, text_features)
            
            # Compute loss and metrics
            loss = self.criterion(logits, labels)
            metrics = self.metrics.compute(logits, labels)
            metrics['loss'] = loss.item()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            self.tracker.update('train', metrics)
    
    def validate(self):
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                image_features = batch['image_features'].to(self.cfg.device)
                text_features = batch['text_features'].to(self.cfg.device)
                labels = batch['labels'].to(self.cfg.device)
                
                # Forward pass
                logits = self.model(image_features, text_features)
                
                # Compute loss and metrics
                loss = self.criterion(logits, labels)
                metrics = self.metrics.compute(logits, labels)
                metrics['loss'] = loss.item()
                
                # Track metrics
                self.tracker.update('val', metrics)
    
    def train(self):
        for epoch in range(self.cfg.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.cfg.max_epochs}")
            
            # Training
            self.train_epoch()
            train_metrics = self.tracker.get_epoch_metrics('train')
            
            # Validation
            self.validate()
            val_metrics = self.tracker.get_epoch_metrics('val')
            
            # Print metrics
            print("\nTraining Metrics:")
            for k, v in train_metrics.items():
                print(f"{k}: {v:.4f}")
            
            print("\nValidation Metrics:")
            for k, v in val_metrics.items():
                print(f"{k}: {v:.4f}")
            
            # Save checkpoint if best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(f"{self.cfg.checkpoint_path}/best_model.pt")
            
            # Reset metrics
            self.tracker.reset()
            self.metrics.reset()
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
        }
        torch.save(checkpoint, path)
