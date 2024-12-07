import torch
from torch.utils.data import DataLoader
from models.memeclip import MemeCLIP
from data.dataset import MemeDataset
from data.collator import MemeCollator
from train import Trainer
from configs import cfg

def main():
    # Set random seed
    torch.manual_seed(cfg.seed)
    
    # Create datasets
    train_dataset = MemeDataset(cfg, cfg.root_dir, split='train')
    val_dataset = MemeDataset(cfg, cfg.root_dir, split='val')
    test_dataset = MemeDataset(cfg, cfg.root_dir, split='test')
    
    # Create collator
    collator = MemeCollator(cfg)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator
    )
    
    # Create model
    model = MemeCLIP(cfg).to(cfg.device)

    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        cfg=cfg
    )
    
    # Train model
    if not cfg.test_only:
        trainer.train()
    
    # Test model
    if cfg.test_only:
        # Load best model
        checkpoint = torch.load(cfg.checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Run evaluation
        trainer.validate_loader(test_loader, "Test")

if __name__ == '__main__':
    main()