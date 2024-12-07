import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class MemeDataset(Dataset):
    def __init__(self, cfg, root_folder, split='train'):
        super().__init__()
        self.cfg = cfg
        self.root_folder = root_folder
        self.split = split
        
        # Load and filter data
        self.df = pd.read_csv(cfg.info_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        if cfg.label == 'target':
            self.df = self.df[self.df['hate'] == 1].reset_index(drop=True)
        
        # Handle float columns
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        # print(f"{self.cfg.img_folder}/{row['name']}")
        image = Image.open(f"{self.cfg.img_folder}/{row['name']}").convert('RGB')
        image = image.resize((self.cfg.image_size, self.cfg.image_size))
        
        # Get text
        text = row['text'] if row['text'] != 'None' else 'null'
        
        return {
            'image': image,
            'text': text,
            'label': row[self.cfg.label],
            'idx_meme': row['name'],
            'origin_text': text
        }