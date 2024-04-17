import torch
import glob
import lightning.pytorch as pl
import sys
sys.path.append('/egr/research-slim/belleva1/EpiGraphReg/')

class ChromosomeDataset(torch.utils.data.Dataset):

    def __init__(self, folder_path):
        self.chromosomes = [torch.load(f) for f in glob.glob(folder_path+'*.pt')]

    def __getitem__(self, idx):
        chr = self.chromosomes[idx]
        return chr['dnase_seq'].unsqueeze(0), chr['cage_seq'].unsqueeze(0)

    def __len__(self):
        return len(self.chromosomes)

def make_data_module(batch_size, num_workers) -> pl.LightningDataModule:
    
    data_module = pl.LightningDataModule.from_datasets(
        train_dataset=ChromosomeDataset('/egr/research-slim/belleva1/EpiGraphReg/data/train_data/'),
        val_dataset=ChromosomeDataset('/egr/research-slim/belleva1/EpiGraphReg/data/val_data/'),
        test_dataset=ChromosomeDataset('/egr/research-slim/belleva1/EpiGraphReg/data/test_data/'),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return data_module