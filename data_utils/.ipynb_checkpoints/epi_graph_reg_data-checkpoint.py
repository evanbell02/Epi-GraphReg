import torch
import glob
import lightning.pytorch as pl
import sys
from pathlib import Path
sys.path.append(str(Path().parent.resolve().parent.resolve()))

class ChromosomeDataset(torch.utils.data.Dataset):

    def __init__(self, folder_path):
        self.chromosomes = [torch.load(f) for f in glob.glob(folder_path+'*.pt')]

    def __getitem__(self, idx):
        chr = self.chromosomes[idx]
        return chr['dnase_seq'].unsqueeze(0), chr['cage_seq'].unsqueeze(0), chr['edges']

    def __len__(self):
        return len(self.chromosomes)

def make_data_module(batch_size, num_workers, data_type='enhancers_only') -> pl.LightningDataModule:

    data_path = '/egr/research-slim/belleva1/EpiGraphReg/data/' + data_type
    data_module = pl.LightningDataModule.from_datasets(
        train_dataset=ChromosomeDataset(data_path + '/train_data/'),
        val_dataset=ChromosomeDataset(data_path + '/val_data/'),
        test_dataset=ChromosomeDataset(data_path + '/test_data/'),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return data_module