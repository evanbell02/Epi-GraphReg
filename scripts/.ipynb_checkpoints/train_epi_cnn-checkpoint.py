import sys
from pathlib import Path
sys.path.append(str(Path().parent.resolve()))
import lightning.pytorch as pl
import torch
import models
import data_utils
from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI(
        models.epi_cnn.EpiCNN,
        data_utils.epi_cnn_data.make_data_module
    )

if __name__ == "__main__":
    cli_main()
