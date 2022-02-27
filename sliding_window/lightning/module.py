import pytorch_lightning as pl
from dataset import SGDDataset
from torch.utils.data import DataLoader
from config import DEFAULT_PARAMS_BERT_WITH_POOLING

class SDGDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=DEFAULT_PARAMS_BERT_WITH_POOLING['batch_size']):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.train_dataset = SGDDataset(
            self.train_df,
            self.tokenizer,
        )

        self.test_dataset = SGDDataset(
            self.test_df,
            self.tokenizer,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )