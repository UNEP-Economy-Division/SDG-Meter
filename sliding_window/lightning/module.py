import pytorch_lightning as pl
import torch
from dataset import SGDDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from tokenizer import BERTTokenizerPooled
from sentence_tokenizer import SentenceProcessor

from config import DEFAULT_PARAMS_BERT_WITH_POOLING

random_state=None
N_WORKERS = 0

class SDGDataModule(pl.LightningDataModule):
    def __init__(
                self, df, 
                train_size=0.8,
                batch_size=DEFAULT_PARAMS_BERT_WITH_POOLING['batch_size']
                ):
        super().__init__()
        self.tokenizer = SentenceProcessor(AutoTokenizer.from_pretrained(DEFAULT_PARAMS_BERT_WITH_POOLING['bert_model']))
        self.batch_size = batch_size
        self.train_df, test_df = train_test_split(df, train_size=train_size, random_state=random_state)
        self.test_df, self.val_df = train_test_split(test_df, train_size=0.5, random_state=random_state)

    def setup(self, stage):
        self.train_dataset = SGDDataset(
            self.train_df,
            self.tokenizer
        )

        self.val_dataset = SGDDataset(
            self.val_df,
            self.tokenizer
        )

        self.test_dataset = SGDDataset(
            self.test_df,
            self.tokenizer
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=N_WORKERS,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=N_WORKERS,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=N_WORKERS,
            collate_fn=collate_fn
        )

def collate_fn(data):
    input_ids = [data[i]['input_ids'] for i in range(len(data))]
    attention_mask = [data[i]['attention_mask'] for i in range(len(data))]
    labels = [data[i]['labels'] for i in range(len(data))]
    return [input_ids, attention_mask, labels]


def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()