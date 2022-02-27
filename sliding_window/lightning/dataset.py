from errno import ECONNABORTED
from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import BertTokenizer
from tokenizer import BERTTokenizerPooled

from copy import deepcopy

from config import DEFAULT_PARAMS_TOKENIZER, DEFAULT_PARAMS_DATA

# torch.multiprocessing.set_start_method('spawn')

class SGDDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame,
        tokenizer: BertTokenizer,
    ):
        self.tokenizer = BERTTokenizerPooled(tokenizer, DEFAULT_PARAMS_TOKENIZER['size'], DEFAULT_PARAMS_TOKENIZER['step'], DEFAULT_PARAMS_TOKENIZER['minimal_length'])

        self.data = data
        self.text_column = DEFAULT_PARAMS_DATA['text_column']
        self.label_columns = DEFAULT_PARAMS_DATA['label_columns']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row[self.text_column]
        labels = data_row[self.label_columns]

        encoding = self.tokenizer.preprocess(text)

        return dict(
            text=text,
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            labels=torch.FloatTensor(labels)
        )