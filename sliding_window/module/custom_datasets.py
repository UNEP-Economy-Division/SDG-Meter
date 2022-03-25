from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import BertTokenizer

class SGDDataset(Dataset):
  def __init__(
    self, 
    data: pd.DataFrame, 
    tokenizer: BertTokenizer, 
  ):
    self.tokenizer = tokenizer

    self.data = data

    self.label_columns = self.data.columns
    self.label_columns.remove('text')
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    text = data_row.text
    labels = data_row[self.label_columns]

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      text=text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )

# class TokenizedDataset(Dataset):
#     ''' Dataset for tokens with labels'''

#     def __init__(self, tokens, labels):
#         self.input_ids = tokens['input_ids']
#         self.attention_mask = tokens['attention_mask']
#         self.labels = labels.to_numpy()

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         print(idx)
#         i = self.input_ids[idx]
#         a = self.attention_mask[idx]
#         l = self.labels[idx]
#         return i, a, l


# def collate_fn_pooled_tokens(data):
#     input_ids = [data[i][0] for i in range(len(data))]
#     attention_mask = [data[i][1] for i in range(len(data))]
#     labels = [data[i][2] for i in range(len(data))]
#     collated = [input_ids, attention_mask, labels]
#     return collated
