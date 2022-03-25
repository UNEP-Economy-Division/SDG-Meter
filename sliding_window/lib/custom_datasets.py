from torch.utils.data import Dataset


class TextDataset(Dataset):
    ''' Dataset for raw texts with labels'''

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels.to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class TokenizedDataset(Dataset):
    ''' Dataset for tokens with labels'''

    def __init__(self, tokens, labels):
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = labels.to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        print(idx)
        i = self.input_ids[idx]
        a = self.attention_mask[idx]
        l = self.labels[idx]
        return i, a, l


def collate_fn_pooled_tokens(data):
    input_ids = [data[i][0] for i in range(len(data))]
    attention_mask = [data[i][1] for i in range(len(data))]
    labels = [data[i][2] for i in range(len(data))]
    collated = [input_ids, attention_mask, labels]
    return collated
