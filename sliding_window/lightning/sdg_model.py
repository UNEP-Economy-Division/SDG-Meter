import pandas as pd

csv_filename = 'C:\\Users\\DEEP-LEARNING\\Desktop\\SDG Git\\SDG-Meter\\sliding_window\\data\\data_sample.csv'
df = pd.read_csv(csv_filename)

from module import SDGDataModule
data_module = SDGDataModule(df)

from config import DEFAULT_PARAMS_BERT_WITH_POOLING
from model import SDGTagger
from pipeline import Pipeline

N_EPOCHS = DEFAULT_PARAMS_BERT_WITH_POOLING['num_epochs']
BATCH_SIZE = DEFAULT_PARAMS_BERT_WITH_POOLING['batch_size']

steps_per_epoch= int(len(df) * 0.8) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5

model = SDGTagger(n_warmup_steps=warmup_steps,
                n_training_steps=total_training_steps )
ppln = Pipeline(data_module)
print(ppln.train(model))

import torch
def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()