import torch
import csv

MODEL_LOAD_FROM_FILE = False
MODEL_PATH = "pretrained_models/roberta"
VISIBLE_GPUS = "1"

DEFAULT_PARAMS_BERT_WITH_POOLING = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size' : 6,
    'learning_rate' : 5e-6,
    'pooling_strategy': 'mean', # options: ['mean','max']
    'size': 510,
    'step': 256,
    'minimal_length': 1,
    'num_labels':  sum(1 for row in csv.reader('../data/labels.csv'))
}