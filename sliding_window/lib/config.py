MODEL_LOAD_FROM_FILE = True
MODEL_PATH = "pretrained_models/roberta"
VISIBLE_GPUS = ""

DEFAULT_PARAMS_BERT_WITH_POOLING = {
    'device' : 'cuda',
    'batch_size' : 6,
    'learning_rate' : 5e-6,
    'pooling_strategy': 'mean', # options: ['mean','max']
    'size': 510,
    'step': 256,
    'minimal_length': 1
}