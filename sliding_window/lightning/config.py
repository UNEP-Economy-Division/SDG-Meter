MODEL_LOAD_FROM_FILE = True
MODEL_PATH = "pretrained_models/roberta"
NUM_GPUS = 1
# BERT_MODEL_NAME = 'roberta-base'

DEFAULT_PARAMS_TOKENIZER = {
    'size': 510,
    'step': 255,
    'minimal_length': 10,
    'max_text_length': 510
}

DEFAULT_PARAMS_BERT_WITH_POOLING = {
    'bert_model' : 'distilroberta-base',
    'device' : 'cuda',
    'num_gpus' : 1,
    'batch_size' : 16,
    'learning_rate' : 2e-6,
    'pooling_strategy': 'max', # options: ['mean','max']
    'num_epochs': 10,
}

DEFAULT_PARAMS_DATA = {
    'text_column': 'text',
    'label_columns': [
                        'SDG1',
                        'SDG2',
                        'SDG3',
                        'SDG4',
                        'SDG5',
                        'SDG6',
                        'SDG7',
                        'SDG8',
                        'SDG9',
                        'SDG10',
                        'SDG11',
                        'SDG12',
                        'SDG13',
                        'SDG14',
                        'SDG15',
                        'SDG16',
                        'SDG17'
                    ]
}