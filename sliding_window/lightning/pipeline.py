import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import DEFAULT_PARAMS_BERT_WITH_POOLING, VISIBLE_GPUS

class Pipeline():
    def __init__(self) -> None:
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )
        
        self.logger = TensorBoardLogger("lightning_logs", name="bert_with_pooling")
        self.early_stopping_callback = EarlyStopping(monitor='train_loss', patience=2)
        
        self.trainer = pl.Trainer(
            logger=self.logger,
            checkpoint_callback=self.checkpoint_callback,
            callbacks=[self.early_stopping_callback],
            max_epochs=DEFAULT_PARAMS_BERT_WITH_POOLING['number_of_epochs'],
            gpus=VISIBLE_GPUS,
            progress_bar_refresh_rate=30
        )
    
    def train(self, model, data_module):
        self.trainer.fit(model, data_module)
        
    def fit(self):
        self.trainer.test()
    
    def get_best_model(self):
        return self.checkpoint_callback.best_model