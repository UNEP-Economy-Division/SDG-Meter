
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import DEFAULT_PARAMS_BERT_WITH_POOLING, NUM_GPUS

class Pipeline():
    def __init__(self, data_module=None) -> None:
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=False,
            monitor="val_loss",
            mode="min"
        )
        
        self.logger = TensorBoardLogger("lightning_logs", name="sdg_classifier")
        self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        
        self.trainer = pl.Trainer(
            logger=self.logger,
            enable_checkpointing=self.checkpoint_callback,
            callbacks=[self.early_stopping_callback],
            max_epochs=DEFAULT_PARAMS_BERT_WITH_POOLING['num_epochs'],
            accelerator="auto",
            gpus=NUM_GPUS,
            auto_select_gpus=bool(NUM_GPUS > 0),
            auto_lr_find=False
        )

        self.data_module = data_module
        
    
    def train(self, model, data_module=None):
        if data_module is None:
            data_module = self.data_module
        assert(data_module is not None)
        # self.trainer.tune(model, data_module)
        self.trainer.fit(model, data_module)
        
    def fit(self, data_module=None):
        if data_module is None:
            data_module = self.data_module
        assert(data_module is not None)
        
        self.trainer.test(data_module)
    
    def get_best_model(self):
        return self.checkpoint_callback.best_model