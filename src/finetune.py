import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import logging
from datetime import datetime
from typing import Dict

import torch

from transformers import set_seed
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.model import ACEMol
from src.parser import parse_config
from src.utils import setup_model_dir, load_latest_checkpoint
from src.data import DataModule

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')

def main():
    """Setup and start training."""
    logging.basicConfig(level=logging.INFO)

    config = parse_config()
    
    model_cnf, data_cnf, train_cnf, base_cnf = config['model'], config['data'], config['train'], config['base']

    if model_cnf['parameters']['torch_dtype'] == 'float32':
        torch.set_float32_matmul_precision('high')

    logger.info(f"Setting seed {train_cnf['seed']}")
    set_seed(train_cnf['seed'])

    device = train_cnf['trainer']['accelerator']

    chkpt_available:bool = setup_model_dir(base_cnf['models_dir'], model_cnf['model_name'], config['cnf_file'], config)

    dm = DataModule(**data_cnf, device=device)
    dm.prepare_data()
    dm.setup(stage="fit")

    if chkpt_available:
        model, chkpt_pth = load_latest_checkpoint(model_cnf, dm.tokenizer, base_cnf['models_dir'])
    else:
        model = ACEMol(**model_cnf, tokenizer=dm.tokenizer, root_dir=base_cnf['models_dir'])

    model_root = Path(base_cnf['models_dir']) / Path(model_cnf['model_name'])
    callbacks = []

    save_name = "model-{epoch:02d}-val_loss={val/loss:.4f}"
    save_mode = 'min'
    monitor = 'val/loss'

    if model_cnf['metric'] == 'roc_auc':
        save_name = "model-{epoch:02d}-roc_auc={val/roc_auc:.4f}"
        save_mode = 'max'
        monitor = 'val/roc_auc'
    if model_cnf['metric'] == 'rmse':
        save_name = "model-{epoch:02d}-rmse={val/rmse:.4f}"
        save_mode = 'min'
        monitor = 'val/rmse'

    callbacks.append(ModelCheckpoint(
        dirpath=model_root / Path('checkpoints'), 
        save_top_k=1, 
        monitor=monitor,
        save_last=False,
        filename=save_name,
        mode=save_mode,
        auto_insert_metric_name=False)
    )      

    timestamp = str(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
    name = model_cnf['model_name']
    id = name + " " + timestamp
    if chkpt_available:
        id = 'ckp_' + id

    trainer = Trainer(**train_cnf['trainer'], default_root_dir=model_root, callbacks=callbacks)

    if chkpt_available:
        trainer.fit(model, datamodule=dm, ckpt_path=chkpt_pth)
    else:
        trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()