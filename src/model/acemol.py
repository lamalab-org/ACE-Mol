import json

from typing import Any, Dict, Union, Optional, List
from collections import defaultdict

from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adafactor
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule
from transformers import ModernBertForMaskedLM

import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from src import utils

from .wrapped_model import WrappedModel

class ACEMol(LightningModule):
    """New and improved model"""

    def __init__(
        self,
        model_name: str,
        parameters: Dict,
        optimizer_params: Dict,
        scheduler_params: Dict,
        root_dir: str,
        optimizer_type:str = 'adam',
        pre_trained: str = None,
        metric: bool = False,
        task: str = None,
        freeze: bool = None,
        **kwargs
    ) -> None:
        """Model constructor.

        Args:
            model_name (str): Name of our model.
            parameters (:obj:`Dict`): Model parameters.
            optimizer_params (:obj:`Dict`): Parameters to set up the optimizer
            scheduler_params (:obj:`Dict`): Parameters to set up the scheduler
            root_dir (str): Root directory.
            optimizer_type(str, optional): Define the optimizer to use, defaults to adam.
            pre_trained (str, optional): Start training from pretrained model.
            metrics (bool): Specify if you want to use the metrics druign val.
        """
        super().__init__()

        self.net = WrappedModel()

        self.bce_loss = CrossEntropyLoss(reduction="none")

        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.optimizer_type = optimizer_type

        self.search = utils.GreedySearch()

        if pre_trained:
            self.load_pretrained(pre_trained)

        self.reset_metrics()

        self.test_root = Path(root_dir) / Path(model_name)

        if freeze is not None:
            for name, param in self.net.named_parameters():
                if 'layers' in name and '21' not in name:
                    param.requres_grad = False

        self.metric = metric
        self.task = task

            
    def load_pretrained(self, pre_trained: str) -> None:
        """Load pre-trained model.
        Makes sure just to load weigths from pl checkpoint
        
        Args:
            pre_trained (str): Path to pre-trained model (pl checkpoint).
        """
        if isinstance(pre_trained, str) and pre_trained.startswith('jablonkagroup'):
            self.net.model = ModernBertForMaskedLM.from_pretrained(pre_trained)
        else:
            state_dict = torch.load(pre_trained, map_location='cpu')
            state_dict = state_dict['state_dict']
            old_keys = list(state_dict.keys())
            for k in old_keys:
                n_k = k.split('.')[1:]
                n_k = '.'.join(n_k)
                state_dict[n_k] = state_dict.pop(k)
            self.net.load_state_dict(state_dict)

    def reset_metrics(self) -> None:
        self.val_metrics = dict()
        self.val_metrics['property'] = defaultdict(list)
        self.val_metrics['molecule'] = defaultdict(list)

        self.test_metrics = dict()
        self.test_metrics['property'] = defaultdict(list)
        self.test_metrics['molecule'] = defaultdict(list)

        self.total_rmse = []
        self.rmse = None
        self.accuracy = None

    def forward(
        self, 
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        return self.net(inputs)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Set up optimizer and scheduler."""

        # Set up optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        weight_decay = self.optimizer_params.pop('weight_decay')
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.net.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.net.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
        ]


        if self.optimizer_type == 'adam':
            optimizer = AdamW(
                optimizer_grouped_parameters,
                **self.optimizer_params
            )
        elif self.optimizer_type == 'adafactor':
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                **self.optimizer_params
            )
        
        # Set up the scheduler
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = self.scheduler_params['warmup_steps']
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        scheduler = LambdaLR(optimizer, lr_lambda, -1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Run a training step for model"""
        outs = self.net(batch)
        loss = outs[0]

        if 'mode' in batch.keys() and batch['mode'] == 'molecule':
            self.log('train/reconstructio_loss', loss.detach().item())
        else:
            self.log('train/property_loss', loss.detach().item())
        scheduler = self.lr_schedulers()
        scheduler.step()

        self.log('train/loss', loss.detach().item())

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outs = self.net(batch)
        loss = outs[0]

        if 'mode' in batch.keys() and batch['mode'] == 'molecule':

            if self.metric:
                self.val_metrics['molecule']['text'] += batch['text']
                self.val_metrics['molecule']['inputs'].append(batch['input_ids'].detach().cpu())
                self.val_metrics['molecule']['logits'].append(utils.process_outs(outs[1].detach().cpu()))
                self.val_metrics['molecule']['labels'].append(batch['labels'].detach().cpu())

            self.log('val/reconstructio_loss', loss.detach().item())
        else:
            if self.metric:
                self.val_metrics['property']['text'] += batch['text']
                self.val_metrics['property']['inputs'].append(batch['input_ids'].detach().cpu())
                self.val_metrics['property']['logits'].append(utils.process_outs(outs[1].detach().cpu()))
                self.val_metrics['property']['labels'].append(batch['labels'].detach().cpu())

            self.log('val/property_loss', loss.detach().item())

        self.log('val/loss', loss.mean().detach().item())

        return loss

    def on_validation_epoch_end(self) -> None:
        """Calculate the RMSE and Spearman on validation predictions."""

        if not self.metric:
            return

        self.val_metrics['property']['inputs'] = utils.process_tensor(self.val_metrics['property']['inputs'])
        self.val_metrics['property']['logits'] = utils.process_tensor(self.val_metrics['property']['logits'])
        self.val_metrics['property']['labels'] = utils.process_tensor(self.val_metrics['property']['labels'])

        if len(self.test_metrics['molecule']['inputs']) > 0:
            self.val_metrics['molecule']['inputs'] = utils.process_tensor(self.val_metrics['molecule']['inputs'])
            self.val_metrics['molecule']['logits'] = utils.process_tensor(self.val_metrics['molecule']['logits'])
            self.val_metrics['molecule']['labels'] = utils.process_tensor(self.val_metrics['molecule']['labels'])

        processed = utils.process_val_prompt(self.val_metrics)

        processed['property']['tasks'] = processed['property']['prompts']

        ress = []

        for task, true, pred in zip(processed['property']['tasks'], processed['property']['trues'], processed['property']['preds']):
            ress.append((task, true, pred))

        df = pd.DataFrame(ress, columns=['task', 'true', 'pred'])
        df = df.drop(df[df['true'].str.contains('nan')].index)
        df = df.drop(df[df['pred'].str.contains('nan')].index)
        df["true"] = pd.to_numeric(df["true"])
        df["pred"] = pd.to_numeric(df["pred"], errors='coerce')

        self.reset_metrics()

        if self.metric == 'roc_auc':
            df = df.dropna()
            roc = roc_auc_score(df["true"], df["pred"])
            
            self.log('val/roc_auc', roc)
            return {
                'roc_auc': roc
            }
        
        if self.metric == 'rmse':
            df = df[df['task'] == self.task]
            df = df.dropna()

            rmse = root_mean_squared_error(df["true"], df["pred"])
            self.log('val/rmse', rmse)

            return {
                'rmse': rmse
            } 

        return {}

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outs = self.net(batch)
        loss = outs[0]

        if 'mode' in batch.keys() and batch['mode'] == 'molecule':
            self.log('test/reconstructio_loss', loss.detach().item())
        else:
            self.log('test/property_loss', loss.detach().item())

        self.log('test/loss', loss.mean().detach().item())

        return loss