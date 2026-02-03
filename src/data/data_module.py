import os
from glob import glob

from typing import Dict, Union, Any
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from lightning import LightningDataModule
from transformers import (
    LineByLineTextDataset, 
    TextDataset, 
    DataCollatorForPermutationLanguageModeling
)

from torch.utils.data import DataLoader
from datasets import load_dataset

from src.tokenizer import BertSmilesTokenizer
from src.data.alternating_collator import AlternatingCollator
from src.data.prompt_collators import PromptPropertyCollator, PromptMoleculeCollator

class DataModule(LightningDataModule):
    def __init__(
        self,
        eval_data_file: str,
        val_data_file: str,
        train_data_file: str,
        collators: Dict,
        alternating: bool = True,
        steps: int = 8,
        batch_size: int = 16,
        task: str = 'property',
        cache_dir: str = None,
        device: str = 'cpu',
    ) -> None:
        """Initialize the data module.
        
        Args:
            eval_data_file (str): Path to the evaluation data.
            val_data_file (str): Path to the validation data.
            train_data_file (str): Path to the train file.]
            collators (:obj:`Dict`): Configs for all the collators.
            alternating (bool, optional): Sets up alternating training. Defaults to True.
            steps (:obj:`Dict`, optional): Number of steps to switch the alternative training.
            task (property, optional): Used to set up the collator for the specified task.
            batch_size (int, optional): Batch size, Defaults to 16.
            cache_dir (str, optional): Cache dir path, defaults to None.
            device (str, optional): Device for torch, defaults to cpu.
        """

        super().__init__()

        self.train_data_file = train_data_file
        self.eval_data_file = eval_data_file
        self.val_data_file = val_data_file
        self.batch_size = batch_size
        self.steps = steps
        self.collators = collators
        self.alternating = alternating
        self.task = task
        self.device = device
        self.cache_dir = cache_dir
        self.cached_file = '_'.join(train_data_file.split('/'))

        self.cache = Path(cache_dir) / Path(self.cached_file)

        self.tokenizer = BertSmilesTokenizer()

        self.train_dataset = None
        self.val_dataset = None
        self.eval_dataset = None


    def get_cached(self) -> Dict[str, str]:
        """Find cached files so we don't have to map each time.
        
        Returns:
            Dict[str, str]: split: location combination of the cached files.
        """

        files = []
        pattern   = "*.arrow"

        for d,_,_ in os.walk(self.cache):
            files.extend(glob(os.path.join(d,pattern)))

        dct = {'train': None, 'test': None, 'validation': None}

        for f in files:
            if 'mapped' not in f:
                f = f.rsplit('.', 1)
                f = f[0] + '-mapped.' + f[1]

            tmp = f.split('/')[-1]
            if 'train' in tmp:
                dct['train'] = f
            if 'test' in tmp:
                dct['test'] = f
            if 'validation' in tmp:
                dct['validation'] = f
        return dct


    def prepare_data(self) -> None:
        """Setup the data lodaers and collators before training."""

        if self.task == 'smiles':
            self.collator = PromptMoleculeCollator(tokenizer=self.tokenizer, device=self.device, **self.collators['smiles'])
        elif self.task == 'smilesprop':
            self.collator = PromptPropertyCollator(tokenizer=self.tokenizer, device=self.device, **self.collators['sentence'])

        if self.alternating and self.task == 'smiles':
            prompt_prop = PromptPropertyCollator(tokenizer=self.tokenizer, device=self.device, **self.collators['sentence'])
            self.collator = AlternatingCollator(
                collator = {'collator': prompt_prop, 'name': 'property'},
                alter = {'collator': self.collator, 'name': 'molecule'},
                steps = self.steps
            )
            

        train_ds = Path(self.train_data_file)
        train_ds = list(train_ds.iterdir())
        train_ds = [str(p) for p in train_ds]

        test_ds = Path(self.eval_data_file)
        test_ds = list(test_ds.iterdir())
        test_ds = [str(p) for p in test_ds]

        val_ds = Path(self.val_data_file)
        val_ds = list(val_ds.iterdir())
        val_ds = [str(p) for p in val_ds]

        dataset = load_dataset("text", data_files={"train": train_ds, 'test': test_ds, 'validation': val_ds}, cache_dir=self.cache)
        
        cahced_f = self.get_cached()

        self.train_dataset = dataset['train'].map(lambda b: self.tokenizer(b), batched=True, load_from_cache_file=True, cache_file_name=cahced_f['train'], num_proc=1)
        self.val_dataset = dataset['validation'].map(lambda b: self.tokenizer(b), batched=True, load_from_cache_file=True, cache_file_name=cahced_f['validation'], num_proc=1)
        self.eval_dataset = dataset['test'].map(lambda b: self.tokenizer(b), batched=True, load_from_cache_file=True, cache_file_name=cahced_f['test'], num_proc=1)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = self.collator
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            collate_fn = self.collator
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size = self.batch_size,
            collate_fn = self.collator
        )