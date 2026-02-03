import os
import sys
import pickle
import yaml
import json
import torch

import numpy as np
import pandas as pd

from typing import Dict, List
from pathlib import Path
from collections import defaultdict

from datasets import Dataset
from lightning import LightningModule
from transformers import BertTokenizer, AutoTokenizer, AutoModel

from torch.utils.data import DataLoader

from src.utils.search import GreedySearch

def embed_target_masked(
    data: pd.DataFrame, 
    model: AutoModel, 
    tokenizer: AutoTokenizer,
    device: str = 'cpu',
    batch_size: int = 1
) -> pd.DataFrame:
    """Embedd molecules helper; target is always masked.

    Args:
        data (pd.DataFrame): dataframe containing targets, smiles, text and task description.
        model (AutoModel): pre-trained model.
        tokenizer (AutoTokenizer): tokenizer.
        device (str, optional): cpu or cuda; defaults to cpu.
        

    Returns:
        pd.DataFrame: input data with additional embeddings column.
    """
    from src.data import PromptPropertyCollator
    
    model.eval()

    collator = PromptPropertyCollator(tokenizer, device)

    ds = Dataset.from_dict({"text": data['text'].values.tolist()})
    ds = ds.map(lambda b: tokenizer.encode(b), batched=True)
    dl = DataLoader(ds, collate_fn=collator, batch_size=batch_size)

    embeddings = []

    for b in dl:
        model_output = model(b['input_ids'], b['attention_mask'], output_hidden_states=True)
        emb = model_output['hidden_states'][-1].mean(axis=0).cpu().detach().tolist()

        for e, t in zip(emb, b['text']):
            embeddings.append({'text': t, 'embeddings': e})

    embeddings = pd.DataFrame(embeddings)

    return pd.merge(embeddings, data, on='text', how='inner')


def load_latest_checkpoint(cnf: Dict, tokenizer: BertTokenizer, root:str) -> LightningModule:
    """
    Load the latest checkpoint for current model.

    Args:
        cnf (str): model config file.
        tokenizer (obj:`ExpressionBertTokenizer`): Tokenizer
        root (str): current root models dir
    """
    from src import model

    pth = Path(root) / Path(cnf['model_name']) / Path('checkpoints')
    last = ''
    epoch = -1

    for f in os.listdir(pth):
        if 'last' in f:
            continue
        e = int(f.split('-')[1])
        if e > epoch:
            epoch = e
            last = f

    chkp = Path(pth) / Path(last)

    if cnf['model'] == 'big_boy':
        m = model.BigBoy.load_from_checkpoint(chkp, **cnf, tokenizer=tokenizer, root_dir=root)
    else:
        m = model.RTPlussModule.load_from_checkpoint(chkp, **cnf, tokenizer=tokenizer, root_dir=root)
    return m, chkp

def load_best_checkpoint(dir: str, metric: str) -> str:
    """Load the best model based on metric
    
    Args:
        dir (str): model location.
        metric (str): metric we want to load from.
    
    Return:
        str: path to the best model.
    """

    root = Path(dir) / Path('checkpoints')

    best_path = ''
    best_val = sys.float_info.max
    if metric == 'roc_auc':
        best_val = sys.float_info.min

    for m in os.listdir(root):
        if metric not in m:
            continue

        val = float(m.split('=')[1][:-5])

        if metric == 'roc_auc':
            if val > best_val:
                best_val = val
                best_path = m
        else:
            if val < best_val:
                best_val = val
                best_path = m
    return root / Path(best_path)

def postprocess(val: Dict) -> Dict:
    """Post process a list of dicts to dict of lists"""
    post = defaultdict(list)

    for v in val:
        for k in v:
            post[k] += v[k]
        
    return post

def process_val_tensor(val: list) -> torch.Tensor:
    """Process the list on tensors to singular, same length tensor.

    Args:
        val (list): List of tensors.
    
    Returns:
        torch.Tensor: combined tensors.
    """
    batch = []
    for v in val:
        for t in v:
            batch.append(t)

    max_length = max(x.size(0) for x in batch)
    result = batch[0].new_full([len(batch), max_length], 0)
    if batch[0].shape[-1] == 507:
        result = batch[0].new_full([len(batch), max_length, 507], 0)
    for i, example in enumerate(batch):
        if batch[0].shape[-1] == 507:
            result[i, : example.shape[0], :] = example
        else:
            result[i, : example.shape[0]] = example
    return result

def process_outs(outs: list) -> list:
    """Process properties from predictions for new model.

    Args:
        trues (list): List of input ids corresponding to the original property.
        preds (list): Predictions for the properties.
        tokenizer (object): Used tokenizer to decode.
    """
    search = GreedySearch()
    
    processed = []

    for o in outs:
        pred = search(o.unsqueeze(0)).squeeze(0)

        processed.append(pred)

    processed = torch.stack(processed, dim=0)

    return processed

def process_tensor(lot: List) -> torch.Tensor:
    """Put the tensors to the same lengths and concat batches.

    Args:
        lot (list): list of tensors.

    Returns:
        torch.Tensor: returns finalized tensor.
    """
    
    final = []
    if len(lot[0].shape) == 2:
        max_len = max([t.shape[1] for t in lot])
        
        for t in lot:
            z = torch.zeros(t.shape[0], max_len)
            z[:, :t.shape[1]] = t
            final.append(z)
    else:
        max_len = max([t.shape[1] for t in lot])

        for t in lot:
            z = torch.zeros(t.shape[0], max_len, t.shape[-1])
            z[:, :t.shape[1], :] = t
            final.append(z)

    final = torch.cat(final, dim=0)
    return final

def fetch_norm_params(dataset_dir: str) -> Dict:
    """Get the normalization params."""
    norm_params_pth = Path(dataset_dir) / Path('normalization_params.json')

    with open(norm_params_pth, 'r') as f:
        norm_params = json.load(f)

    return norm_params

def to_dataset(
    molecules: List,
    tasks: str | List,
    targets: List
) -> pd.DataFrame:
    """Convert the input molecules to model format.

    Args:
        molecules (List): one or more SMILES string.
        tasks (str | List): one or more tasks, if there are multiple.
            molecules same task is used for all of them.
        targets (List): list of corresponding targets.

    Returns:
        pd.DataFrame: 
    """
    
    if isinstance(tasks, str):
        tasks = [tasks] * len(molecules)
    
    assert len(molecules) == len(targets), "Number of targets and molecules not equal."
    assert len(molecules) == len(tasks), "Number of tasks and molecules not equal."

    data = []
    
    for m, td, tgt in zip(molecules, tasks, targets):
        text = f'{td} | {tgt:.4f} | {m}'

        data.append({'smiles': m, 'description': td, 'target': tgt, 'text': text})

    return pd.DataFrame(data)