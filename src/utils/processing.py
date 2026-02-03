from typing import Dict

import torch
import numpy as np


def process_results_prompt(results: Dict) -> Dict:
    """Extract property, prediction, and real molecule."""

    from src.tokenizer import BertSmilesTokenizer

    tokenizer = BertSmilesTokenizer()

    sep = tokenizer.bert_tokenizer.sep_token_id
    
    processed = dict()
    
    for k in results.keys():
        molecules = {'trues': [], 'preds': [], 'prompts': []}
        props = {'trues': [], 'preds': [], 'prompts': []}
    
        for res in results[k]:

            ### process property
            for prop in res['property']['text']:
                prop = prop.split(' | ')
                props['prompts'].append(prop[0])
                props['trues'].append(prop[1])

            for pred, mask in zip(res['property']['logits'], res['property']['labels']):
                mask = (mask != -100) & (mask != 0)
                pred = pred[mask]
                pred = tokenizer.decode(pred.astype(int))
                props['preds'].append(pred)

            if 'molecule' not in res.keys():
                continue

            ### process molecule
            for mol in res['molecule']['text']:
                mol = mol.split(' | ')
                molecules['prompts'].append(mol[0])
                molecules['trues'].append(mol[2])

            
            for mol, pred in zip(res['molecule']['inputs'], res['molecule']['logits']):
                mol_idxes = list(np.where(mol == sep)[0][1:])
                pred = pred[mol_idxes[0]+1: mol_idxes[1]]
                pred = tokenizer.decode(pred.astype(int))
                molecules['preds'].append(pred)

        processed[k] = dict()
        processed[k]['molecule'] = molecules
        processed[k]['property'] = props

    return processed


def process_val_prompt(results: Dict) -> Dict:
    """Extract property, prediction, and real molecule."""

    from src.tokenizer import BertSmilesTokenizer

    tokenizer = BertSmilesTokenizer()

    sep = tokenizer.bert_tokenizer.sep_token_id
    
    processed = dict()
    
    molecules = {'trues': [], 'preds': [], 'prompts': []}
    props = {'trues': [], 'preds': [], 'prompts': []}

    for prop in results['property']['text']:
        prop = prop.split(' | ')
        props['prompts'].append(prop[0])
        props['trues'].append(prop[1])

    for pred, mask in zip(results['property']['logits'], results['property']['labels']):
        mask = (mask != -100) & (mask != 0)
        pred = pred[mask]
        pred = tokenizer.decode(pred.type(torch.int32))
        props['preds'].append(pred)

    processed = dict()
    processed['molecule'] = molecules
    processed['property'] = props

    return processed