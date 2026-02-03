import numpy as np

from typing import Dict, Union
from collections import defaultdict

import torch
from scipy.stats import pearsonr
from transformers import BertTokenizer

from src.utils.search import GreedySearch, BeamSearch


def rmse(x:list, y:list) -> float:
    """Calculate root mean squared error.
    
    Args:
        x (list): list of floats.
        y (list): list of floats.

    Returns:
        float: rmse.
    """

    return np.sqrt(sum((np.array(x) - np.array(y)) ** 2) / len(x))

def re_norm(val:Union[float, list], norm_min:float, norm_max:float) -> Union[float, list]:
    """ReNormalize the MinMax norm."""
    if type(val) == float:
        return val * (norm_max - norm_min) + norm_min

    re_normed = []
    for v in val:
        re_normed.append(v * (norm_max - norm_min) + norm_min)
    return re_normed

def property_from_pred(trues: list, preds: list, tokenizer: object) -> list:
    """Process properties from predictions for new model.

    Args:
        trues (list): List of input ids corresponding to the original property.
        preds (list): Predictions for the properties.
        tokenizer (object): Used tokenizer to decode.
    """
    search = GreedySearch()
    
    new_trues = []
    new_preds = []

    for t, p in zip(trues, preds):
        p = torch.tensor(p)
        pred = search(p.unsqueeze(0)).squeeze(0)

        pred = tokenizer.decode(pred).replace(' ', '')
        tru = tokenizer.decode(t).replace(' ', '')

        # if we did not generate a number default to -1
        try:
            pred = float(pred)
        except:
            pred = -1

        tru = float(pred)

        new_trues.append(tru)
        new_preds.append(pred)

    return {'trues': new_trues, 'pred': new_preds}

def molecule_from_pred(trues: list, preds: list, tokenizer: object) -> list:
    """Process molecules from predictions for new model.

    Args:
        trues (list): List of input ids corresponding to the original property.
        preds (list): Predictions for the properties.
        tokenizer (object): Used tokenizer to decode.
    """
    search = GreedySearch()
    
    new_trues = []
    new_preds = []

    for t, p in zip(trues, preds):
        p = torch.tensor(p)
        pred = search(p.unsqueeze(0)).squeeze(0)

        pred = tokenizer.decode(pred).split(' ')
        tru = tokenizer.decode(t).split(' ')

        new_trues.append(tru)
        new_preds.append(pred)

    return {'trues': new_trues, 'pred': new_preds}

def get_property_values(prop_data: Dict, tokenizer: BertTokenizer) -> Dict:
    """Process and extract the predicted properties."""
    search = GreedySearch()

    property_true = defaultdict(list)
    property_pred = defaultdict(list)
    
    for  inp, logit, label in zip(prop_data['inputs'], prop_data['logits'], prop_data['labels']):

        inp = torch.tensor(inp)
        logit = torch.tensor(logit)
        label = torch.tensor(label)
        
        pred = search(logit.unsqueeze(0)).squeeze(0)

        x_tokens = tokenizer.decode(inp).split(' ')
        y_tokens = tokenizer.decode(label).split(' ')
        y_hat_tokens = tokenizer.decode(pred).split(' ')

        label = tokenizer.get_sample_label(y_tokens, x_tokens)
        gen = tokenizer.get_sample_prediction(y_hat_tokens, x_tokens)

        _, target_prop = tokenizer.aggregate_tokens(
                    label, label_mode=True
                )
        
        _, gen_prop = tokenizer.aggregate_tokens(gen, label_mode=False)

        for k, v in target_prop.items():
            property_true[k].append(v)
        
        for k, v in gen_prop.items():
            property_pred[k].append(v)

    return {'trues': property_true, 'pred': property_pred}

def rmse_metric(prop_data: Dict, tokenizer: BertTokenizer, norm_params: Dict = None) -> float:
    """Calculate the rmse for provided property.

    Args:
        prop_dict (dict): property dict.
        tokenizer (obj:`ExpressionBertTokenizer`): tokenizer

    Returns:
        float: calculated rmse
    """

    prop_processed = get_property_values(prop_data, tokenizer)

    total_rmse = dict()
    for k in prop_processed['trues'].keys():
        trues = prop_processed['trues'][k]
        preds = prop_processed['pred'][k]

        if norm_params:
            if k in norm_params:
                np_min = norm_params[k]['min']
                np_max = norm_params[k]['max']
                trues = re_norm(trues, np_min, np_max)
                preds = re_norm(preds, np_min, np_max)

        rms_e = rmse(trues, preds)
        total_rmse[k] = rms_e

    return total_rmse

def pearson_metric(prop_data: Dict, tokenizer: BertTokenizer, norm_params: Dict = None) -> float:
    """Calculate the rmse for provided property.

    Args:
        prop_dict (dict): property dict.
        tokenizer (obj:`BertTokenizer`): tokenizer

    Returns:
        float: pearson
    """

    prop_processed = get_property_values(prop_data, tokenizer)

    total_pearson = defaultdict(dict)
    for k in prop_processed['trues'].keys():
        trues = prop_processed['trues'][k]
        preds = prop_processed['pred'][k]

        if norm_params:
            if k in norm_params:
                np_min = norm_params[k]['min']
                np_max = norm_params[k]['max']
                trues = re_norm(trues, np_min, np_max)
                preds = re_norm(preds, np_min, np_max)

        pearson, p_val = pearsonr(trues, preds)
        total_pearson[k] = {'pearson': pearson, 'p_val': p_val}

    return total_pearson

def get_molecules(mole_data: Dict, tokenizer: BertTokenizer) -> Dict:
    """Process and extract the predicted properties."""
    search = GreedySearch()
    
    gen_true = []
    gen_pred = []
    
    for inp, logit, label in zip(mole_data['inputs'], mole_data['logits'], mole_data['labels']):

        inp = torch.tensor(inp)
        logit = torch.tensor(logit)
        label = torch.tensor(label)

        pred = search(logit.unsqueeze(0)).squeeze(0)

        x_tokens = tokenizer.decode(inp).split(' ')
        y_tokens = tokenizer.decode(label).split(' ')
        y_hat_tokens = tokenizer.decode(pred).split(' ')

        label = tokenizer.get_sample_label(y_tokens, x_tokens)
        gen = tokenizer.get_sample_prediction(y_hat_tokens, x_tokens)

        label = label[label.index('|')+1:]
        gen = gen[gen.index('|')+1:]

        inp = tokenizer.decode(inp, clean_up_tokenization_spaces=False).split(' | ')[1]
        inp = inp.split(' ')
        mask = [i == tokenizer.mask_token for i in inp]

        gens = [gen[i] for i in range(len(gen)) if mask[i]]
        targets = [label[i] for i in range(len(label)) if mask[i]]

        gen_true.append(targets)
        gen_pred.append(gens)

    return {'trues': gen_true, 'pred': gen_pred}
    
def accuracy_metric(mole_data: Dict, tokenizer: BertTokenizer) -> float:
    """Calculate the rmse for provided property.

    Args:
        prop_dict (dict): property dict.
        tokenizer (obj:`BertTokenizer`): tokenizer

    Returns:
        float: calculated accuracy
    """

    mole_processed = get_molecules(mole_data, tokenizer)

    accuracy = []

    for t, g in zip(mole_processed['trues'], mole_processed['pred']):
        if len(t) == 0:
                continue
        accuracy.append(sum([a == b for a, b in zip(t, g)]) / len(t))

    return {'accuracy': sum(accuracy)/len(accuracy)}

def avg_edit_distance(mole_data: Dict, tokenizer: BertTokenizer) -> float:
    """Calculate the rmse for provided property.

    Args:
        prop_dict (dict): property dict.
        tokenizer (obj:`BertTokenizer`): tokenizer

    Returns:
        float: average edit distance
    """

    mole_processed = get_molecules(mole_data, tokenizer)

    edit_dist = []

    for t, g in zip(mole_processed['trues'], mole_processed['pred']):
        if len(t) == 0 or len(g) == 0:
            continue
        matches = [a == b for a, b in zip(t, g)]
        edit_dist.append(sum(matches))
        

    return {'edit_dist': sum(edit_dist)/len(edit_dist)}

def get_topk_molecules(mole_data: Dict, tokenizer: BertTokenizer, beam_width: int = 3) -> Dict:
    """Process and extract the predicted properties."""
    search = BeamSearch(beam_width=beam_width)
    
    gen_true = []
    gen_pred = []
    
    for  inp, logit, label in zip(mole_data['inputs'], mole_data['logits'], mole_data['labels']):

        inp = torch.tensor(inp)
        logit = torch.tensor(logit)
        label = torch.tensor(label)
        
        pred = search(logit.unsqueeze(0))[0].squeeze(0)

        x_tokens = tokenizer.decode(inp).split(' ')
        y_tokens = tokenizer.decode(label).split(' ')
        y_hat_tokens = []
        for t in pred:
            y_hat_tokens.append(tokenizer.decode(t).split(' '))

        label = tokenizer.get_sample_label(y_tokens, x_tokens)
        gen = tokenizer.get_sample_prediction(y_hat_tokens, x_tokens)

        label = label[label.index('|')+1:]
        gen = gen[gen.index('|')+1:]

        inp = tokenizer.decode(inp, clean_up_tokenization_spaces=False).split(' | ')[1]
        inp = inp.split(' ')
        mask = [i == tokenizer.mask_token for i in inp]

        gens = [gen[i] for i in range(len(gen)) if mask[i]]
        targets = [label[i] for i in range(len(label)) if mask[i]]

        gen_true.append(targets)
        gen_pred.append(gens)

    return {'trues': gen_true, 'pred': gen_pred}

def topk_accuracy_metric(mole_data: Dict, tokenizer: BertTokenizer, k: int = 3) -> float:
    """Calculate the rmse for provided property.

    Args:
        prop_dict (dict): property dict.
        tokenizer (obj:`BertTokenizer`): tokenizer.
        k (int): k for accuracy.

    Returns:
        float: calculated rmse
    """

    mole_processed = get_topk_molecules(mole_data, tokenizer, k)

    accuracy = []

    for t, g in zip(mole_processed['trues'], mole_processed['pred']):
        if len(t) == 0:
            continue
        acc = 0
        for i in range(len(t)):
            if t[i] in g[i]:
                acc += 1
        accuracy.append(acc / len(t))

    return {'top_k_accuracy': sum(accuracy)/len(accuracy), 'k': k}