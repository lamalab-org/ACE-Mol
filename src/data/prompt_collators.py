from typing import Dict, List, Union

import random

import torch
import numpy as np


class PromptPropertyCollator():
    """Collator for masking the property in the prompt property molecule.

    Data should be in the format: prompt [SEP] value [SEP] molecule.

    Args:
        tokenizer (obj`object`): tokenizer used to tokenize data, need to get special used tokens.
        device (str, optional): device to put the tensors on, defaults to 'cpu'.
        padd_same (bool, optional): if true padds the tensors to the same lenthg, defaults to True.
    """
    def __init__(self, tokenizer:object, device:str = 'cpu', padd_same:bool = True) -> None:
        self.tokenizer = tokenizer
        self.separator_token = tokenizer.bert_tokenizer.vocab[tokenizer.bert_tokenizer.sep_token]
        self.mask_token = tokenizer.bert_tokenizer.vocab[tokenizer.bert_tokenizer.mask_token]
        self.pad_token = tokenizer.bert_tokenizer.vocab[tokenizer.bert_tokenizer.pad_token]

        self.padd_same = padd_same

        self.device = device

    def bring_to_same_len(self, batch: Dict[str, List[Union[str, List]]]) -> Dict[str, List[Union[str, List]]]:
        """
        Pad the sequence to the same length. 
        """
        max_len = max([len(k) for k in batch['input_ids']])

        for i in range(len(batch['text'])):
            input_ids = batch['input_ids'][i]
            attention_mask = batch['attention_mask'][i]
            labels = batch['labels'][i]
            
            diff = max_len - len(input_ids)
            input_ids += [self.pad_token] * diff
            attention_mask += [0] * diff
            labels += [-100] * diff
            
            batch['input_ids'][i] = input_ids
            batch['attention_mask'][i] = attention_mask
            batch['labels'][i] = labels

            if 'token_type_ids' in batch.keys():
                token_ids = batch['token_type_ids'][i]
                token_ids += [0] * diff
                batch['token_type_ids'][i] = token_ids

        return batch

    def __call__(self, batch: Dict[str, List[Union[str, List]]]) -> Dict[str, Union[str, torch.Tensor]]:
        """Mask the property"""

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        l = len(batch['text'])
        batch['labels'] = []

        for i in range(l):
            input_ids = batch['input_ids'][i]
            attention_mask = batch['attention_mask'][i]
            labels = [-100] * len(input_ids)

            prop_pos = [i for i, v in enumerate(input_ids) if v == self.separator_token]
            prop_start = prop_pos[0]+1
            prop_end = prop_pos[1]

            for k in range(prop_start, prop_end):
                labels[k] = input_ids[k]
                input_ids[k] = self.mask_token
                attention_mask[k] = 0

            batch['input_ids'][i] = input_ids
            batch['attention_mask'][i] = attention_mask
            batch['labels'].append(labels)

        if self.padd_same:
            batch = self.bring_to_same_len(batch)

        for k in batch.keys():
            if k == 'text':
                continue
            batch[k] = torch.tensor(batch[k], dtype=torch.long)
            batch[k] = batch[k].to(self.device)

        return batch

class PromptMoleculeCollator(PromptPropertyCollator):
    """Collator for masking the property in the prompt property molecule.

    Data should be in the format: prompt [SEP] value [SEP] molecule.

    Args:
        tokenizer (obj`object`): tokenizer used to tokenize data, need to get special used tokens.
        mask_prob (float): probability of token getting masked.
        device (str, optional): device to put the tensors on, defaults to 'cpu'.
        padd_same (bool, optional): if true padds the tensors to the same lenthg, defaults to True.
    """
    def __init__(self, tokenizer:object, mask_prob:float, device:str = 'cpu', padd_same:bool = True) -> None:
        super().__init__(tokenizer, device, padd_same)

        self.mask_prob = mask_prob

    def get_indices_to_mask(self, indices: List) -> List:
        """Get indices to mask based on the probability."""
        to_mask = []

        for i in indices:
            if random.uniform(0, 1) < self.mask_prob:
                to_mask.append(i)
        return to_mask

    def get_batch_weights(self, labels: list) -> np.ndarray:
        """Get the weights of atoms per batch to balance the loss.
        
        Args:
            labels (list): original labels, aka tokens.

        Returns:
            np.ndarray: weights for each token.
        """
        labels = np.array(labels)

        masked_atoms = np.array(labels * (100 + labels))

        weights = np.zeros(masked_atoms.shape)

        for i in range(len(masked_atoms)):
            tokens, counts = np.unique(masked_atoms[i], return_counts=True)

            for token, count in zip(tokens, counts):
                if token == 0:
                    continue

                weights[masked_atoms == token] = 1 / count

        return weights

    def __call__(self, batch: Dict[str, List[Union[str, List]]]) -> Dict[str, Union[str, torch.Tensor]]:
        """Mask the property"""

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        l = len(batch['text'])
        batch['labels'] = []

        for i in range(l):
            input_ids = batch['input_ids'][i]
            attention_mask = batch['attention_mask'][i]
            labels = [-100] * len(input_ids)

            sep_pos = [i for i, v in enumerate(input_ids) if v == self.separator_token]
            mol_start = sep_pos[1]+1
            mol_end = sep_pos[2]
            idxs = list(range(mol_start, mol_end))
            mask_indices = self.get_indices_to_mask(idxs)
            
            for k in mask_indices:
                labels[k] = input_ids[k]
                input_ids[k] = self.mask_token
                attention_mask[k] = 0

            batch['input_ids'][i] = input_ids
            batch['attention_mask'][i] = attention_mask
            batch['labels'].append(labels)

        if self.padd_same:
            batch = self.bring_to_same_len(batch)

        batch['weights'] = self.get_batch_weights(batch['labels'])

        for k in batch.keys():
            if k == 'text':
                continue
            batch[k] = torch.tensor(batch[k], dtype=torch.long)
            batch[k] = batch[k].to(self.device)

        return batch
    
