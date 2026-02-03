from typing import Dict, List, Union

import torch

class AlternatingCollator():
    """Alternates between two collator"""

    def __init__(
        self,
        collator: Dict,
        alter: Dict,
        steps: int = 8
    ) -> None:
        """Alternating collator constructor.

        Args:
            collator (:obj:`Dict`): Base collator and it's name.
            alter (:obj:`Dict`): Alternating collator and it's name.
            steps (int): Switch collator every n steps
        """
        
        self.collator = collator
        self.alter = alter
        self.max_steps = steps
        self.current_step = 0
        self.flag = True

        self.tokenizer = collator['collator'].tokenizer

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Collatorate the batch
        
        Args:
            examples (list): our input data from dataloader.

        Returns:
            dict: collatorated batch
        """

        if self.current_step % self.max_steps == 0:
            self.flag = not self.flag

        if self.flag:
            outs = self.collator['collator'](examples)
            outs['mode'] = self.collator['name']
        else:
            outs = self.alter['collator'](examples)
            outs['mode'] = self.alter['name']

        return outs

        
