from pathlib import Path
from typing import Dict, Union, Any
from collections import defaultdict

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForMaskedLM

class WrappedModel(nn.Module):
    """Model wrapper to include the numerical encoding."""

    def __init__(self) -> None:
        super().__init__()
        
        base_model = 'jablonkagroup/ACEMol'
            
        config = AutoConfig.from_pretrained(base_model)
        
        self.model = AutoModelForMaskedLM.from_config(config=config)

    def forward(
        self, 
        inputs: Dict[str, Union[torch.Tensor, Any]],
        output_hidden_states: bool = False
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """Handle the numerical encodings"""
        
        model_inputs = inputs.copy()
        if 'text' in model_inputs.keys():
            model_inputs.pop('text') 
        if 'mode' in model_inputs.keys():
            model_inputs.pop('mode')

        outputs = self.model(**model_inputs, output_hidden_states=output_hidden_states)

        return outputs