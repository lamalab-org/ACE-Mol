from pathlib import Path

from typing import List, Dict

from transformers import AutoTokenizer

from datasets.formatting.formatting import LazyBatch

from src.utils.consts import SMILES_TOKENIZER_PATTERN

from src.tokenizer.regex_tokenizer import RegexTokenizer

class BertSmilesTokenizer():
    """
    Bert tokenizer extended wit smiles vocab.
    """
    def __init__(self) -> None:
        self.separator = ' | '
        self.bert_tokenizer = AutoTokenizer.from_pretrained('jablonkagroup/ACEMol')

        self.smiles_parser = RegexTokenizer(regex_pattern=SMILES_TOKENIZER_PATTERN)

    def __call__(self, inp: str) -> Dict:
        return self.encode(inp)

    def encode(self, inp: str) -> Dict:
        """Tokenize the task | value | smiles | smiles (optional) for the model.

        Args: 
            inp (str): task | value | smiles | smiles (optional).

        Returns:
            dict: tokenized object.
        """

        if type(inp) == str:
            if self.separator in inp:
                inp = inp.split(self.separator)
                inp[2] = ' ' + ' '.join(self.smiles_parser.tokenize(inp[2]))
                if len(inp) > 3:
                    inp[3] = ' ' + ' '.join(self.smiles_parser.tokenize(inp[3]))
                inp = self.bert_tokenizer.sep_token.join(inp)
            else:
                inp = ' ' + ' '.join(self.smiles_parser.tokenize(inp))

            out = self.bert_tokenizer(inp)
        else: 
            out = []
            for el in inp['text']:
                if self.separator in el:
                    el = el.split(self.separator)
                    el[2] = ' '.join(self.smiles_parser.tokenize(el[2]))
                    if len(el) > 3:
                        el[3] = ' '.join(self.smiles_parser.tokenize(el[3]))
                    el = self.bert_tokenizer.sep_token.join(el)
                else:
                    el = ' ' + ' '.join(self.smiles_parser.tokenize(el))

                out.append(self.bert_tokenizer(el))
            out = {k: [dic[k] for dic in out] for k in out[0]}
             
        return out
        
    def decode(self, inp: List) -> str:
        """Decode tokenized str.

        Args: 
            inp (list): tokenized int list.

        Returns:
            str: decoded str.
        """

        dec = self.bert_tokenizer.decode(inp)

        return dec