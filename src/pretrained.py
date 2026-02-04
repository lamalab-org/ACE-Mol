from typing import List
from pathlib import Path

import torch

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler

from transformers import AutoTokenizer, ModernBertForMaskedLM, AutoModelForMaskedLM, AutoConfig

from src.utils import to_dataset, embed_target_masked, generate_scaffold, scaffold_kfold_split
from src.tokenizer import BertSmilesTokenizer

torch.set_float32_matmul_precision('medium')

class PretrainedACEMol():
    """ACEMol wrapper for easier use of pre-trained model.
    
    PretrainedACEMol can load either pre-trained models from hf or checkpoint file.

    Args:
        path (str | Path, optional): huggingface path or local .ckpt file.
            defaults to ACEMol-light.
        device (str, optional): cpu or cuda. defaults to cpu.
    """
    
    def __init__(
        self,
        path: str | Path = 'jablonkagroup/ACEMol-light',
        device: str = 'cpu'
    ) -> None:

        self.device = device
        # Use just one gpu
        if device.startswith('cuda'):
            self.device = 'cuda:0'

        if path.endswith('.ckpt'):
            base_model = 'jablonkagroup/ACEMol'
            
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    
            new_state_dict = dict()
            
            for k, v in state_dict.items():
                if "model.model." in k:
                    k = k.replace("model.model.", "model.")
                elif "model." in k:
                    k = k.replace("model.", "")

                k = k.replace("net.", "")

                new_state_dict[k] = v
            
            config = AutoConfig.from_pretrained(base_model)
            self.model = AutoModelForMaskedLM.from_config(config).to(self.device)
    
            self.model.load_state_dict(new_state_dict)
        else:
            self.model = ModernBertForMaskedLM.from_pretrained(path, torch_dtype='auto').to(self.device)

        self.tokenizer = BertSmilesTokenizer()

    def classify(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        multiclass: bool = False
    ) -> pd.DataFrame:
        """Classification prediction on embeddings.

        Args:
            train (pd.DataFrame): training dataset containing embeddings.
            test (pd.DataFrame): test dataset containing embeddings.
            multiclass (bool): if true use multiclass classifier.

        Returns:
            pd.DataFrame: test table with additional per class probabilities.
        """
        train = train.copy()
        test = test.copy()
        x = train['embeddings'].values.tolist()
        y = train['target'].values.tolist()
    
        lambda_theory = np.sqrt(2 * np.log(len(x[0])) / len(x))
    
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ("clf", LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1 / lambda_theory,
                class_weight="balanced",
            )),
        ])
    
        if multiclass:
            pipe = Pipeline([
                ('scaler', MinMaxScaler()),
                ("clf", OneVsRestClassifier(
                    LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=1 / lambda_theory,
                    class_weight="balanced",
                ))),
            ])
    
        pipe.fit(x, y)
    
        x_test = test['embeddings'].values.tolist()
    
        probs = pipe.predict_proba(x_test)
    
        for i in range(len(pipe.classes_)):
            test[f'class_probability_{pipe.classes_[i]}'] = probs[:, i].tolist()
    
        return test
    
    def regress(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> pd.DataFrame:
        """Regression prediction on embeddings.

        Args:
            molecules (str | List): one or more SMILES string.
            tasks (str | List): one or more tasks, if there are multiple
                molecules same task is used for all of them.

        Returns:
            pd.DataFrame: test table with additional predictions column.
        """
        train = train.copy()
        test = test.copy()
        x = train['embeddings'].values.tolist()
        y = train['target'].values.tolist()
    
        lambda_theory = np.sqrt(2 * np.log(len(x[0])) / len(x))
    
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ("reg", Lasso(1/lambda_theory)),
        ])
        
        pipe.fit(x, y)
    
        x_test = test['embeddings'].values.tolist()
    
        probs = pipe.predict(x_test)
    
        test['prediction'] = probs.tolist()
    
        return test

    def embed(
        self,
        molecules: List,
        tasks: str | List,
        targets: List,
        batch_size: int = 1
    ) -> pd.DataFrame:
        """Embedding of the molecules. The target is always masked to ensure no data leakage.

        Args:
            molecules (List): one or more SMILES string.
            tasks (str | List): one or more tasks, if there are multiple.
                molecules same task is used for all of them.
            targets (List): list of corresponding targets.
            batch_size (int, optional): batch size to use while embedding. defaults to 1.

        Returns:
            pd.DataFrame: table containing task description, SMILES and embedding.
        """
        data = to_dataset(molecules, tasks, targets)

        embeddings = embed_target_masked(data, self.model, self.tokenizer, self.device, batch_size)
        
        return embeddings