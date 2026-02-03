import os
import yaml
import shutil
import json

from pathlib import Path
from typing import Dict, Any

def setup_model_dir(root_dir:str, model_name:str, cnf_file:str, config:Dict[str, Any]) -> bool:
    """
    Setup model dir, create folder structure for each trained model
    to keep the logs, copy of the config and more.

    Args:
        root_dir (str): Root directory where we have all of our models.
        model_name (str): Name of the model, to create the structure in.
        cnf_file(str): Location of the config file.
        config (:obj:`Dict[str, Any]`): Configs.

    Returns:
        bool: True if model training exists
    """

    pth = Path(root_dir) / Path(model_name)

    # Model exists, notify that we will load and continue from latest checkpoint
    if os.path.exists(pth):
        ckp = pth / Path('checkpoints')
        if os.path.exists(ckp) and len(os.listdir(ckp)) != 0:
            return True

    # Create the directory structure
    pth.mkdir(parents=True, exist_ok = True)

    os.makedirs(pth / Path('logs'), exist_ok = True)
    os.makedirs(pth / Path('figures'), exist_ok = True)
    os.makedirs(pth / Path('evals'), exist_ok = True)
    os.makedirs(pth / Path('checkpoints'), exist_ok = True)

    # Copy config file
    cnf_yml = pth / Path('config.yaml')
    with open(cnf_yml, 'w') as f:
        yaml.dump(config, f)

    model_params = config['model']['parameters']

    # Copy just the model param file as JSON for transformers.AutoConfig
    with open(pth / Path('model.json'), 'w') as f:
        json.dump(model_params, f)

    return False
        