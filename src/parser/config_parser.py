import os
import yaml
import argparse

from typing import Dict
from pathlib import Path

def parse_config() -> Dict:
    """
    Load and parse the config file.

    Returns:
        Dict: config file
    """

    parser = argparse.ArgumentParser(description="Train argument parser...")

    parser.add_argument('-c', '--config', help='Path to config file.')
    parser.add_argument('-p', '--pretrained', help='Path to the pretrained model (optional).', default=None)
    parser.add_argument('-f', '--freeze', help='If we freeze the model or not.', default=None)

    # Model args
    parser.add_argument('-m', '--model-name', help='Model name.')

    # Data params
    parser.add_argument('-b', '--batch-size', help='Set batch size.')
    parser.add_argument('-s', '--steps', help='Set number of alternating steps.')
    parser.add_argument('-e', '--eval-data-file', help='Location to the eval test file.')
    parser.add_argument('-t', '--train-data-file', help='Location to the train data file.')
    parser.add_argument('-v', '--val-data-file', help='Location to the validation data file.')
    parser.add_argument('-r', '--rebalance', help='Rebalance the train set each epoch', default=False)

    # Trainer Params
    parser.add_argument('-a', '--accelerator', help='Select gpu or cpu.')
    parser.add_argument('-st', '--strategy', help='Set training strategy.')
    parser.add_argument('-d', '--devices', help='Number of devices')

    args = parser.parse_args()

    # Config sanity check
    assert(args.config is not None, 'The config file has not been provided')

    # Load and return config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config['cnf_file'] = args.config
    config['rebalance'] = args.config

    config['model']['pre_trained'] = args.pretrained
    config['model']['freeze'] = args.freeze

    dct = vars(args)
    dct = {k: v for k, v in dct.items() if v is not None}

    config['model']['model_name'] = dct.get('model_name', config['model']['model_name'])

    config['data']['batch_size'] = int(dct.get('batch_size', config['data']['batch_size']))
    config['data']['steps'] = dct.get('steps', config['data']['steps'])
    config['data']['eval_data_file'] = dct.get('eval_data_file', config['data']['eval_data_file'])
    config['data']['val_data_file'] = dct.get('val_data_file', config['data']['val_data_file'])
    config['data']['train_data_file'] = dct.get('train_data_file', config['data']['train_data_file'])

    config['train']['trainer']['accelerator'] = dct.get('accelerator', config['train']['trainer']['accelerator'])

    if 'strategy' in dct.keys():
        config['train']['trainer']['strategy'] = dct['strategy']

    if 'devices' in dct.keys():
        config['train']['trainer']['devices'] = dct['devices']

    return config
