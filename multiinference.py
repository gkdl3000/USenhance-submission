#!/usr/bin/python3
import os
import numpy as np
import random
import argparse
from trainer import Cyc_Trainer
import yaml
import torch

def setseed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGan_finetune.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)

    trainer.multiinference()
    
    
###################################
if __name__ == '__main__':
    random_seed = 1234
    setseed(random_seed)
    main()