#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-21 9:23 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : config.py.py
# @IDE     : PyCharm
import yaml
from easydict import EasyDict


def convert_scientific_to_float(data):
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = convert_scientific_to_float(value)
        elif isinstance(value, str) and 'e' in value.lower():
            try:
                data[key] = float(value)
            except ValueError:
                pass  # If the conversion fails, keep the original value
    return data


def load_yaml_config(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
            config = convert_scientific_to_float(config)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            config = {}

    cfg = EasyDict(config)
    return cfg


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, value in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, value))