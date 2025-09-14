# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import os
import logging
from datetime import datetime

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("Cannot import tensorboard. Will log to txt files only.")
    SummaryWriter = None

from utils.dist import is_primary


class Logger(object):
    def __init__(self, log_dir=None, tb_log_dir=None) -> None:
        self.log_dir = log_dir
        self.tb_log_dir = tb_log_dir
        time_str = datetime.now()
        time_str = time_str.strftime('%d-%m-%Y-%Hh-%Mm-%Ss')

        if SummaryWriter is not None and is_primary():
            self.tb_writer = SummaryWriter(self.tb_log_dir)
        else:
            self.tb_writer = None

        assert self.log_dir is not None
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.log = os.path.join(self.log_dir, time_str + '_log.txt')
        self.log_writer = self.create_logger()

    def log_scalars(self, scalar_dict, step, prefix=None):
        if self.tb_writer is None:
            return
        for k in scalar_dict:
            v = scalar_dict[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            if prefix is not None:
                k = prefix + k
            self.tb_writer.add_scalar(k, v, step)

    def create_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_handle = logging.FileHandler(filename=self.log)
        file_handle.setLevel(log_level)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
        return logger

    def info(self, information):
        self.log_writer.info(information)
