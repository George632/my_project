# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import sys
import pickle

import numpy as np
import torch
import shutil
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# WFTR
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible
from utils.logger import Logger
from utils import config


def make_args_parser():
    parser = argparse.ArgumentParser("Wireframe Reconstruction Using Transformers", add_help=False)

    # Batch Size
    parser.add_argument('--batch_size', default=20, type=int, help='batch size per gpu')
    parser.add_argument('--num_workers', default=4, type=int, help='Dataset num workers')

    # Config
    parser.add_argument('--config', default="./config/WFTR.yaml", type=str, help='config file')

    # Training
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # Testing
    parser.add_argument("--test_only", default=True, action="store_true")
    parser.add_argument("--test_ckpt", default='./checkpoint_best/checkpoint_best.pth', type=str)

    # I/O
    parser.add_argument("--log_dir", default='./log', type=str)
    parser.add_argument("--checkpoint_dir", default='./checkpoint', type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

    # Distributed Training
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)
    args = parser.parse_args()

    return args


def initial_config(args):
    assert args.config is not None
    cfg = config.load_yaml_config(args.config)
    cfg.update(vars(args))

    cfg.logger = Logger(cfg.log_dir, cfg.checkpoint_dir)
    cfg.logger.info('------------------------------------- Start Logging -------------------------------------')
    config.log_config_to_file(cfg, logger=cfg.logger)
    return cfg


def train_model(
        cfg,
        model,
        model_no_ddp,
        optimizer,
        criterion,
        dataloaders,
        best_val_metrics,
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    We always evaluate the final checkpoint and report both the final AP and best AP on the val set.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    cfg.logger.info(f"Model is {model}")
    cfg.logger.info(f"Training started at epoch {cfg.start_epoch} until {cfg.max_epoch}.")
    cfg.logger.info(f"One training epoch = {num_iters_per_epoch} iters.")
    cfg.logger.info(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(cfg.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(cfg.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        cfg.logger.info(f"Found final eval file {final_eval}. Skipping training.")
        return

    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)

        aps = train_one_epoch(
            cfg,
            epoch,
            model,
            optimizer,
            criterion,
            dataloaders["train"],
            cfg.logger,
        )

        # latest checkpoint is always stored in checkpoint.pth
        save_checkpoint(
            cfg.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        aps.compute_metrics()
        metric_str = aps.metrics_to_str()
        metrics_dict = aps.metrics_to_dict()
        curr_iter = epoch * len(dataloaders["train"])
        if is_primary():
            cfg.logger.info("==" * 10)
            cfg.logger.info(f"Epoch [{epoch}/{cfg.max_epoch}]; Metrics {metric_str}")
            cfg.logger.info("==" * 10)
            cfg.logger.log_scalars(metrics_dict, curr_iter, prefix="Train/")

        if (
                epoch > 0
                and cfg.save_separate_checkpoint_every_epoch > 0
                and epoch % cfg.save_separate_checkpoint_every_epoch == 0
        ):
            # separate checkpoints are stored as checkpoint_{epoch}.pth
            save_checkpoint(
                cfg.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                best_val_metrics,
            )

        if epoch % cfg.eval_every_epoch == 0 or epoch == (cfg.max_epoch - 1):
            ap_calculator = evaluate(
                cfg,
                epoch,
                model,
                criterion,
                dataloaders["test"],
                cfg.logger,
                curr_iter,
            )
            metrics = ap_calculator.compute_metrics(return_ap_dict=True)
            ap05 = metrics['0.05_edges_f1']
            metric_str = ap_calculator.metrics_to_str()
            metrics_dict = ap_calculator.metrics_to_dict()
            if is_primary():
                cfg.logger.info("==" * 10)
                cfg.logger.info(f"Evaluate Epoch [{epoch}/{cfg.max_epoch}]; Metrics {metric_str}")
                cfg.logger.info("==" * 10)
                cfg.logger.log_scalars(metrics_dict, curr_iter, prefix="Test/")

            if is_primary() and (
                    len(best_val_metrics) == 0 or best_val_metrics['0.05_edges_f1'] < ap05
            ):
                best_val_metrics = metrics
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    cfg.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    best_val_metrics,
                    filename=filename,
                )
                cfg.logger.info(
                    f"Epoch [{epoch}/{cfg.max_epoch}] saved current best val checkpoint at {filename}; ap05 {ap05}"
                )

    # always evaluate last checkpoint
    epoch = cfg.max_epoch - 1
    curr_iter = epoch * len(dataloaders["train"])
    ap_calculator = evaluate(
        cfg,
        epoch,
        model,
        criterion,
        dataloaders["test"],
        cfg.logger,
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics(return_ap_dict=True)
    metric_str = ap_calculator.metrics_to_str()
    if is_primary():
        cfg.logger.info("==" * 10)
        cfg.logger.info(f"Evaluate Final [{epoch}/{cfg.max_epoch}]; Metrics {metric_str}")
        cfg.logger.info("==" * 10)

        with open(final_eval, "w") as fh:
            fh.write("Training Finished.\n")
            fh.write("==" * 10)
            fh.write("Final Eval Numbers.\n")
            fh.write(metric_str)
            fh.write("\n")
            fh.write("==" * 10)
            fh.write("Best Eval Numbers.\n")
            fh.write(ap_calculator.metrics_to_str(best_val_metrics))
            fh.write("\n")

        with open(final_eval_pkl, "wb") as fh:
            pickle.dump(metrics, fh)


def test_model(cfg, model, model_no_ddp, criterion, dataloaders):
    if cfg.test_ckpt is None or not os.path.isfile(cfg.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {cfg.test_ckpt}"
        sys.exit(1)

    sd = torch.load(cfg.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    # criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ap_calculator = evaluate(
        cfg,
        epoch,
        model,
        criterion,
        dataloaders["test"],
        cfg.logger,
        curr_iter,
    )
    ap_calculator.compute_metrics(save_wireframe=False)
    metric_str = ap_calculator.metrics_to_str()
    if is_primary():
        print("==" * 10)
        print(f"Test model; Metrics {metric_str}")
        print("==" * 10)


def main(local_rank, cfg):
    # --------------------------------- multiple GPU Training -- Just Test ---------------------------------------
    if cfg.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print(
            "To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=cfg.ngpus,
            dist_url=cfg.dist_url,
            dist_backend="nccl",
        )

    # --------------------------------- Set Seed to Keep Training Stable -------------------------------------------
    torch.cuda.set_device(local_rank)
    np.random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + get_rank())

    # --------------------------------- Build Dataset, WFTR Model, Criterion, and Optimizer Model -------------------
    # Dataset Model
    datasets = build_dataset(cfg.Dataset)
    dataloaders = {}
    if cfg.test_only:
        dataset_splits = ["test"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=datasets[split].collate_batch
        )
        dataloaders[split + "_sampler"] = sampler

    # WFTR Model
    model, _ = build_model(cfg)
    model = model.cuda(local_rank)
    model_no_ddp = model
    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    # Criterion Model
    criterion = build_criterion(cfg)
    criterion = criterion.cuda(local_rank)

    # Optimizer Model
    optimizer = build_optimizer(cfg, model_no_ddp)

    # ------------------------------------------- Run Train or Test Model -----------------------------------------
    if cfg.test_only:
        # criterion = None  # faster evaluation
        test_model(cfg, model, model_no_ddp, criterion, dataloaders)
    else:
        assert (
                cfg.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(cfg.checkpoint_dir):
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        # copy important python files
        shutil.copy('./eval/ap_calculator.py', cfg.checkpoint_dir)
        shutil.copy('./models/model_wftr.py', cfg.checkpoint_dir)
        shutil.copy('./config/WFTR.yaml', cfg.checkpoint_dir)
        shutil.copy('./utils/wireframe_util.py', cfg.checkpoint_dir)
        shutil.copy('criterion.py', cfg.checkpoint_dir)
        shutil.copy('main.py', cfg.checkpoint_dir)
        # resume model
        loaded_epoch, best_val_metrics = resume_if_possible(
            cfg.checkpoint_dir, model_no_ddp, optimizer
        )
        cfg.start_epoch = loaded_epoch + 1
        train_model(
            cfg,
            model,
            model_no_ddp,
            optimizer,
            criterion,
            dataloaders,
            best_val_metrics,
        )


def launch_distributed(cfg):
    world_size = cfg.ngpus
    if world_size == 1:
        main(local_rank=0, cfg=cfg)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(cfg,))


if __name__ == "__main__":
    # --------------------------------- Loading Initial Config -------------------------------------------
    args = make_args_parser()
    cfg = initial_config(args)

    # --------------------------------- Run Single/Parallel Model ----------------------------------------
    try:
        # multiple process -- linux/macos: spawn, windows: fork, linux: forkserver
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(cfg)

    # -------------------------------------------- Done ---------------------------------------------------
    cfg.logger.info("---------------------------------------- Done !!! ---------------------------------------------")
